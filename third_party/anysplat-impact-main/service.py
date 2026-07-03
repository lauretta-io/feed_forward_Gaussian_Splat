from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import sys
import cv2
import numpy as np
from pydantic import BaseModel

import base64
import httpx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_cv_image
from save_ply import save_gaussians_simple
from qwen_description import VideoAnalyzer

import time

app = FastAPI(title="AnySplat Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Global variables for model and latest result
model = None
device = None
qwen_analyzer = None
latest_result = None
glb_output_path = "/home/lauretta/quang/impact/ImpactComponentFrontEnd/output"
video_path = "/home/lauretta/quang/impact/AnySplat_impact/video_output_path"
# caps = [cv2.VideoCapture("rtsp://192.168.1.190:554/stream/main"),
#         cv2.VideoCapture("rtsp://192.168.1.192:554/stream/main")]

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model, device, qwen_analyzer
    
    print("Loading AnySplat model...")
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model loaded successfully on {device}")
    qwen_analyzer = VideoAnalyzer(model_name="Qwen/Qwen2-VL-2B-Instruct")
    print("Qwen Video Analyzer initialized")

def base64_to_cv2(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    # Remove the data:image/jpeg;base64, prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 to bytes
    img_bytes = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    
    # Decode to OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

class ImageRequest(BaseModel):
    camera1_image: str
    camera2_image: str

@app.post("/images")
async def process_images(request: ImageRequest):
    """
    POST endpoint to process 2 images
    Expects exactly 2 image files
    """
    global latest_result
    begin = time.time()

    try:
        # Read and process uploaded images
        frames = []
        # Process camera1_image
        try:
            frame1 = base64_to_cv2(request.camera1_image)

            if frame1 is None:
                raise ValueError("Could not decode camera1_image")

            frames.append(frame1)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error decoding camera1_image: {str(e)}"
            )
        # Process camera2_image
        try:
            frame2 = base64_to_cv2(request.camera2_image)
            if frame2 is None:
                raise ValueError("Could not decode camera2_image")
            frames.append(frame2)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error decoding camera2_image: {str(e)}"
            )
        images_captured_time = time.time() - begin
        
        async with httpx.AsyncClient() as client:
            try:
                timestamp_payload = {
                    "id": 0,
                    "time": (time.time() - begin), 
                    "analysis_id": "image-capturing"
                }
                response = await client.post(
                    "http://localhost:3001/api/record-timestamp",
                    json=timestamp_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0  # optional timeout
                )
                
                # Check response
                if response.status_code == 200:
                    external_result = response.json()
                    print(f"External API response: {external_result}")
                else:
                    print(f"External API error: {response.status_code}")
            except httpx.ConnectError as e:
                print(f"✗ Connection failed - is the server running? {e}")
            except Exception as e:
                print(f"✗ Request failed: {type(e).__name__} - {e}")
        # begin = time.time()
        # Process images through the model
        processed_images = [process_cv_image(frame) for frame in frames]
        images_tensor = torch.stack(processed_images, dim=0).unsqueeze(0).to(device)  # [1, 2, 3, 448, 448]
        # processing_time = time.time() - begin
        async with httpx.AsyncClient() as client:
            timestamp_payload = {
                "id": 2,
                "time": (time.time() - begin), 
                "analysis_id": "anysplat-processing"
            }
            response = await client.post(
                "http://localhost:3001/api/record-timestamp",
                json=timestamp_payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0  # optional timeout
            )
            
            # Check response
            if response.status_code == 200:
                external_result = response.json()
                print(f"External API response: {external_result}")
            else:
                print(f"External API error: {response.status_code}")

        begin = time.time()
        b, v, _, h, w = images_tensor.shape
        
        # Run inference
        with torch.no_grad():
            gaussians, pred_context_pose = model.inference((images_tensor + 1) * 0.5)
        gaussian_splatting_time = time.time() - begin
        async with httpx.AsyncClient() as client:
            timestamp_payload = {
                "id": 1,
                "time": (time.time() - begin), 
                "analysis_id": "anysplat-processing"
            }
            response = await client.post(
                "http://localhost:3001/api/record-timestamp",
                json=timestamp_payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0  # optional timeout
            )
            
            # Check response
            if response.status_code == 200:
                external_result = response.json()
                print(f"External API response: {external_result}")
            else:
                print(f"External API error: {response.status_code}")

        # Save output
        # output_path = "glb_output/output.glb"
        os.makedirs(glb_output_path, exist_ok=True)
        save_gaussians_simple(gaussians, os.path.join(glb_output_path, "output.glb"))
        # save_gaussians_to_ply(gaussians, os.path.join(glb_output_path, "output.ply"), binary=True)

        async with httpx.AsyncClient() as client:
            reload_payload = {
                "folder_name": "home/lauretta/quang/impact/AnySplat_impact/glb_output",
            }
            response = await client.post(
                "http://localhost:3001/api/refresh-model",
                json=reload_payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0  # optional timeout
            )
            
            # Check response
            if response.status_code == 200:
                external_result = response.json()
                print(f"External API response: {external_result}")
            else:
                print(f"External API error: {response.status_code}")

        pred_all_extrinsic = pred_context_pose['extrinsic']
        pred_all_intrinsic = pred_context_pose['intrinsic']
        save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, video_path, model.decoder)
        begin = time.time()
        analysis = qwen_analyzer(f"{video_path}/rgb.mp4")
        print(f"Environment: {analysis['environment']}")
        print(f"Number of People: {analysis['number_of_people']}")
        print(f"Activities: {analysis['activities']}")
        print(f"Threats: {analysis['threats']}")
        print(f"Is Anomaly: {analysis['is_anomaly']}")
        print(f"Anomaly Reason: {analysis['anomaly_reason']}")
        processing_time = time.time() - begin
        # Store result for GET endpoint
        latest_result = {
            "status": "success",
            "ply_file_url": f"http://localhost:3001/output/output.glb",
            'gaussian_splatting_time': int(gaussian_splatting_time * 1000),  # Convert to milliseconds
            'images_captured_time': int(images_captured_time * 1000),  # Convert to milliseconds
            'processing_time': int(processing_time * 1000),  # Convert to milliseconds
            "people_count": analysis['number_of_people'],
            "environment": analysis['environment'],
            "activity": analysis['activities'],
            "threats": analysis['threats'],
            "is_anomaly": analysis['is_anomaly'],
            "anomaly_reason": analysis['anomaly_reason'],   
            "num_images_processed": len(frames),
            "image_shape": [h, w],
            "device": str(device),
            "gaussians_shape": {
                "means": list(gaussians["means"].shape),
                "scales": list(gaussians["scales"].shape),
                "rotations": list(gaussians["rotations"].shape),
            } if isinstance(gaussians, dict) else "N/A"
        }
        
        return JSONResponse(
            content=latest_result,
            status_code=200
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/result")
async def get_result():
    """
    GET endpoint to retrieve the latest processing result
    """
    if latest_result is None:
        return JSONResponse(
            content={
                "status": "no_data",
                "message": "No images have been processed yet"
            },
            status_code=200
        )
    
    return JSONResponse(content=latest_result, status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3005)