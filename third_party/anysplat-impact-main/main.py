from pathlib import Path
import torch
import os
import sys
from time import time, sleep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image, process_cv_image
from save_ply import save_gaussians_to_ply, save_gaussians_simple, inspect_gaussians
from stream_ply import stream_glb
import cv2
import numpy as np

import asyncio
import websockets
import json
import base64
from collections import deque
import threading

class GaussianSplattingStreamer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
    
    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def send_gaussian_data(self, glb_data, frame_id):
        """Send .glb file as base64 encoded string"""
        if self.clients:
            # Encode binary .glb data to base64
            encoded_data = base64.b64encode(glb_data).decode('utf-8')
            
            message = json.dumps({
                'type': 'gaussian_splat',
                'frame_id': frame_id,
                'data': encoded_data,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # Broadcast to all connected clients
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket, path):
        await self.register(websocket)
    
    async def start_server(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

# Usage in your processing loop
async def glb_streamer(gaussian_storage):
    streamer = GaussianSplattingStreamer()
    
    # Start server in background
    server_task = asyncio.create_task(streamer.start_server())
    
    frame_id = 0
    while True:
        # Your Gaussian Splatting processing
        glb_data = gaussian_storage.get_latest_glb_data()  # Returns binary .glb data
        if glb_data is not None:
            # Stream to frontend
            await streamer.send_gaussian_data(glb_data, frame_id)
        
            frame_id += 1
        await asyncio.sleep(0.5)  # 2 FPS

class GaussianStorage:
    def __init__(self, max_queue=3):
        self.max_size = max_queue
        self.storage = deque(maxlen=self.max_size)
        
    def add_gaussian(self, gaussian_data):
        if len(self.storage) >= self.max_size:
            self.storage.popleft()
        self.storage.append(gaussian_data)
    
    def get_latest_glb_data(self):
        if self.storage:
            return self.storage[-1]
        return None

if __name__ == "__main__":
    # Load the model from Hugging Face
    video_urls = ["rtsp://admin:lauretta123456!@192.168.1.187:554/unicast/c8/s0/live",
                  "rtsp://admin:lauretta123456!@192.168.1.187:554/unicast/c4/s0/live"]
    # gauss_storage = GaussianStorage(max_queue=3)

    # def run_websocket_server():
    #     asyncio.run(glb_streamer(gauss_storage))
    
    # websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    # websocket_thread.start()
    def save_new_gaussian_data(gaussians):
        if os.path.exists("glb_output/locked_output.glb"):
            os.rename("glb_output/locked_output.glb", "glb_output/stream_output.glb")
        save_gaussians_simple(gaussians, "glb_output/locked_output.glb")

    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    caps = []
    for video_string in video_urls:
        caps.append(cv2.VideoCapture(video_string))
    
    begin = time()
    frame_run = 0
    while True:
        frames = [cap.read()[1] for cap in caps]
        if any(frame is None for frame in frames):
            print("One of the video streams has ended or cannot be read.")
            continue
        
        frame_run += 1
        images = [process_cv_image(frame) for frame in frames]
        images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
        b, v, _, h, w = images.shape

        gaussians, pred_context_pose = model.inference((images+1)*0.5)
        save_new_gaussian_data(gaussians)
        # save_gaussians_simple(gaussians, f"output_{frame_run}.glb")

        if frame_run % 10 == 0:
            current_time = time()
            print(f"FPS: {frame_run / (current_time - begin):.2f}")

        sleep(0.1)