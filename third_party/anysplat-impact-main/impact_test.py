from pathlib import Path
import torch
import os
import sys
from time import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
from save_ply import save_gaussians_to_ply, save_gaussians_simple, inspect_gaussians
import cv2
import numpy as np


# Load the model from Hugging Face
video_urls = ["rtsp://192.168.1.190:554/stream/main",
              "rtsp://192.168.1.192:554/stream/main"]
model = AnySplat.from_pretrained("lhjiang/anysplat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

caps = []
for video_string in video_urls:
   caps.append(cv2.VideoCapture(video_string))

image_paths = []
frames = [cap.read()[1] for cap in caps]
output_images = 'tmp_images'
for i, frame in enumerate(frames):
    if frame is not None:
        image_path = os.path.join(output_images, f"{i}.jpg")
        cv2.imwrite(image_path, frame)
        image_paths.append(image_path)

images = [process_image(frame) for frame in image_paths]
images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
b, v, _, h, w = images.shape
print("running inference now")
# Run Inference
begin = time()
gaussians, pred_context_pose = model.inference((images+1)*0.5)
print("running postprocess now")
pred_all_extrinsic = pred_context_pose['extrinsic']
pred_all_intrinsic = pred_context_pose['intrinsic']
end = time()
print(f"Inference time: {end - begin:.2f} seconds")
save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, "video_output_path", model.decoder)
end2 = time()
print(f"Total time (inference + rendering): {end2 - begin:.2f} seconds")


# 1. Save as PLY (full 3DGS format - best quality)
print("\n[1/3] Saving PLY format (full Gaussian Splatting data)...")
save_gaussians_to_ply(gaussians, "output.ply", binary=True)

# 2. Save as GLB (point cloud - for standard 3D viewers)
print("\n[2/3] Saving GLB format (point cloud representation)...")
save_gaussians_simple(gaussians, "output.glb")
