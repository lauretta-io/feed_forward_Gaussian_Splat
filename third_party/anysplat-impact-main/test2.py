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

def save_gaussians_to_gltf(gaussians, output_path, use_glb=True):
    """
    Convert Gaussians directly to glTF/GLB format.
    
    WARNING: This converts Gaussians to a point cloud and loses:
    - Gaussian scales
    - Gaussian rotations
    - Spherical harmonics (except DC/color)
    - Opacity information
    
    For full 3DGS rendering, use PLY -> .splat conversion instead.
    """
    try:
        from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene, Material, PbrMetallicRoughness
    except ImportError:
        print("Error: pygltflib not installed")
        print("Install with: pip install pygltflib")
        return
    
    print(f"\nConverting Gaussians to {'GLB' if use_glb else 'glTF'} point cloud...")
    print("WARNING: Gaussian properties (scales, rotations, SH) will be lost!")
    print("This is only a point cloud representation.\n")
    
    # Extract positions
    if hasattr(gaussians, 'means'):
        means = gaussians.means.detach().cpu().numpy()
    elif hasattr(gaussians, 'xyz'):
        means = gaussians.xyz.detach().cpu().numpy()
    else:
        raise AttributeError("Cannot find position attribute")
    
    # Remove batch dimension if present
    if means.ndim == 3 and means.shape[0] == 1:
        means = means[0]
    
    positions = means.astype(np.float32)
    
    # Extract colors from harmonics
    colors = None
    if hasattr(gaussians, 'harmonics'):
        harmonics = gaussians.harmonics.detach().cpu().numpy()
        # Remove batch dimension
        if harmonics.ndim == 4 and harmonics.shape[0] == 1:
            harmonics = harmonics[0]  # [N, 3, 25]
        
        # Extract DC component (first SH coefficient = base color)
        # Shape is [N, 3, 25] -> take [:, :, 0] for DC component
        if harmonics.ndim == 3:
            colors = harmonics[:, :, 0]  # [N, 3]
        elif harmonics.ndim == 2:
            colors = harmonics[:, :3]  # [N, 3] already flattened
    
    # Fallback to features/colors
    if colors is None:
        for attr in ['features', 'colors', 'shs', 'sh']:
            if hasattr(gaussians, attr):
                colors = getattr(gaussians, attr).detach().cpu().numpy()
                if colors.ndim == 4 and colors.shape[0] == 1:
                    colors = colors[0]
                if colors.ndim == 3 and colors.shape[0] == 1:
                    colors = colors[0]
                if colors.shape[-1] >= 3:
                    colors = colors[:, :3]
                break
    
    # Default to white if no colors
    if colors is None or colors.size == 0:
        colors = np.ones((len(positions), 3), dtype=np.float32)
    else:
        colors = colors.astype(np.float32)
    
    # Ensure colors are [N, 3]
    if colors.ndim == 1:
        colors = np.tile(colors[:, None], (1, 3))
    elif colors.shape[1] > 3:
        colors = colors[:, :3]
    
    # Normalize colors to [0, 1]
    colors = np.clip(colors, 0, 1)
    
    # Add alpha channel
    colors = np.concatenate([colors, np.ones((len(colors), 1), dtype=np.float32)], axis=1)
    
    print(f"Converting {len(positions)} Gaussians to point cloud...")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Colors shape: {colors.shape}")
    
    # Create glTF
    gltf = GLTF2()
    
    # Create buffer with positions and colors
    positions_bytes = positions.tobytes()
    colors_bytes = colors.tobytes()
    buffer_data = positions_bytes + colors_bytes
    
    buffer = Buffer(byteLength=len(buffer_data))
    gltf.buffers.append(buffer)
    
    # Buffer views
    positions_view = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(positions_bytes),
        target=34962  # ARRAY_BUFFER
    )
    gltf.bufferViews.append(positions_view)
    
    colors_view = BufferView(
        buffer=0,
        byteOffset=len(positions_bytes),
        byteLength=len(colors_bytes),
        target=34962
    )
    gltf.bufferViews.append(colors_view)
    
    # Accessors
    positions_accessor = Accessor(
        bufferView=0,
        componentType=5126,  # FLOAT
        count=len(positions),
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist()
    )
    gltf.accessors.append(positions_accessor)
    
    colors_accessor = Accessor(
        bufferView=1,
        componentType=5126,
        count=len(colors),
        type="VEC4"
    )
    gltf.accessors.append(colors_accessor)
    
    # Material
    material = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
    )
    gltf.materials.append(material)
    
    # Mesh primitive (points)
    primitive = Primitive(
        attributes={"POSITION": 0, "COLOR_0": 1},
        mode=0,  # POINTS
        material=0
    )
    
    mesh = Mesh(primitives=[primitive])
    gltf.meshes.append(mesh)
    
    # Node and scene
    node = Node(mesh=0)
    gltf.nodes.append(node)
    
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0
    
    # Save as GLB or glTF
    if use_glb or output_path.endswith('.glb'):
        # Save as GLB (single binary file)
        gltf.set_binary_blob(buffer_data)
        output_path = output_path if output_path.endswith('.glb') else output_path.replace('.gltf', '.glb')
        gltf.save(output_path)
        file_size = os.path.getsize(output_path)
        print(f"\n✓ Saved to {output_path}")
        print(f"  Points: {len(positions):,}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    else:
        # Save as glTF + bin
        bin_path = output_path.replace('.gltf', '.bin')
        with open(bin_path, 'wb') as f:
            f.write(buffer_data)
        
        # Set the buffer URI to point to the bin file
        gltf.buffers[0].uri = os.path.basename(bin_path)
        
        # Save the glTF JSON
        gltf.save(output_path)
        
        gltf_size = os.path.getsize(output_path)
        bin_size = os.path.getsize(bin_path)
        print(f"\n✓ Saved to {output_path}")
        print(f"  Binary data: {bin_path}")
        print(f"  Points: {len(positions):,}")
        print(f"  Total size: {(gltf_size + bin_size) / 1024 / 1024:.2f} MB")
    
    print(f"\nTo view: Upload to https://gltf-viewer.donmccurdy.com/")


# Load the model from Hugging Face
video_urls = ["rtsp://192.168.1.190:554/stream/main",
              "rtsp://192.168.1.192:554/stream/main"]
model = AnySplat.from_pretrained("lhjiang/anysplat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Load and preprocess example images (replace with your own image paths)
# image_names = []
# for i in range(50):
#     image_names.append(f"/home/lauretta/quang/AnySplat/test_images_1/{i*2}.jpg")
#caps = []
#for video_string in video_urls:
#    caps.append(cv2.VideoCapture(video_string))

image_paths = ["/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0001.jpg", 
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0010.jpg",
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0019.jpg", 
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0028.jpg",
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0037.jpg", 
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0046.jpg",
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0055.jpg", 
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0064.jpg",
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0073.jpg", 
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0082.jpg",
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0091.jpg", 
               "/home/lauretta/quang/impact/AnySplat_impact/examples/vrnerf/riverview/21_DSC0100.jpg"
               ]
# frames = [cap.read()[1] for cap in caps]
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

# Inspect the Gaussians structure
inspect_gaussians(gaussians)

# Save in multiple formats
print("\n" + "="*70)
print("Saving outputs in different formats...")
print("="*70)

# 1. Save as PLY (full 3DGS format - best quality)
print("\n[1/3] Saving PLY format (full Gaussian Splatting data)...")
save_gaussians_to_ply(gaussians, "output.ply", binary=True)

# 2. Save as GLB (point cloud - for standard 3D viewers)
print("\n[2/3] Saving GLB format (point cloud representation)...")
save_gaussians_to_gltf(gaussians, "output.glb", use_glb=True)

# 3. Optional: Save as glTF (JSON + bin)
print("\n[3/3] Saving glTF format (point cloud representation)...")
begin = time()
save_gaussians_to_gltf(gaussians, "output.gltf", use_glb=False)
print(f"glTF saving time: {time() - begin:.2f} seconds")

print("\n" + "="*70)
print("Summary of outputs:")
print("="*70)
print("✓ output.ply  - Full 3D Gaussian Splatting (use with SuperSplat)")
print("✓ output.glb  - Point cloud in GLB format (single file)")
print("✓ output.gltf + output.bin - Point cloud in glTF format")
print("\nRecommended viewers:")
print("  PLY:  https://playcanvas.com/supersplat/editor")
print("  GLB:  https://gltf-viewer.donmccurdy.com/")
print("="*70)