import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
import io

import struct

def create_glb_binary(gltf, buffer_data):
    """
    Manually create GLB binary without saving to file.
    GLB format: [Header][JSON chunk][BIN chunk]
    """
    import json
    
    # Convert glTF to JSON
    gltf_dict = gltf.to_dict()
    json_data = json.dumps(gltf_dict, separators=(',', ':')).encode('utf-8')
    
    # Pad JSON to 4-byte alignment
    json_padding = (4 - len(json_data) % 4) % 4
    json_data += b' ' * json_padding
    
    # Pad binary data to 4-byte alignment
    bin_padding = (4 - len(buffer_data) % 4) % 4
    buffer_data += b'\x00' * bin_padding
    
    # GLB Header (12 bytes)
    # Magic: 0x46546C67 ("glTF")
    # Version: 2
    # Length: total file size
    total_length = 12 + 8 + len(json_data) + 8 + len(buffer_data)
    header = struct.pack('<III', 0x46546C67, 2, total_length)
    
    # JSON chunk header (8 bytes)
    # Chunk length, Chunk type: 0x4E4F534A ("JSON")
    json_chunk_header = struct.pack('<II', len(json_data), 0x4E4F534A)
    
    # BIN chunk header (8 bytes)
    # Chunk length, Chunk type: 0x004E4942 ("BIN\0")
    bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)
    
    # Combine all parts
    glb_binary = (
        header +
        json_chunk_header + json_data +
        bin_chunk_header + buffer_data
    )
    
    return glb_binary

def stream_glb(gaussians):
    """
    Simplified version - just saves positions as point cloud.
    
    Args:
        binary: If True (default), save as binary PLY
    """
    print("\nUsing simplified PLY format...")
    
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
    
    # Try to get colorsdef save_gaussians_to_gltf(gaussians, output_path, use_glb=True):
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

    # Save as GLB (single binary file)
    gltf.set_binary_blob(buffer_data)

    glb_binary = create_glb_binary(gltf, buffer_data)
    return glb_binary

    buffer = io.BytesIO()
    gltf.save(buffer)
    return buffer.getvalue()
