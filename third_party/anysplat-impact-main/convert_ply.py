"""
Convert 3D Gaussian Splatting PLY to glTF and other compressed formats

IMPORTANT NOTES:
1. Standard glTF doesn't natively support Gaussian Splatting parameters
2. glTF conversion will lose the Gaussian properties (scales, rotations, SH coefficients)
3. For 3DGS, better formats are: .splat, .ksplat, .spz, or compressed PLY
4. If you need glTF for a specific viewer/pipeline, the Gaussians will be converted to a point cloud or mesh
"""

import numpy as np
import torch
from plyfile import PlyData
import struct
import gzip
import json
import os


# ============================================================================
# METHOD 1: Convert PLY to .SPLAT format (Best compression for 3DGS)
# ============================================================================

def ply_to_splat(ply_path, splat_path):
    """
    Convert PLY to .splat format (optimized for 3DGS).
    This is the most efficient format for Gaussian Splatting.
    
    .splat format stores each Gaussian in a compact binary format.
    Much smaller than PLY and designed specifically for real-time rendering.
    """
    print(f"Converting {ply_path} to .splat format...")
    
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    num_points = len(vertex)
    
    print(f"Processing {num_points} Gaussians...")
    
    # Extract data
    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    
    # Extract scales (stored as log in PLY)
    scales = np.stack([
        np.exp(vertex['scale_0']),
        np.exp(vertex['scale_1']),
        np.exp(vertex['scale_2'])
    ], axis=1)
    
    # Extract colors (SH DC component)
    colors = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=1)
    
    # Convert to 0-255 range
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    
    # Extract opacity (stored as logit in PLY)
    opacity_logit = vertex['opacity']
    opacity = 1 / (1 + np.exp(-opacity_logit))  # sigmoid
    opacity = np.clip(opacity * 255, 0, 255).astype(np.uint8)
    
    # Extract rotations (quaternions)
    rotations = np.stack([
        vertex['rot_0'],
        vertex['rot_1'],
        vertex['rot_2'],
        vertex['rot_3']
    ], axis=1)
    
    # Normalize quaternions
    rotations = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)
    
    # Convert rotations to uint8 (compress)
    # Store as normalized int8 for the first 3 components
    # The 4th component can be reconstructed
    rot_compressed = np.clip(rotations[:, :3] * 127, -128, 127).astype(np.int8)
    
    # Write .splat file
    with open(splat_path, 'wb') as f:
        for i in range(num_points):
            # Position (3x float32)
            f.write(struct.pack('fff', *positions[i]))
            
            # Scale (3x float32)
            f.write(struct.pack('fff', *scales[i]))
            
            # Color (4x uint8: RGB + opacity)
            f.write(struct.pack('BBBB', colors[i, 0], colors[i, 1], colors[i, 2], opacity[i]))
            
            # Rotation (4x int8, but we only store 3)
            f.write(struct.pack('bbb', *rot_compressed[i]))
            f.write(struct.pack('b', 0))  # Padding
    
    original_size = os.path.getsize(ply_path)
    compressed_size = os.path.getsize(splat_path)
    compression_ratio = original_size / compressed_size
    
    print(f"✓ Saved to {splat_path}")
    print(f"  Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"  Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")


# ============================================================================
# METHOD 2: Convert to glTF (loses Gaussian properties)
# ============================================================================

def ply_to_gltf_pointcloud(ply_path, gltf_path, max_points=None, use_glb=True):
    """
    Convert PLY to glTF as a point cloud.
    
    WARNING: This loses all Gaussian Splatting properties!
    Only positions and colors are preserved.
    
    Args:
        ply_path: Input PLY file
        gltf_path: Output glTF/GLB file path
        max_points: Maximum number of points (downsample if exceeded)
        use_glb: If True, save as GLB (single binary file), otherwise glTF + bin
    
    Requires: pip install pygltflib numpy
    """
    try:
        from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene, Material, PbrMetallicRoughness
        from pygltflib.utils import ImageFormat
    except ImportError:
        print("Error: pygltflib not installed")
        print("Install with: pip install pygltflib")
        return
    
    print(f"Converting {ply_path} to {'GLB' if use_glb else 'glTF'} point cloud...")
    print("WARNING: Gaussian properties will be lost!")
    
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Extract positions
    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
    
    # Extract colors
    if 'f_dc_0' in vertex.data.dtype.names:
        colors = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
        colors = np.clip(colors, 0, 1).astype(np.float32)
    elif 'red' in vertex.data.dtype.names:
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1) / 255.0
        colors = colors.astype(np.float32)
    else:
        colors = np.ones((len(positions), 3), dtype=np.float32)
    
    # Add alpha channel
    colors = np.concatenate([colors, np.ones((len(colors), 1), dtype=np.float32)], axis=1)
    
    # Downsample if needed
    if max_points and len(positions) > max_points:
        print(f"Downsampling from {len(positions)} to {max_points} points...")
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]
    
    print(f"Converting {len(positions)} points...")
    
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
    if use_glb or gltf_path.endswith('.glb'):
        # Save as GLB (single binary file)
        gltf.set_binary_blob(buffer_data)
        output_path = gltf_path if gltf_path.endswith('.glb') else gltf_path.replace('.gltf', '.glb')
        gltf.save(output_path)
        print(f"✓ Saved to {output_path}")
        print(f"  Points: {len(positions)}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    else:
        # Save as glTF + bin
        bin_path = gltf_path.replace('.gltf', '.bin')
        with open(bin_path, 'wb') as f:
            f.write(buffer_data)
        
        # Set the buffer URI to point to the bin file
        gltf.buffers[0].uri = os.path.basename(bin_path)
        
        # Save the glTF JSON
        gltf.save(gltf_path)
        
        print(f"✓ Saved to {gltf_path}")
        print(f"  Binary data: {bin_path}")
        print(f"  Points: {len(positions)}")
        print(f"  Total size: {(os.path.getsize(gltf_path) + os.path.getsize(bin_path)) / 1024 / 1024:.2f} MB")


# ============================================================================
# METHOD 3: Compress PLY with gzip
# ============================================================================

def compress_ply_gzip(ply_path, output_path=None):
    """
    Compress PLY file with gzip.
    Simple compression that maintains all data.
    """
    if output_path is None:
        output_path = ply_path + '.gz'
    
    print(f"Compressing {ply_path} with gzip...")
    
    with open(ply_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            f_out.writelines(f_in)
    
    original_size = os.path.getsize(ply_path)
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    
    print(f"✓ Compressed to {output_path}")
    print(f"  Original: {original_size / 1024 / 1024:.2f} MB")
    print(f"  Compressed: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"  Ratio: {compression_ratio:.2f}x")


# ============================================================================
# METHOD 4: Convert to .SPZ format (Niantic's compressed format)
# ============================================================================

def ply_to_spz(ply_path, spz_path):
    """
    Convert to SPZ format (90% smaller than PLY).
    This maintains all Gaussian properties.
    
    Requires: pip install spz-converter (if available)
    Or use: https://github.com/niantic-oss/spz
    """
    print("SPZ conversion requires the spz library or converter tool")
    print("See: https://scaniverse.com/news/spz-gaussian-splat-open-source-file-format")
    print("Or use: https://github.com/francescofugazzi/3dgsconverter")
    

# ============================================================================
# METHOD 5: Use 3dgsconverter tool (RECOMMENDED)
# ============================================================================

def convert_with_3dgsconverter(ply_path, output_format='splat'):
    """
    Use the 3dgsconverter tool for professional conversions.
    
    Install: pip install git+https://github.com/francescofugazzi/3dgsconverter
    
    Supported formats:
    - splat: Antimatter15's format
    - ksplat: Compressed splat with K-means clustering
    - sog: PlayCanvas SuperSplat format
    - spz: Niantic's compressed format
    """
    import subprocess
    
    base_name = os.path.splitext(ply_path)[0]
    output_path = f"{base_name}.{output_format}"
    
    cmd = [
        "3dgsconverter",
        "-i", ply_path,
        "-o", output_path,
        "-f", output_format
    ]
    
    if output_format in ['sog', 'ksplat']:
        # Add compression level
        cmd.extend(["--compression_level", "5"])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"✓ Converted to {output_path}")


# ============================================================================
# Comparison of formats
# ============================================================================

def print_format_comparison():
    """
    Print a comparison of different 3DGS formats.
    """
    comparison = """
    3D Gaussian Splatting Format Comparison:
    
    Format  | Size     | Quality | Compatibility | Use Case
    --------|----------|---------|---------------|---------------------------
    .ply    | 100%     | Perfect | Universal     | Original format, editing
    .splat  | ~30%     | Perfect | Web viewers   | Web deployment, fast loading
    .ksplat | ~15%     | High    | KSplat viewer | Maximum compression
    .spz    | ~10%     | High    | Scaniverse    | Mobile apps, storage
    .sog    | ~20%     | High    | SuperSplat    | PlayCanvas ecosystem
    .gltf   | Varies   | Loss    | Universal     | Standard 3D pipelines*
    .ply.gz | ~40%     | Perfect | Needs unzip   | Simple compression
    
    *glTF loses Gaussian properties - only for point cloud representation
    
    RECOMMENDATION:
    - For web viewing: .splat or .sog
    - For maximum compression: .spz or .ksplat
    - For editing/processing: .ply
    - For standard 3D pipelines: .gltf (with data loss warning)
    """
    print(comparison)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    
    print("\n" + "="*70)
    print("3D Gaussian Splatting PLY Converter")
    print("="*70 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage: python convert_ply.py <input.ply> [format]")
        print("\nSupported formats:")
        print("  splat   - Antimatter15 format (30% of original)")
        print("  gltf    - glTF point cloud (loses Gaussian properties)")
        print("  glb     - GLB point cloud (single file, loses Gaussian properties)")
        print("  gz      - Gzipped PLY (40% of original)")
        print("  compare - Show format comparison\n")
        
        print("Examples:")
        print("  python convert_ply.py output.ply splat")
        print("  python convert_ply.py output.ply glb")
        print("  python convert_ply.py output.ply gltf")
        print("  python convert_ply.py output.ply gz")
        print("  python convert_ply.py compare\n")
        
        print("For best compression, install 3dgsconverter:")
        print("  pip install git+https://github.com/francescofugazzi/3dgsconverter")
        print("  3dgsconverter -i input.ply -o output.spz -f spz\n")
        sys.exit(1)
    
    if sys.argv[1] == 'compare':
        print_format_comparison()
        sys.exit(0)
    
    ply_path = sys.argv[1]
    format_type = sys.argv[2] if len(sys.argv) > 2 else 'splat'
    
    if not os.path.exists(ply_path):
        print(f"Error: File not found: {ply_path}")
        sys.exit(1)
    
    base_name = os.path.splitext(ply_path)[0]
    
    if format_type == 'splat':
        ply_to_splat(ply_path, f"{base_name}.splat")
    elif format_type == 'gltf':
        ply_to_gltf_pointcloud(ply_path, f"{base_name}.gltf", use_glb=False)
    elif format_type == 'glb':
        ply_to_gltf_pointcloud(ply_path, f"{base_name}.glb", use_glb=True)
    elif format_type == 'gz':
        compress_ply_gzip(ply_path)
    else:
        print(f"Unknown format: {format_type}")
        print("Supported: splat, gltf, glb, gz")def save_gaussians_to_gltf(gaussians, output_path, use_glb=True):
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