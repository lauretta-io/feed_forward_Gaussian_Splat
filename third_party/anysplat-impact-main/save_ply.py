import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
import io

def inspect_gaussians(gaussians):
    """
    Print information about the Gaussians object to understand its structure.
    """
    print("\n=== Gaussians Object Structure ===")
    print(f"Type: {type(gaussians)}")
    print(f"Dir: {[attr for attr in dir(gaussians) if not attr.startswith('_')]}")
    
    # Try to access common attributes
    for attr in ['means', 'scales', 'rotations', 'opacities', 'harmonics', 'features', 'colors', 'shs', 
                 'xyz', 'scale', 'rotation', 'opacity', 'quats', 'sh', 'covariances']:
        if hasattr(gaussians, attr):
            val = getattr(gaussians, attr)
            if torch.is_tensor(val):
                print(f"  {attr}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
            else:
                print(f"  {attr}: type={type(val)}")
    print("=" * 50)


def save_gaussians_to_ply(gaussians, output_path, binary=True):
    """
    Save 3D Gaussians to PLY file in standard 3DGS format.
    Handles Gaussians objects with attributes.
    
    Args:
        gaussians: Gaussians object with attributes
        output_path: Path to save the PLY file
        binary: If True (default), save as binary PLY (smaller, faster)
                If False, save as ASCII PLY (human-readable)
    """
    
    print("Extracting Gaussian parameters...")
    
    # Extract parameters - try different attribute names
    # Means/positions
    if hasattr(gaussians, 'means'):
        means = gaussians.means
    elif hasattr(gaussians, 'xyz'):
        means = gaussians.xyz
    else:
        raise AttributeError("Cannot find position attribute (tried 'means', 'xyz')")
    
    # Scales
    if hasattr(gaussians, 'scales'):
        scales = gaussians.scales
    elif hasattr(gaussians, 'scale'):
        scales = gaussians.scale
    else:
        raise AttributeError("Cannot find scale attribute (tried 'scales', 'scale')")
    
    # Rotations
    if hasattr(gaussians, 'rotations'):
        rotations = gaussians.rotations
    elif hasattr(gaussians, 'quats'):
        rotations = gaussians.quats
    elif hasattr(gaussians, 'rotation'):
        rotations = gaussians.rotation
    else:
        raise AttributeError("Cannot find rotation attribute (tried 'rotations', 'quats', 'rotation')")
    
    # Opacities
    if hasattr(gaussians, 'opacities'):
        opacities = gaussians.opacities
    elif hasattr(gaussians, 'opacity'):
        opacities = gaussians.opacity
    else:
        raise AttributeError("Cannot find opacity attribute (tried 'opacities', 'opacity')")
    
    # Features/colors
    features = None
    if hasattr(gaussians, 'harmonics'):
        features = gaussians.harmonics
    elif hasattr(gaussians, 'features'):
        features = gaussians.features
    elif hasattr(gaussians, 'shs'):
        features = gaussians.shs
    elif hasattr(gaussians, 'sh'):
        features = gaussians.sh
    elif hasattr(gaussians, 'colors'):
        features = gaussians.colors
    
    # Convert to numpy and remove batch dimension
    means = means.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()
    opacities = opacities.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if means.ndim == 3 and means.shape[0] == 1:
        means = means[0]
    if scales.ndim == 3 and scales.shape[0] == 1:
        scales = scales[0]
    if rotations.ndim == 3 and rotations.shape[0] == 1:
        rotations = rotations[0]
    if opacities.ndim == 2 and opacities.shape[0] == 1:
        opacities = opacities[0]
    
    if features is not None:
        features = features.detach().cpu().numpy()
        # Handle different feature/harmonics shapes
        if features.ndim == 4:
            # Shape: [batch, N, 3, SH_coeffs] -> need to flatten to [N, 3*SH_coeffs]
            if features.shape[0] == 1:
                features = features[0]  # Remove batch: [N, 3, SH_coeffs]
            # Reshape to [N, 3*SH_coeffs]
            num_points = features.shape[0]
            num_channels = features.shape[1]  # 3
            num_sh = features.shape[2]  # 25
            features = features.reshape(num_points, num_channels * num_sh)
        elif features.ndim == 3:
            if features.shape[0] == 1:
                features = features[0]  # Remove batch dimension
    else:
        # Default to gray
        features = np.ones((means.shape[0], 3)) * 0.5
    
    print(f"\nShapes after extraction:")
    print(f"  means: {means.shape}")
    print(f"  scales: {scales.shape}")
    print(f"  rotations: {rotations.shape}")
    print(f"  opacities: {opacities.shape}")
    print(f"  features: {features.shape}")
    
    # Handle opacities shape
    if opacities.ndim == 2:
        if opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)
        else:
            # If it has multiple columns, take the first one or mean
            print(f"  Warning: opacities has shape {opacities.shape}, taking first column")
            opacities = opacities[:, 0]
    elif opacities.ndim > 2:
        print(f"  Warning: opacities has {opacities.ndim} dimensions, flattening")
        opacities = opacities.reshape(-1)
    
    # Ensure opacities length matches
    if len(opacities) != len(means):
        print(f"  Warning: opacities length {len(opacities)} != means length {len(means)}")
        if len(opacities) > len(means):
            opacities = opacities[:len(means)]
        else:
            opacities = np.pad(opacities, (0, len(means) - len(opacities)), constant_values=1.0)
    
    # Handle scales shape
    if scales.ndim == 1:
        scales = np.tile(scales[:, None], (1, 3))
    elif scales.shape[1] != 3:
        if scales.shape[1] == 1:
            scales = np.tile(scales, (1, 3))
        else:
            print(f"  Warning: unexpected scales shape {scales.shape}")
            scales = scales[:, :3]
    
    # Handle rotations shape
    if rotations.shape[1] != 4:
        raise ValueError(f"Rotations should be [N, 4] quaternions, got {rotations.shape}")
    
    # Handle features shape
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    if features.shape[0] != means.shape[0]:
        print(f"  Warning: features shape {features.shape} doesn't match means")
        if features.shape[0] > means.shape[0]:
            features = features[:means.shape[0]]
        else:
            # Pad with proper dimensions
            pad_width = [(0, means.shape[0] - features.shape[0])] + [(0, 0)] * (features.ndim - 1)
            features = np.pad(features, pad_width, constant_values=0.5)
    
    # Extract RGB from features
    # Features should now be [N, 3*SH_coeffs] after reshaping, where first 3 values are RGB
    if features.shape[-1] >= 3:
        # For harmonics: [N, 75] where 75 = 3 channels * 25 SH coefficients
        # First 3 values are the DC components (RGB)
        rgb = features[:, :3]
    elif features.shape[-1] == 1:
        rgb = np.tile(features, (1, 3))
    else:
        rgb = np.pad(features, ((0, 0), (0, 3 - features.shape[-1])), constant_values=0.5)
    
    print(f"\nFinal shapes:")
    print(f"  means: {means.shape}")
    print(f"  scales: {scales.shape}")
    print(f"  rotations: {rotations.shape}")
    print(f"  opacities: {opacities.shape}")
    print(f"  rgb: {rgb.shape}")
    
    # Normalize quaternions
    rotations = rotations / (np.linalg.norm(rotations, axis=-1, keepdims=True) + 1e-8)
    
    # Apply inverse sigmoid to opacity (logit transform)
    opacities = np.clip(opacities, 1e-6, 1 - 1e-6)
    opacities_logit = np.log(opacities / (1 - opacities))
    
    # Prepare data for PLY
    num_points = means.shape[0]
    print(f"\nPreparing PLY data for {num_points} points...")
    
    # Standard 3DGS PLY format
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    
    # Add higher order SH coefficients if available
    if features.shape[-1] > 3:
        num_extra = features.shape[-1] - 3
        for i in range(num_extra):
            dtype_full.append((f'f_rest_{i}', 'f4'))
    
    # Create structured array
    elements = np.empty(num_points, dtype=dtype_full)
    
    # Fill positions
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    
    # Normals (set to 0)
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    
    # RGB (DC component)
    elements['f_dc_0'] = rgb[:, 0]
    elements['f_dc_1'] = rgb[:, 1]
    elements['f_dc_2'] = rgb[:, 2]
    
    # Opacity
    elements['opacity'] = opacities_logit
    
    # Scales (log transform)
    elements['scale_0'] = np.log(np.clip(scales[:, 0], 1e-8, None))
    elements['scale_1'] = np.log(np.clip(scales[:, 1], 1e-8, None))
    elements['scale_2'] = np.log(np.clip(scales[:, 2], 1e-8, None))
    
    # Rotations (quaternion)
    elements['rot_0'] = rotations[:, 0]
    elements['rot_1'] = rotations[:, 1]
    elements['rot_2'] = rotations[:, 2]
    elements['rot_3'] = rotations[:, 3]
    
    # Higher order SH
    if features.shape[-1] > 3:
        for i in range(3, features.shape[-1]):
            elements[f'f_rest_{i-3}'] = features[:, i]
    
    # Create and write PLY (binary or ASCII)
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element], text=(not binary))
    ply_data.write(output_path)
    
    # Get file size
    file_size = os.path.getsize(output_path)
    
    print(f"\n✓ Successfully saved {num_points:,} Gaussians to {output_path}")
    print(f"  Format: {'Binary' if binary else 'ASCII'} PLY")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    
    if binary:
        # Estimate ASCII size for comparison
        ascii_size_estimate = file_size * 2.5  # Binary is typically 2-3x smaller
        savings = (1 - file_size / ascii_size_estimate) * 100
        print(f"  Estimated savings vs ASCII: ~{savings:.0f}%")


def save_gaussians_simple(gaussians, output_path, use_glb=True):
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
        buffer = io.BytesIO()
        gltf.save(buffer)
        return buffer.getvalue()
        

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
    positions_viewgltf = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(positions_bytes),
        target=34962  # ARRAY_BUFFER
    )
    gltf.bufferViews.append(positions_view)
    
    colors_view = BufferView(
        buffer=0,
        byteOffsetgltf=len(positions_bytes),
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
    print(f"\nTo view: Upload to https://gltf-viewer.donmccurdy.com/")
    colors = None
    for attr in ['harmonics', 'features', 'colors', 'shs', 'sh']:
        if hasattr(gaussians, attr):
            colors = getattr(gaussians, attr).detach().cpu().numpy()
            # Remove batch dimension
            if colors.ndim == 3 and colors.shape[0] == 1:
                colors = colors[0]
            break
    
    num_points = means.shape[0]
    
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]
    
    elements = np.empty(num_points, dtype=dtype)
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    
    # Set colors
    if colors is not None and colors.size > 0:
        if colors.ndim == 1:
            colors = colors.reshape(-1, 1)
        
        if colors.shape[1] >= 3:
            rgb = colors[:, :3]
        else:
            rgb = np.tile(colors[:, 0:1], (1, 3))
        
        # Normalize to 0-255
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        elements['red'] = rgb[:, 0]
        elements['green'] = rgb[:, 1]
        elements['blue'] = rgb[:, 2]
    else:
        elements['red'] = 255
        elements['green'] = 255
        elements['blue'] = 255
    
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element], text=(not binary))
    ply_data.write(output_path)
    
    file_size = os.path.getsize(output_path)
    print(f"✓ Saved {num_points:,} points to {output_path}")
    print(f"  Format: {'Binary' if binary else 'ASCII'} PLY")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")