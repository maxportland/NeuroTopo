"""
Mesh I/O utilities.

Supports loading and saving meshes in various formats.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from neurotopo.core.mesh import Mesh


def load_mesh(filepath: str) -> Mesh:
    """
    Load a mesh from file.
    
    Supports: OBJ, PLY, STL, OFF, GLB, GLTF
    FBX support requires external conversion.
    
    Args:
        filepath: Path to mesh file
        
    Returns:
        Loaded Mesh object
    """
    import trimesh
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    # Handle FBX specially
    if suffix == '.fbx':
        return _load_fbx(filepath)
    
    # Use trimesh for other formats
    tm = trimesh.load(str(filepath), process=False)
    
    # Handle scene vs single mesh
    if isinstance(tm, trimesh.Scene):
        # Merge all geometries
        meshes = list(tm.geometry.values())
        if not meshes:
            raise ValueError(f"No meshes found in {filepath}")
        
        if len(meshes) == 1:
            tm = meshes[0]
        else:
            # Concatenate all meshes
            tm = trimesh.util.concatenate(meshes)
    
    return Mesh(
        vertices=np.array(tm.vertices),
        faces=tm.faces.tolist(),
        name=filepath.stem
    )


def _load_fbx(filepath: Path) -> Mesh:
    """Load FBX file using available methods."""
    
    # Try pyfbx-i42 if available
    try:
        import pyfbx
        # Implementation would go here
        raise ImportError("pyfbx not configured")
    except ImportError:
        pass
    
    # Try blender python if available
    try:
        return _load_fbx_via_blender(filepath)
    except Exception:
        pass
    
    # Try Assimp library directly via ctypes
    try:
        return _load_fbx_via_assimp_ctypes(filepath)
    except Exception as e:
        pass
    
    # Fallback: look for converted OBJ
    obj_path = filepath.with_suffix('.obj')
    if obj_path.exists():
        print(f"FBX not directly supported, loading converted {obj_path}")
        return load_mesh(str(obj_path))
    
    raise NotImplementedError(
        f"FBX loading requires conversion. Please convert {filepath} to OBJ format.\n"
        f"You can use Blender: blender --background --python-expr "
        f"\"import bpy; bpy.ops.import_scene.fbx(filepath='{filepath}'); "
        f"bpy.ops.export_scene.obj(filepath='{obj_path}')\""
    )


def _load_fbx_via_assimp_ctypes(filepath: Path) -> Mesh:
    """Load FBX using assimp library via ctypes."""
    import ctypes
    import ctypes.util
    
    # Try to find assimp library
    lib_name = ctypes.util.find_library('assimp')
    if not lib_name:
        # Try common locations on macOS
        for path in ['/usr/local/lib/libassimp.dylib', '/opt/homebrew/lib/libassimp.dylib']:
            if Path(path).exists():
                lib_name = path
                break
    
    if not lib_name:
        raise ImportError("Assimp library not found")
    
    assimp = ctypes.CDLL(lib_name)
    
    # Define structures (simplified)
    class aiVector3D(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]
    
    class aiFace(ctypes.Structure):
        _fields_ = [
            ("mNumIndices", ctypes.c_uint),
            ("mIndices", ctypes.POINTER(ctypes.c_uint))
        ]
    
    class aiMesh(ctypes.Structure):
        _fields_ = [
            ("mPrimitiveTypes", ctypes.c_uint),
            ("mNumVertices", ctypes.c_uint),
            ("mNumFaces", ctypes.c_uint),
            ("mVertices", ctypes.POINTER(aiVector3D)),
            ("mNormals", ctypes.POINTER(aiVector3D)),
            # ... more fields
            ("mFaces", ctypes.POINTER(aiFace)),
        ]
    
    class aiScene(ctypes.Structure):
        _fields_ = [
            ("mFlags", ctypes.c_uint),
            ("mRootNode", ctypes.c_void_p),
            ("mNumMeshes", ctypes.c_uint),
            ("mMeshes", ctypes.POINTER(ctypes.POINTER(aiMesh))),
            # ... more fields
        ]
    
    # Import function
    aiImportFile = assimp.aiImportFile
    aiImportFile.restype = ctypes.POINTER(aiScene)
    aiImportFile.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    
    aiReleaseImport = assimp.aiReleaseImport
    aiReleaseImport.argtypes = [ctypes.POINTER(aiScene)]
    
    # Load file
    flags = 0x1 | 0x8 | 0x80  # Triangulate | JoinIdenticalVertices | GenNormals
    scene = aiImportFile(str(filepath).encode(), flags)
    
    if not scene:
        raise RuntimeError(f"Failed to load {filepath}")
    
    try:
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for i in range(scene.contents.mNumMeshes):
            mesh = scene.contents.mMeshes[i].contents
            
            # Get vertices
            for j in range(mesh.mNumVertices):
                v = mesh.mVertices[j]
                all_vertices.append([v.x, v.y, v.z])
            
            # Get faces
            for j in range(mesh.mNumFaces):
                face = mesh.mFaces[j]
                indices = [face.mIndices[k] + vertex_offset for k in range(face.mNumIndices)]
                all_faces.append(indices)
            
            vertex_offset += mesh.mNumVertices
        
        return Mesh(
            vertices=np.array(all_vertices),
            faces=all_faces,
            name=filepath.stem
        )
    finally:
        aiReleaseImport(scene)


def _load_fbx_via_blender(filepath: Path) -> Mesh:
    """Load FBX by calling Blender in background."""
    import subprocess
    import tempfile
    import json
    
    # Create temp output file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    
    # Blender script
    script = f'''
import bpy
import json

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import FBX
bpy.ops.import_scene.fbx(filepath="{filepath}")

# Get mesh data
result = {{"vertices": [], "faces": []}}
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mesh = obj.data
        offset = len(result["vertices"])
        
        for v in mesh.vertices:
            co = obj.matrix_world @ v.co
            result["vertices"].append([co.x, co.y, co.z])
        
        for p in mesh.polygons:
            result["faces"].append([i + offset for i in p.vertices])

# Save
with open("{output_path}", "w") as f:
    json.dump(result, f)
'''
    
    # Run Blender
    result = subprocess.run(
        ['blender', '--background', '--python-expr', script],
        capture_output=True,
        timeout=60
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Blender failed: {result.stderr.decode()}")
    
    # Load result
    with open(output_path) as f:
        data = json.load(f)
    
    Path(output_path).unlink()
    
    return Mesh(
        vertices=np.array(data['vertices']),
        faces=data['faces'],
        name=filepath.stem
    )


def save_mesh(mesh: Mesh, filepath: str) -> None:
    """
    Save mesh to file.
    
    Args:
        mesh: Mesh to save
        filepath: Output path (format determined by extension)
    """
    import trimesh
    
    filepath = Path(filepath)
    
    # Convert to trimesh
    tm = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=np.array([f[:3] if len(f) > 3 else f for f in mesh.faces]),
        process=False
    )
    
    # Export
    tm.export(str(filepath))


def convert_fbx_to_obj(fbx_path: str, obj_path: str = None) -> str:
    """
    Convert FBX to OBJ format.
    
    Args:
        fbx_path: Input FBX file
        obj_path: Output OBJ file (default: same name with .obj extension)
        
    Returns:
        Path to output OBJ file
    """
    fbx_path = Path(fbx_path)
    if obj_path is None:
        obj_path = fbx_path.with_suffix('.obj')
    else:
        obj_path = Path(obj_path)
    
    mesh = load_mesh(str(fbx_path))
    save_mesh(mesh, str(obj_path))
    
    return str(obj_path)
