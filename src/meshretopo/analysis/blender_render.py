"""
Blender-based mesh rendering for semantic analysis.

Uses Blender's headless CLI mode to render high-quality images
without Python version conflicts. Blender runs as a subprocess
with a Python script passed to it.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("meshretopo.analysis.blender_render")

# Blender render script template
BLENDER_RENDER_SCRIPT = '''
import bpy
import math
import mathutils
import json
import sys
import os

def setup_scene():
    """Clear default scene and set up rendering with good surface detail visibility."""
    # Delete all default objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Set render engine - use EEVEE for faster rendering
    # Note: Blender 5.0+ uses 'BLENDER_EEVEE', earlier versions used 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    
    # Enable Fast GI (ambient occlusion) for better surface detail - Blender 5.0 API
    bpy.context.scene.eevee.use_fast_gi = True
    bpy.context.scene.eevee.fast_gi_method = 'GLOBAL_ILLUMINATION'
    bpy.context.scene.eevee.fast_gi_distance = 1.0
    
    # Enable shadows for depth perception
    bpy.context.scene.eevee.use_shadows = True
    
    # Set render samples for quality
    bpy.context.scene.eevee.taa_render_samples = 64
    
    # Set background to neutral gray (good for AI analysis)
    world = bpy.data.worlds.get('World')
    if world is None:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.25, 0.25, 0.25, 1.0)  # Slightly lighter gray

def import_mesh(filepath):
    """Import mesh file."""
    ext = filepath.lower().split('.')[-1]
    
    if ext == 'obj':
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == 'ply':
        bpy.ops.wm.ply_import(filepath=filepath)
    elif ext == 'stl':
        bpy.ops.wm.stl_import(filepath=filepath)
    elif ext == 'fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in ('gltf', 'glb'):
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    # Get imported object
    obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
    if obj is None:
        # Try to find any mesh object
        for o in bpy.data.objects:
            if o.type == 'MESH':
                obj = o
                break
    
    if obj is None:
        raise RuntimeError("No mesh object found after import")
    
    return obj

def setup_material(obj, wireframe=False):
    """Set up a clay-like material for good visibility of surface details.
    
    Args:
        obj: The Blender mesh object
        wireframe: If True, add wireframe overlay using geometry nodes
    """
    mat = bpy.data.materials.new(name="ClayMaterial")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes for a matcap-style shader that shows surface details
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    
    # Set clay-like properties - good for showing surface detail
    principled.inputs['Base Color'].default_value = (0.9, 0.85, 0.8, 1.0)  # Light clay color
    principled.inputs['Roughness'].default_value = 0.4  # Lower roughness for sharper highlights
    principled.inputs['Metallic'].default_value = 0.0
    
    # Add subsurface scattering for softer, more organic look
    if 'Subsurface Weight' in principled.inputs:
        principled.inputs['Subsurface Weight'].default_value = 0.1
    elif 'Subsurface' in principled.inputs:
        principled.inputs['Subsurface'].default_value = 0.1
    
    # Position nodes
    output.location = (300, 0)
    principled.location = (0, 0)
    
    # Link nodes
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign to object - clear existing materials first
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    
    if wireframe:
        setup_wireframe_overlay(obj)
    
    # Recalculate normals to ensure proper shading
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

def setup_wireframe_overlay(obj):
    """Add wireframe overlay using Blender's Wireframe modifier.
    
    Creates a visible edge network overlaid on the mesh surface.
    This helps AI analysis see the actual topology structure.
    """
    # Calculate wireframe thickness based on mesh size
    bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8
    radius = max((v - center).length for v in bbox)
    
    # Wireframe thickness: ~0.15% of mesh radius 
    wire_thickness = radius * 0.0015
    print(f"Wireframe thickness: {wire_thickness:.4f} (mesh radius: {radius:.2f})")
    
    # Duplicate the object for wireframe overlay
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    wire_obj = bpy.context.active_object
    wire_obj.name = obj.name + "_wireframe"
    
    # Add wireframe modifier to the duplicate
    wireframe_mod = wire_obj.modifiers.new(name="Wireframe", type='WIREFRAME')
    wireframe_mod.thickness = wire_thickness
    wireframe_mod.use_replace = True  # Replace mesh with wireframe only
    wireframe_mod.use_even_offset = True
    wireframe_mod.use_boundary = True
    
    # Create black material for wireframe
    wire_mat = bpy.data.materials.new(name="WireframeMaterial")
    wire_mat.use_nodes = True
    wire_nodes = wire_mat.node_tree.nodes
    wire_links = wire_mat.node_tree.links
    wire_nodes.clear()
    
    wire_output = wire_nodes.new('ShaderNodeOutputMaterial')
    wire_principled = wire_nodes.new('ShaderNodeBsdfPrincipled')
    wire_principled.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)  # Black
    wire_principled.inputs['Roughness'].default_value = 0.9
    wire_principled.inputs['Metallic'].default_value = 0.0
    wire_output.location = (300, 0)
    wire_principled.location = (0, 0)
    wire_links.new(wire_principled.outputs['BSDF'], wire_output.inputs['Surface'])
    
    # Assign material to wireframe object
    wire_obj.data.materials.clear()
    wire_obj.data.materials.append(wire_mat)
    
    # Slightly offset wireframe to prevent z-fighting (push outward)
    # Use a Displace modifier with a small offset
    displace_mod = wire_obj.modifiers.new(name="Offset", type='DISPLACE')
    displace_mod.direction = 'NORMAL'
    displace_mod.strength = wire_thickness * 0.5  # Small outward offset
    displace_mod.mid_level = 0.0
    
    print(f"Wireframe overlay added using Wireframe modifier")

def setup_lighting(obj):
    """Set up studio lighting relative to mesh bounds for good feature visibility."""
    # Get object bounds to scale lighting appropriately
    bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8
    radius = max((v - center).length for v in bbox)
    
    print(f"Mesh center: {center}, radius: {radius}")
    
    # Scale light distance and energy based on mesh size
    light_dist = radius * 2.5
    # Use constant energy that works well for clay rendering
    base_energy = 500.0
    
    # Key light - main light from front-right-top (classic portrait lighting)
    # Using SUN light for more even illumination
    bpy.ops.object.light_add(type='SUN', location=(
        center.x + light_dist,
        center.y - light_dist,
        center.z + light_dist
    ))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 3.0
    # Point at center
    direction = center - key.location
    key.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Fill light - from front-left, softer
    bpy.ops.object.light_add(type='SUN', location=(
        center.x - light_dist,
        center.y - light_dist,
        center.z + light_dist * 0.5
    ))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 1.5
    direction = center - fill.location
    fill.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Rim light from back
    bpy.ops.object.light_add(type='SUN', location=(
        center.x,
        center.y + light_dist,
        center.z + light_dist * 0.3
    ))
    rim = bpy.context.object
    rim.name = "RimLight"
    rim.data.energy = 1.0
    direction = center - rim.location
    rim.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Bottom fill to reduce harsh shadows under features
    bpy.ops.object.light_add(type='AREA', location=(
        center.x,
        center.y - light_dist * 0.5,
        center.z - light_dist * 0.5
    ))
    bottom = bpy.context.object
    bottom.name = "BottomFill"
    bottom.data.energy = base_energy * 0.3
    bottom.data.size = radius * 3
    direction = center - bottom.location
    bottom.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def setup_camera(obj, azimuth, elevation, distance_factor=2.5):
    """Position camera to look at object from given angles."""
    # Get object bounds
    bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8
    
    # Calculate bounding sphere radius
    radius = max((v - center).length for v in bbox)
    distance = radius * distance_factor
    
    # Convert angles to radians
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    
    # Camera position
    x = distance * math.cos(el_rad) * math.sin(az_rad)
    y = -distance * math.cos(el_rad) * math.cos(az_rad)  # Blender Y is forward
    z = distance * math.sin(el_rad)
    
    cam_pos = center + mathutils.Vector((x, y, z))
    
    # Create camera
    bpy.ops.object.camera_add(location=cam_pos)
    camera = bpy.context.object
    
    # Point at center using track-to constraint
    constraint = camera.constraints.new(type='TRACK_TO')
    
    # Create empty at center for tracking
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
    target = bpy.context.object
    target.name = "CameraTarget"
    
    # Select camera again
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera
    
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    return camera, target

def render_view(output_path, resolution):
    """Render current view to file."""
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    
    # Render
    bpy.ops.render.render(write_still=True)

def main():
    # Parse arguments (passed after --)
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    
    if len(argv) < 2:
        print("Usage: blender --background --python script.py -- input.obj output_dir [resolution] [wireframe]")
        sys.exit(1)
    
    input_file = argv[0]
    output_dir = argv[1]
    resolution = (1024, 1024)  # Default resolution
    wireframe = True  # Default to wireframe ON for topology analysis
    
    if len(argv) > 2:
        res = int(argv[2])
        resolution = (res, res)
    
    if len(argv) > 3:
        wireframe = argv[3].lower() in ('true', '1', 'yes', 'on')
    
    # Camera angles: (azimuth, elevation, name)
    views = [
        (0, 0, "front"),
        (180, 0, "back"),
        (90, 0, "right"),
        (-90, 0, "left"),
        (45, 30, "front_top"),
        (-45, 30, "back_top"),
    ]
    
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Resolution: {resolution}")
    print(f"Wireframe: {wireframe}")
    
    # Setup
    setup_scene()
    obj = import_mesh(input_file)
    print(f"Imported mesh: {obj.name} with {len(obj.data.vertices)} vertices")
    
    setup_material(obj, wireframe=wireframe)
    setup_lighting(obj)  # Pass obj for bounds-relative lighting
    
    # Render each view
    results = []
    for azimuth, elevation, name in views:
        print(f"Rendering view: {name} (az={azimuth}, el={elevation})")
        camera, target = setup_camera(obj, azimuth, elevation)
        output_path = os.path.join(output_dir, f"view_{name}.png")
        render_view(output_path, resolution)
        
        results.append({
            "name": name,
            "azimuth": azimuth,
            "elevation": elevation,
            "path": output_path,
        })
        
        # Remove camera and target for next view
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.objects.remove(target, do_unlink=True)
    
    # Write results metadata
    meta_path = os.path.join(output_dir, "render_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"SUCCESS: Rendered {len(results)} views to {output_dir}")

if __name__ == "__main__":
    main()
'''


def find_blender() -> Optional[str]:
    """Find Blender executable on the system."""
    # Common locations
    candidates = [
        # macOS
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/Applications/Blender.app/Contents/MacOS/blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        # Linux
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/snap/bin/blender",
        # Windows
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
        "C:/Program Files/Blender Foundation/Blender/blender.exe",
    ]
    
    # Check PATH first
    import shutil
    blender_path = shutil.which("blender")
    if blender_path:
        return blender_path
    
    # Check common locations
    for path in candidates:
        if os.path.isfile(path):
            return path
    
    return None


class BlenderRenderer:
    """
    Render meshes using Blender's headless mode.
    
    This avoids Python version conflicts by running Blender as a subprocess.
    """
    
    def __init__(
        self,
        blender_path: Optional[str] = None,
        resolution: tuple[int, int] = (1024, 1024),
        timeout: int = 120,
    ):
        """
        Initialize Blender renderer.
        
        Args:
            blender_path: Path to Blender executable (auto-detect if None)
            resolution: Output image resolution
            timeout: Maximum render time in seconds
        """
        self.blender_path = blender_path or find_blender()
        self.resolution = resolution
        self.timeout = timeout
        
        if self.blender_path is None:
            logger.warning("Blender not found on system")
    
    @property
    def available(self) -> bool:
        """Check if Blender is available."""
        return self.blender_path is not None
    
    def render_mesh(
        self,
        mesh_path: str | Path,
        output_dir: Optional[str | Path] = None,
        wireframe: bool = True,
    ) -> tuple[list[np.ndarray], list[dict]]:
        """
        Render mesh from multiple viewpoints.
        
        Args:
            mesh_path: Path to mesh file (OBJ, PLY, STL, FBX, GLTF)
            output_dir: Directory for output images (temp if None)
            wireframe: If True, overlay wireframe on mesh (default: True for topology analysis)
            
        Returns:
            Tuple of (list of render images, list of camera params)
        """
        if not self.available:
            raise RuntimeError("Blender not available")
        
        mesh_path = Path(mesh_path).resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        
        # Create temp directory if needed
        cleanup_dir = False
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="blender_render_")
            cleanup_dir = True
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write render script
            script_path = Path(output_dir) / "render_script.py"
            with open(script_path, 'w') as f:
                f.write(BLENDER_RENDER_SCRIPT)
            
            # Run Blender
            cmd = [
                self.blender_path,
                "--background",
                "--python", str(script_path),
                "--",
                str(mesh_path),
                str(output_dir),
                str(self.resolution[0]),
                str(wireframe).lower(),  # Pass wireframe parameter
            ]
            
            logger.info(f"Running Blender render: {' '.join(cmd[:4])} ...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            if result.returncode != 0:
                logger.error(f"Blender failed: {result.stderr}")
                raise RuntimeError(f"Blender render failed: {result.stderr}")
            
            # Load results
            meta_path = Path(output_dir) / "render_meta.json"
            if not meta_path.exists():
                raise RuntimeError("Render metadata not found")
            
            with open(meta_path) as f:
                metadata = json.load(f)
            
            # Load images
            from PIL import Image
            
            renders = []
            params = []
            
            for view in metadata:
                img_path = Path(view["path"])
                if img_path.exists():
                    img = Image.open(img_path)
                    renders.append(np.array(img))
                    params.append({
                        "name": view["name"],
                        "azimuth": view["azimuth"],
                        "elevation": view["elevation"],
                    })
            
            logger.info(f"Loaded {len(renders)} rendered views")
            return renders, params
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Blender render timed out after {self.timeout}s")
            
        finally:
            # Cleanup temp directory
            if cleanup_dir:
                import shutil
                try:
                    shutil.rmtree(output_dir)
                except Exception:
                    pass


def render_mesh_with_blender(
    mesh_path: str | Path,
    resolution: tuple[int, int] = (1024, 1024),
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Convenience function to render mesh with Blender.
    
    Args:
        mesh_path: Path to mesh file
        resolution: Output resolution
        
    Returns:
        Tuple of (render images, camera parameters)
    """
    renderer = BlenderRenderer(resolution=resolution)
    
    if not renderer.available:
        raise RuntimeError(
            "Blender not found. Install Blender and ensure it's in PATH, "
            "or set blender_path explicitly."
        )
    
    return renderer.render_mesh(mesh_path)
