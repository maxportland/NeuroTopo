"""Analyze topology quality of source vs retopo mesh."""
import trimesh
import numpy as np
from collections import Counter
import subprocess
import tempfile
from pathlib import Path

def analyze_mesh_topology(mesh, name):
    """Analyze topology quality metrics for a mesh."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    
    # Vertex valence analysis (in triangulated form for comparison)
    edges = mesh.edges_unique
    valence = Counter()
    for e in edges:
        valence[e[0]] += 1
        valence[e[1]] += 1
    valences = list(valence.values())
    val_counts = Counter(valences)
    
    print(f"\nVertex valence distribution:")
    total_verts = len(mesh.vertices)
    regular_count = 0
    for v in sorted(val_counts.keys()):
        pct = 100 * val_counts[v] / total_verts
        marker = ""
        if v == 6:  # Regular for triangles
            marker = " <- regular (triangles)"
            regular_count += val_counts[v]
        elif v == 4:  # Regular for quads
            marker = " <- regular (quads)"
        print(f"  Valence {v}: {val_counts[v]:5d} ({pct:5.1f}%){marker}")
    
    # Poles analysis
    n3_poles = val_counts.get(3, 0) + val_counts.get(2, 0) + val_counts.get(1, 0)
    n5_poles = val_counts.get(5, 0)
    n6_plus = sum(val_counts.get(v, 0) for v in val_counts if v >= 7)
    
    print(f"\nPole analysis (non-regular vertices):")
    print(f"  3-poles (valence 3 or less): {n3_poles} ({100*n3_poles/total_verts:.1f}%)")
    print(f"  5-poles (valence 5): {n5_poles} ({100*n5_poles/total_verts:.1f}%)")
    print(f"  6+ poles (valence 7+): {n6_plus} ({100*n6_plus/total_verts:.1f}%)")
    
    # Check manifold
    print(f"\nMesh properties:")
    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Euler number: {mesh.euler_number}")
    
    return {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'valence_dist': dict(val_counts),
        'n3_poles': n3_poles,
        'n5_poles': n5_poles,
        'n6_plus_poles': n6_plus,
    }


# Load source mesh via Blender conversion
fbx_path = '/Users/maxdavis/Projects/MeshRepair/test_mesh2.fbx'
blender_path = '/Applications/Blender.app/Contents/MacOS/blender'

print("Converting FBX via Blender...")
with tempfile.TemporaryDirectory() as tmpdir:
    obj_path = Path(tmpdir) / 'converted.obj'
    script = f'''
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"{fbx_path}")
bpy.ops.wm.obj_export(filepath=r"{obj_path}", export_selected_objects=False, export_triangulated_mesh=False)
'''
    script_path = Path(tmpdir) / 'convert.py'
    script_path.write_text(script)
    subprocess.run([blender_path, '--background', '--python', str(script_path)], 
                   capture_output=True, timeout=60)
    source = trimesh.load(obj_path, force='mesh')

# Load retopo mesh
retopo = trimesh.load('/Users/maxdavis/Projects/MeshRepair/results/20260119_191055/test_mesh2_retopo.obj', force='mesh')

# Analyze both
source_stats = analyze_mesh_topology(source, "SOURCE MESH (test_mesh2.fbx)")
retopo_stats = analyze_mesh_topology(retopo, "RETOPO OUTPUT")

# Summary comparison
print(f"\n{'='*60}")
print(f"  COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"Face reduction: {source_stats['faces']} -> {retopo_stats['faces']} ({100*retopo_stats['faces']/source_stats['faces']:.1f}%)")
print(f"Vertex reduction: {source_stats['vertices']} -> {retopo_stats['vertices']} ({100*retopo_stats['vertices']/source_stats['vertices']:.1f}%)")

# Quality comparison
src_irregular = source_stats['n3_poles'] + source_stats['n5_poles'] + source_stats['n6_plus_poles']
ret_irregular = retopo_stats['n3_poles'] + retopo_stats['n5_poles'] + retopo_stats['n6_plus_poles']
print(f"\nIrregular vertex ratio:")
print(f"  Source: {100*src_irregular/source_stats['vertices']:.1f}%")
print(f"  Retopo: {100*ret_irregular/retopo_stats['vertices']:.1f}%")

print(f"\nHigh-valence poles (6+, problematic):")
print(f"  Source: {source_stats['n6_plus_poles']} ({100*source_stats['n6_plus_poles']/source_stats['vertices']:.1f}%)")
print(f"  Retopo: {retopo_stats['n6_plus_poles']} ({100*retopo_stats['n6_plus_poles']/retopo_stats['vertices']:.1f}%)")
