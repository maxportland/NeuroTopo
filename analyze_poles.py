#!/usr/bin/env python3
"""Analyze pole distribution in retopo output mesh."""

import subprocess
import tempfile
from pathlib import Path

blender_path = "/Applications/Blender.app/Contents/MacOS/blender"
input_mesh = "/Users/maxdavis/Projects/MeshRepair/results/20260121_011004/test_mesh2_retopo.obj"

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    
    script = '''
import bpy
import bmesh

# Import mesh
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"''' + input_mesh + '''")

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.verts.ensure_lookup_table()

# Analyze vertex valences
valence_counts = {}
for v in bm.verts:
    val = len(v.link_edges)
    valence_counts[val] = valence_counts.get(val, 0) + 1

print("Valence distribution:")
for val in sorted(valence_counts.keys()):
    count = valence_counts[val]
    pct = 100 * count / len(bm.verts)
    print(f"  Valence {val}: {count} ({pct:.1f}%)")

# Find irregular poles and analyze them
def get_valence(v):
    return len(v.link_edges)

def is_on_boundary(v):
    return any(e.is_boundary for e in v.link_edges)

def vertex_curvature_estimate(v):
    if len(v.link_faces) < 2:
        return 1.0
    normals = [f.normal for f in v.link_faces]
    if len(normals) < 2:
        return 0.0
    total = 0
    count = 0
    for i, n1 in enumerate(normals):
        for n2 in normals[i+1:]:
            total += n1.dot(n2)
            count += 1
    if count == 0:
        return 0.0
    avg_dot = total / count
    return 1.0 - max(0, avg_dot)

irregular_flat = []
irregular_curved = []

for v in bm.verts:
    valence = get_valence(v)
    if valence == 4:
        continue
    if is_on_boundary(v):
        continue
    
    curv = vertex_curvature_estimate(v)
    
    if curv < 0.15:
        irregular_flat.append((v.index, valence, curv))
    else:
        irregular_curved.append((v.index, valence, curv))

print(f"")
print(f"Irregular vertices in FLAT areas: {len(irregular_flat)}")
print(f"Irregular vertices in CURVED areas: {len(irregular_curved)}")

# Count by valence
flat_by_val = {}
for idx, val, curv in irregular_flat:
    flat_by_val[val] = flat_by_val.get(val, 0) + 1
print(f"")
print("Flat-area poles by valence:")
for val in sorted(flat_by_val.keys()):
    print(f"  Valence {val}: {flat_by_val[val]}")
'''
    
    script_path = tmpdir / "analyze.py"
    script_path.write_text(script)
    
    result = subprocess.run(
        [blender_path, "--background", "--python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Extract print statements
    for line in result.stdout.split('\n'):
        if any(x in line for x in ['Valence', 'Irregular', 'Vertex', 'Flat', 'distribution']):
            print(line)
