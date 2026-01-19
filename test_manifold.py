#!/usr/bin/env python3
"""Analyze mesh structure."""
from meshretopo.core.mesh import Mesh
from meshretopo import RetopoPipeline

mesh = Mesh.from_file('test_mesh.obj')
pipeline = RetopoPipeline(backend='hybrid', target_faces=5000)
output, _ = pipeline.process(mesh, enable_timing=False)

print(f"Total faces: {len(output.faces)}")

true_quads = sum(1 for f in output.faces if len(set(f)) == 4)
degen = sum(1 for f in output.faces if len(f) == 4 and len(set(f)) == 3)
print(f"True quads: {true_quads}")
print(f"Degenerate (triangles): {degen}")

# Check if manifold
import trimesh
tm = trimesh.Trimesh(vertices=output.vertices, faces=output.faces, process=False)
print(f"\nTrimesh analysis:")
print(f"  Is watertight: {tm.is_watertight}")
print(f"  Is manifold (trimesh): Not directly available")

# Check for degenerate faces
degen_count = 0
for face in output.faces:
    if len(set(face)) < 3:
        degen_count += 1
print(f"  Degenerate faces (< 3 unique): {degen_count}")

# Check edge manifoldness
edge_count = {}
for face in output.faces:
    unique = list(set(face))
    n = len(unique)
    for i in range(n):
        e = tuple(sorted([unique[i], unique[(i+1) % n]]))
        edge_count[e] = edge_count.get(e, 0) + 1

non_manifold_edges = sum(1 for c in edge_count.values() if c > 2)
boundary_edges = sum(1 for c in edge_count.values() if c == 1)
print(f"  Non-manifold edges (>2 faces): {non_manifold_edges}")
print(f"  Boundary edges (1 face): {boundary_edges}")
