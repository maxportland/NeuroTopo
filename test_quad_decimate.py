#!/usr/bin/env python3
"""Test quad-preserving decimation."""

import sys
sys.path.insert(0, '/Users/maxdavis/Projects/MeshRepair/src')

import numpy as np
from collections import Counter, defaultdict

def load_obj_as_quads(path):
    """Load OBJ preserving quads."""
    vertices = []
    faces = []
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'f':
            faces.append([int(x.split('/')[0])-1 for x in parts[1:]])
    return np.array(vertices, dtype=np.float32), faces

def analyze_topology(vertices, faces, name):
    """Analyze topology."""
    vertex_edges = defaultdict(int)
    seen_edges = set()
    
    for face in faces:
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i+1) % n]
            if v0 != v1:
                edge = (min(v0, v1), max(v0, v1))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    vertex_edges[v0] += 1
                    vertex_edges[v1] += 1
    
    valence = Counter(vertex_edges.values())
    total = len(vertices)
    regular = valence.get(4, 0)
    high_poles = sum(valence.get(v, 0) for v in range(6, max(valence.keys())+1) if v in valence)
    
    quads = sum(1 for f in faces if len(f) == 4 and len(set(f)) == 4)
    tris = sum(1 for f in faces if len(f) == 3 or (len(f) == 4 and len(set(f)) < 4))
    
    print(f"\n{name}")
    print(f"  Vertices: {total}, Faces: {len(faces)}")
    print(f"  Quads: {quads}, Tris: {tris}")
    print(f"  Regular (V4): {regular} ({100*regular/total:.1f}%)")
    print(f"  High poles (V6+): {high_poles} ({100*high_poles/total:.1f}%)")
    
    return 100*regular/total, 100*high_poles/total

# Load source mesh
print("Loading source mesh...")
verts, faces = load_obj_as_quads('/tmp/source_mesh_quads.obj')
print(f"Loaded: {len(verts)} vertices, {len(faces)} faces")

# Analyze original
print("\n=== BEFORE DECIMATION ===")
analyze_topology(verts, faces, "Original")

# Apply quad-preserving decimation
print("\n=== APPLYING QUAD-PRESERVING DECIMATION ===")
from neurotopo.remesh.quad_decimate import QuadDecimator, QuadDecimateConfig

config = QuadDecimateConfig(
    target_face_ratio=0.20,  # 20% of original
    preserve_boundary=True,
    max_valence=6,
)

decimator = QuadDecimator(config)
new_verts, new_faces = decimator.decimate(verts, faces)

print("\n=== AFTER DECIMATION ===")
analyze_topology(new_verts, new_faces, "Decimated")

# Save result
output_path = '/tmp/quad_decimated.obj'
with open(output_path, 'w') as f:
    f.write("# Quad-decimated mesh\n")
    for v in new_verts:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for face in new_faces:
        indices = ' '.join(str(v+1) for v in face)  # OBJ is 1-indexed
        f.write(f"f {indices}\n")
print(f"\nSaved to {output_path}")
