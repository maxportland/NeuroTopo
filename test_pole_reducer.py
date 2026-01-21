#!/usr/bin/env python3
"""Test pole reducer on retopo output."""

import sys
sys.path.insert(0, '/Users/maxdavis/Projects/MeshRepair/src')

import numpy as np
from collections import Counter

def load_obj_as_mixed(obj_path):
    """Load OBJ preserving quads and triangles."""
    vertices = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1
                    face_verts.append(idx)
                faces.append(face_verts)
    
    return np.array(vertices, dtype=np.float32), faces

def analyze_valence(vertices, faces, name):
    """Analyze valence distribution."""
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i + 1) % n]
            if v0 != v1:
                edges.add((min(v0, v1), max(v0, v1)))
    
    valence = Counter()
    for v in range(len(vertices)):
        val = sum(1 for e in edges if v in e)
        valence[val] += 1
    
    total = len(vertices)
    print(f"\n{name}")
    print(f"  Vertices: {total}, Faces: {len(faces)}")
    
    high_poles = sum(valence.get(v, 0) for v in range(6, max(valence.keys())+1))
    regular = valence.get(4, 0)
    
    print(f"  Regular (V4): {regular} ({100*regular/total:.1f}%)")
    print(f"  High poles (V6+): {high_poles} ({100*high_poles/total:.1f}%)")
    
    return valence

# Load mesh
obj_path = "/Users/maxdavis/Projects/MeshRepair/results/20260119_192039/test_mesh2_retopo.obj"
vertices, faces = load_obj_as_mixed(obj_path)

print("=== BEFORE POLE REDUCTION ===")
analyze_valence(vertices, faces, "Original")

# Apply pole reducer
from neurotopo.postprocess.pole_reduction import PoleReducer

# Convert face list to array (pole reducer needs homogeneous array)
max_size = max(len(f) for f in faces)
faces_arr = np.zeros((len(faces), max_size), dtype=np.int32)
for i, f in enumerate(faces):
    faces_arr[i, :len(f)] = f
    if len(f) < max_size:
        faces_arr[i, len(f):] = f[-1]

print(f"\nApplying pole reduction...")
reducer = PoleReducer(
    max_valence=5,  # Reduce vertices with valence > 5
    iterations=5,   # More iterations
    preserve_boundary=True,
)
new_verts, new_faces = reducer.reduce(vertices, faces_arr)

# Convert back to face list for analysis
new_face_list = []
for f in new_faces:
    unique = list(dict.fromkeys(f))
    new_face_list.append(unique)

print("\n=== AFTER POLE REDUCTION ===")
analyze_valence(new_verts, new_face_list, "After PoleReducer")
