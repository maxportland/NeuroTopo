#!/usr/bin/env python3
"""Full pipeline test with timing."""
import time
from neurotopo.core.mesh import Mesh
from neurotopo import RetopoPipeline

print("Loading mesh...")
mesh = Mesh.from_file('test_mesh.obj')
print(f"Input: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

for backend in ['trimesh', 'hybrid']:
    print(f"\n{'='*50}")
    print(f"Running pipeline with {backend} backend...")
    start = time.time()
    pipeline = RetopoPipeline(backend=backend, target_faces=5000)
    output, score = pipeline.process(mesh, enable_timing=False)
    elapsed = time.time() - start

    print(f"\n=== {backend.upper()} Results ===")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Output: {output.num_vertices} vertices, {output.num_faces} faces")
    print(f"\n{score.summary()}")
