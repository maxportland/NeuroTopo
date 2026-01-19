#!/usr/bin/env python3
"""Test the evaluation fix."""
import numpy as np
from meshretopo.core.mesh import Mesh
from meshretopo import RetopoPipeline
from meshretopo.evaluation.metrics import MeshEvaluator

mesh = Mesh.from_file('test_mesh.obj')
pipeline = RetopoPipeline(backend='hybrid', target_faces=5000)
output, _ = pipeline.process(mesh, enable_timing=False)

# Manual computation
aspect_ratios = []
angle_deviations = []
quad_count = 0
skipped = 0

for face in output.faces:
    unique_verts = len(set(face))
    vertices = output.vertices[face]
    
    if len(face) == 4 and unique_verts == 4:
        quad_count += 1
        edges = [np.linalg.norm(vertices[(i+1)%4] - vertices[i]) for i in range(4)]
        if min(edges) > 1e-10:
            ar = max(edges) / min(edges)
        else:
            ar = 10.0
        aspect_ratios.append(ar)
        
        for i in range(4):
            p0 = vertices[(i-1) % 4]
            p1 = vertices[i]
            p2 = vertices[(i+1) % 4]
            e1 = p0 - p1
            e2 = p2 - p1
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                angle = np.arccos(cos_a)
            else:
                angle = np.pi/2
            angle_deviations.append(abs(angle - np.pi/2))
    elif len(face) == 4 and unique_verts == 3:
        skipped += 1
    elif len(face) == 3:
        pass

print(f'=== Manual computation ===')
print(f'Quad count: {quad_count}')
print(f'Skipped degenerate: {skipped}')
print(f'AR mean: {np.mean(aspect_ratios):.2f}')
print(f'Angle mean: {np.degrees(np.mean(angle_deviations)):.1f} deg')

# Now via evaluator
print(f'\n=== Via MeshEvaluator ===')
evaluator = MeshEvaluator()
score = evaluator.evaluate(output, mesh)
print(f'AR mean: {score.quad_quality.aspect_ratio_mean:.2f}')
print(f'Angle mean: {np.degrees(score.quad_quality.angle_deviation_mean):.1f} deg')
print(f'Quad score: {score.quad_score:.1f}')
