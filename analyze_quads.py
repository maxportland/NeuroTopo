#!/usr/bin/env python3
"""Analyze quad quality in detail."""

import importlib
import numpy as np
from meshretopo.core.mesh import Mesh
from meshretopo import RetopoPipeline
import meshretopo.evaluation.metrics as metrics_module
importlib.reload(metrics_module)

mesh = Mesh.from_file('test_mesh.obj')
pipeline = RetopoPipeline(backend='hybrid', target_faces=5000)
output, score = pipeline.process(mesh, enable_timing=False)

# Analyze quad quality
print('=== Quad Quality Analysis ===')
true_quads = [f for f in output.faces if len(set(f)) == 4]
degen_quads = [f for f in output.faces if len(f) == 4 and len(set(f)) == 3]

print(f'Total faces: {len(output.faces)}')
print(f'True quads (4 unique verts): {len(true_quads)}')
print(f'Degenerate quads (repeated vert): {len(degen_quads)}')

# Check aspect ratios of true quads
quad_ars = []
quad_angle_devs = []
for face in true_quads[:100]:  # Sample
    verts = output.vertices[face]
    edges = [np.linalg.norm(verts[(i+1)%4] - verts[i]) for i in range(4)]
    if min(edges) > 1e-10:
        ar = max(edges) / min(edges)
        quad_ars.append(ar)
        
        # Angles
        for i in range(4):
            e1 = verts[(i-1)%4] - verts[i]
            e2 = verts[(i+1)%4] - verts[i]
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(e1, e2)/(n1*n2), -1, 1)
                angle = np.arccos(cos_a)
                quad_angle_devs.append(abs(angle - np.pi/2))

print(f'\nTrue quad stats (sample of 100):')
print(f'  Aspect ratio: mean={np.mean(quad_ars):.2f}, std={np.std(quad_ars):.2f}')
print(f'  Angle deviation from 90deg: mean={np.degrees(np.mean(quad_angle_devs)):.1f} deg')

# Check triangles (degenerate quads)
tri_ars = []
tri_angle_devs = []
for face in degen_quads[:100]:
    unique = list(set(face))
    if len(unique) == 3:
        verts = output.vertices[unique]
        edges = [np.linalg.norm(verts[(i+1)%3] - verts[i]) for i in range(3)]
        if min(edges) > 1e-10:
            ar = max(edges) / min(edges)
            tri_ars.append(ar)
            
            for i in range(3):
                e1 = verts[(i-1)%3] - verts[i]
                e2 = verts[(i+1)%3] - verts[i]
                n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_a = np.clip(np.dot(e1, e2)/(n1*n2), -1, 1)
                    angle = np.arccos(cos_a)
                    tri_angle_devs.append(abs(angle - np.pi/3))  # 60 deg ideal

print(f'\nDegenerate quad (triangle) stats (sample of 100):')
print(f'  Aspect ratio: mean={np.mean(tri_ars):.2f}')
print(f'  Angle deviation from 60deg: mean={np.degrees(np.mean(tri_angle_devs)):.1f} deg')

# Score breakdown 
print(f'\nScore computed values:')
print(f'  Aspect ratio: {score.quad_quality.aspect_ratio_mean:.2f}')
print(f'  Angle deviation: {np.degrees(score.quad_quality.angle_deviation_mean):.1f} deg')
print(f'  Irregular ratio: {score.quad_quality.irregular_vertex_ratio:.2f}')
print(f'\nFinal quad score: {score.quad_score:.1f}')
