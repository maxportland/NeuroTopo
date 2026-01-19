#!/usr/bin/env python3
"""Analyze quad quality distribution."""
import numpy as np
from meshretopo.core.mesh import Mesh
from meshretopo import RetopoPipeline

mesh = Mesh.from_file('test_mesh.obj')
pipeline = RetopoPipeline(backend='hybrid', target_faces=5000)
output, _ = pipeline.process(mesh, enable_timing=False)

# Compute AR for ALL true quads
ars = []
angle_devs = []

for face in output.faces:
    if len(face) == 4 and len(set(face)) == 4:
        verts = output.vertices[face]
        edges = [np.linalg.norm(verts[(i+1)%4] - verts[i]) for i in range(4)]
        if min(edges) > 1e-10:
            ar = max(edges) / min(edges)
        else:
            ar = 10.0
        ars.append(ar)
        
        for i in range(4):
            p0 = verts[(i-1) % 4]
            p1 = verts[i]
            p2 = verts[(i+1) % 4]
            e1 = p0 - p1
            e2 = p2 - p1
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                angle = np.arccos(cos_a)
            else:
                angle = np.pi/2
            angle_devs.append(abs(angle - np.pi/2))

ars = np.array(ars)
angle_devs = np.array(angle_devs)

print(f'Total true quads: {len(ars)}')
print(f'\nAspect Ratio distribution:')
print(f'  Min: {ars.min():.2f}')
print(f'  25%: {np.percentile(ars, 25):.2f}')
print(f'  50%: {np.percentile(ars, 50):.2f}')
print(f'  75%: {np.percentile(ars, 75):.2f}')
print(f'  90%: {np.percentile(ars, 90):.2f}')
print(f'  95%: {np.percentile(ars, 95):.2f}')
print(f'  Max: {ars.max():.2f}')
print(f'  Mean: {ars.mean():.2f}')

print(f'\nBad quads (AR > 3): {np.sum(ars > 3)} ({100*np.sum(ars > 3)/len(ars):.1f}%)')
print(f'Good quads (AR < 2): {np.sum(ars < 2)} ({100*np.sum(ars < 2)/len(ars):.1f}%)')

print(f'\nAngle deviation (degrees):')
angle_deg = np.degrees(angle_devs)
print(f'  Mean: {angle_deg.mean():.1f}')
print(f'  50%: {np.percentile(angle_deg, 50):.1f}')
print(f'  90%: {np.percentile(angle_deg, 90):.1f}')
