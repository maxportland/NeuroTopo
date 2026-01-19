#!/usr/bin/env python3
"""Analyze valence distribution."""
import numpy as np
from collections import Counter
from neurotopo.core.mesh import Mesh
from neurotopo import RetopoPipeline

mesh = Mesh.from_file('test_mesh.obj')
pipeline = RetopoPipeline(backend='hybrid', target_faces=5000)
output, _ = pipeline.process(mesh, enable_timing=False)

# Compute valence
valence = np.zeros(output.num_vertices, dtype=int)
for face in output.faces:
    for vi in set(face):
        valence[vi] += 1

# Count
counts = Counter(valence)
print("Valence distribution:")
for v in sorted(counts.keys()):
    pct = counts[v] / len(valence) * 100
    print(f"  Valence {v}: {counts[v]} ({pct:.1f}%)")

print(f"\nIrregular (not 4): {sum(counts[v] for v in counts if v != 4)} ({(1 - counts.get(4, 0)/len(valence))*100:.1f}%)")
