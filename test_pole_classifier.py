#!/usr/bin/env python3
"""Test the pole classifier on the retopo output mesh."""

import sys
sys.path.insert(0, '/Users/maxdavis/Projects/MeshRepair/src')

from neurotopo.core.mesh import Mesh
from neurotopo.analysis.neural.pole_classifier import (
    HybridPoleClassifier, 
    PoleFeatureExtractor,
    TORCH_AVAILABLE
)

print(f"PyTorch available: {TORCH_AVAILABLE}")

# Load the retopo output mesh
mesh_path = "/Users/maxdavis/Projects/MeshRepair/results/20260121_011004/test_mesh2_retopo.obj"
mesh = Mesh.from_file(mesh_path)
print(f"Loaded mesh: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

# Classify poles with the trained model
model_path = "/Users/maxdavis/Projects/MeshRepair/models/pole_classifier.pt"
classifier = HybridPoleClassifier(model_path=model_path)
poles = classifier.classify_poles(mesh)

print(f"\nTotal irregular vertices: {len(poles)}")

# Count by classification
fix_poles = [p for p in poles if p.prediction == 'fix']
keep_poles = [p for p in poles if p.prediction == 'keep']

print(f"  - Should FIX: {len(fix_poles)}")
print(f"  - Should KEEP: {len(keep_poles)}")

# Breakdown by valence
print("\nBy valence:")
for valence in [2, 3, 5, 6, 7]:
    fix_v = [p for p in fix_poles if p.valence == valence]
    keep_v = [p for p in keep_poles if p.valence == valence]
    if fix_v or keep_v:
        print(f"  V{valence}: {len(fix_v)} fix, {len(keep_v)} keep")

# High confidence fixes
high_conf_fixes = [p for p in fix_poles if p.confidence >= 0.8]
print(f"\nHigh-confidence fixes (conf >= 0.8): {len(high_conf_fixes)}")

# Show some examples
print("\nExample fixable poles (high confidence):")
for p in high_conf_fixes[:10]:
    print(f"  Vertex {p.vertex_idx}: V{p.valence}, curv={p.curvature:.4f}, "
          f"neighbors={p.neighbor_valences}, conf={p.confidence:.2f}")

# Show kept poles in flat areas (potential false negatives)
flat_keeps = [p for p in keep_poles if p.curvature < 0.1]
print(f"\nKept poles in flat areas (potential false negatives): {len(flat_keeps)}")
for p in flat_keeps[:5]:
    print(f"  Vertex {p.vertex_idx}: V{p.valence}, curv={p.curvature:.4f}, "
          f"boundary={p.is_boundary}, conf={p.confidence:.2f}")
