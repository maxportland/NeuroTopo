"""Test neural network on the 86.6% quality mesh."""

from neurotopo.core.mesh import Mesh
from neurotopo.analysis.neural.pole_classifier import HybridPoleClassifier

mesh = Mesh.from_file('results/20260121_011004/test_mesh2_retopo.obj')
print(f'Mesh: {mesh.num_vertices} verts, {mesh.num_faces} faces')

classifier = HybridPoleClassifier(model_path='models/pole_classifier.pt')
poles = classifier.classify_poles(mesh)

print(f'Total irregular vertices: {len(poles)}')

# By valence breakdown
by_valence = {}
for p in poles:
    key = (p.valence, p.prediction)
    by_valence[key] = by_valence.get(key, 0) + 1

print('\nBreakdown by valence:')
for (val, pred), count in sorted(by_valence.items()):
    print(f'  V{val} {pred}: {count}')

# Show high-confidence fixes
fix_high_conf = [p for p in poles if p.prediction == 'fix' and p.confidence >= 0.8]
print(f'\nHigh-confidence fixes: {len(fix_high_conf)}')

# Count total predictions
fix_count = sum(1 for p in poles if p.prediction == 'fix')
keep_count = sum(1 for p in poles if p.prediction == 'keep')
print(f'\nTotal to fix: {fix_count}')
print(f'Total to keep: {keep_count}')

# Show some kept poles
kept = [p for p in poles if p.prediction == 'keep']
print(f'\nSample kept poles:')
for p in kept[:5]:
    print(f'  V{p.valence} curv={p.curvature:.4f} boundary={p.is_boundary} conf={p.confidence:.2f}')
