"""Test neural network integration in the remesh pipeline."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from neurotopo.core.mesh import Mesh
from neurotopo.core.fields import ScalarField, FieldLocation
from neurotopo.remesh.hybrid import HybridRemesher

# Check if we have output from previous runs
output_dirs = [
    "outputs",
    "results",
    "quick_start_output",
]

mesh_file = None
for d in output_dirs:
    dir_path = Path(d)
    if dir_path.exists():
        for f in dir_path.rglob("*.obj"):
            # Look for a retopo output
            if "retopo" in f.name.lower() or "output" in f.name.lower():
                mesh_file = f
                break
    if mesh_file:
        break

# Fall back to a test mesh
if not mesh_file:
    mesh_file = Path("test_meshes/torus.obj")
    
print(f"Using mesh: {mesh_file}")

# Load mesh
mesh = Mesh.from_file(mesh_file)
print(f"Loaded mesh: {mesh.num_vertices} verts, {mesh.num_faces} faces")
print(f"Is quad: {mesh.is_quad}")

# Check quad quality
if mesh.is_quad:
    remesher = HybridRemesher()
    quality = remesher._assess_quad_quality(mesh)
    print(f"Quad quality (regular verts): {quality:.1%}")

# Test neural network classifier directly
print("\n=== Testing Neural Network Classifier ===")
from neurotopo.analysis.neural.pole_classifier import HybridPoleClassifier

model_path = Path("models/pole_classifier.pt")
if model_path.exists():
    classifier = HybridPoleClassifier(model_path=str(model_path))
    
    poles = classifier.classify_poles(mesh)
    fix_count = sum(1 for p in poles if p.prediction == 'fix')
    keep_count = sum(1 for p in poles if p.prediction == 'keep')
    
    print(f"Total poles: {len(poles)}")
    print(f"  - To fix: {fix_count}")
    print(f"  - To keep: {keep_count}")
    
    # Show some examples
    fix_poles = [p for p in poles if p.prediction == 'fix'][:5]
    keep_poles = [p for p in poles if p.prediction == 'keep'][:5]
    
    print("\nPoles to fix:")
    for p in fix_poles:
        print(f"  V{p.valence} at idx={p.vertex_idx}, curv={p.curvature:.4f}, "
              f"boundary={p.is_boundary}, conf={p.confidence:.2f}")
    
    print("\nPoles to keep:")
    for p in keep_poles:
        print(f"  V{p.valence} at idx={p.vertex_idx}, curv={p.curvature:.4f}, "
              f"boundary={p.is_boundary}, conf={p.confidence:.2f}")
else:
    print(f"Model not found at {model_path}")
    classifier = HybridPoleClassifier()  # Rule-based fallback
    
print("\n=== Neural network integration test complete ===")
