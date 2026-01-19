# API Reference

## Core Classes

### Mesh

The fundamental mesh data structure.

```python
from meshretopo.core.mesh import Mesh

# Create from arrays
mesh = Mesh(
    vertices=np.array([[0,0,0], [1,0,0], [0,1,0]]),
    faces=[[0, 1, 2]],
    name="triangle"
)

# Properties
mesh.num_vertices  # Number of vertices
mesh.num_faces     # Number of faces
mesh.is_manifold   # True if manifold topology
mesh.is_closed     # True if no boundary edges
mesh.is_triangular # True if all faces are triangles
mesh.is_quad       # True if all faces are quads
mesh.diagonal      # Bounding box diagonal length
mesh.total_area    # Total surface area

# Methods
mesh.compute_normals()  # Compute vertex normals
mesh.triangulate()      # Convert to triangles
mesh.copy()             # Deep copy
```

### RetopoPipeline

Main entry point for retopology.

```python
from meshretopo import RetopoPipeline

pipeline = RetopoPipeline(
    backend='trimesh',       # 'trimesh', 'hybrid', 'guided_quad', 'isotropic'
    target_faces=1000,       # Desired output face count
    neural_weight=0.5,       # Weight for neural guidance (0-1)
    feature_weight=0.5,      # Weight for feature preservation (0-1)
    preserve_boundary=True,  # Keep boundary edges
    edge_flow_optimization=False,  # Post-process edge alignment
)

# Process mesh
output_mesh, score = pipeline.process(
    input_mesh,
    evaluate=True  # Compute quality metrics
)

# Access results
print(f"Output: {output_mesh.num_faces} faces")
print(f"Score: {score.overall_score:.1f}/100")
print(score.summary())
```

### Auto-Tuning

Automatic parameter optimization.

```python
from meshretopo import auto_retopo
from meshretopo.tuning import AutoTuner

# Simple usage
output, score = auto_retopo(
    mesh,
    target_faces=None,     # Auto-determine
    time_budget=60.0,      # Max seconds
    verbose=True
)

# Advanced usage
tuner = AutoTuner(
    backends=['trimesh', 'hybrid'],
    neural_weight_range=(0.3, 0.9),
    feature_weight_range=(0.1, 0.7),
    reduction_range=(0.1, 0.4),
    max_iterations=20,
    early_stop_score=75.0,
    time_limit=120.0,
)

result = tuner.tune(mesh, objective='overall', verbose=True)
print(f"Best config: {result.best_config}")
print(f"Best score: {result.best_score}")
```

### Quality Evaluation

```python
from meshretopo.evaluation import MeshEvaluator, RetopologyScore

evaluator = MeshEvaluator(sample_count=10000)
score = evaluator.evaluate(
    retopo_mesh=output,
    original_mesh=input_mesh  # For fidelity metrics
)

# Access individual metrics
score.overall_score      # Weighted combination (0-100)
score.quad_score        # Quad quality (0-100)
score.fidelity_score    # Geometric fidelity (0-100)
score.topology_score    # Topology quality (0-100)

# Detailed metrics
score.quad_quality.aspect_ratio_mean
score.quad_quality.angle_deviation_mean
score.quad_quality.irregular_vertex_ratio

score.geometric_fidelity.hausdorff_distance
score.geometric_fidelity.mean_distance
score.geometric_fidelity.coverage

score.topology.is_manifold
score.topology.num_boundaries
```

## Remeshing Backends

### TrimeshRemesher
Fast triangle decimation using quadric error metrics.

```python
from meshretopo.remesh import TrimeshRemesher

remesher = TrimeshRemesher()
result = remesher.remesh(mesh, guidance, target_faces=500)
```

### HybridRemesher
Combines triangle decimation with quad conversion.

```python
from meshretopo.remesh import HybridRemesher

remesher = HybridRemesher(
    quad_ratio=0.8,           # Target % of quads
    optimization_passes=5,    # Smoothing iterations
    preserve_boundary=True
)
```

### GuidedQuadRemesher
Direction-field guided quad generation.

```python
from meshretopo.remesh import GuidedQuadRemesher

remesher = GuidedQuadRemesher(
    use_direction_field=True,
    smoothing_iterations=3
)
```

## Post-Processing

### QuadOptimizer
Improve quad quality through vertex optimization.

```python
from meshretopo.postprocess import QuadOptimizer

optimizer = QuadOptimizer(
    iterations=10,
    smoothing_weight=0.3,
    angle_weight=0.5,
    edge_weight=0.3,
    surface_weight=0.7,
)

optimized_verts = optimizer.optimize(
    vertices, faces, original_trimesh
)
```

### EdgeFlowOptimizer
Align edges with direction field.

```python
from meshretopo.postprocess import EdgeFlowOptimizer

optimizer = EdgeFlowOptimizer(
    iterations=5,
    alignment_weight=0.4,
    smoothing_weight=0.3
)
```

## Visualization

```python
from meshretopo.visualization import (
    visualize_mesh,
    visualize_quality_heatmap,
    visualize_comparison,
    save_visualization_report
)

# Basic mesh view
visualize_mesh(mesh, title="My Mesh")

# Quality heatmap
visualize_quality_heatmap(mesh, metric='aspect_ratio')

# Before/after comparison
visualize_comparison(original, retopo, title="Comparison")

# Save report
save_visualization_report(original, retopo, score, "output_dir")
```

## File I/O

```python
from meshretopo.core.io import load_mesh, save_mesh

# Load (supports OBJ, PLY, STL, FBX via trimesh)
mesh = load_mesh("model.obj")
mesh = load_mesh("model.fbx")

# Save
save_mesh(mesh, "output.obj")
```

## Test Meshes

```python
from meshretopo.test_meshes import (
    create_sphere,
    create_cube,
    create_torus,
    create_cylinder,
    create_cone,
    create_bunny_like,
    create_mechanical_part,
    create_plane,
)

# Create procedural test meshes
sphere = create_sphere(subdivisions=3)  # 1280 faces
cube = create_cube(subdivisions=2)      # 48 faces
torus = create_torus()                  # 1024 faces
```
