# Configuration Guide

## Pipeline Parameters

### RetopoPipeline

```python
pipeline = RetopoPipeline(
    backend='trimesh',          # Remeshing algorithm
    target_faces=1000,          # Desired output faces
    neural_weight=0.5,          # Neural vs classical guidance blend
    feature_weight=0.5,         # Feature preservation strength
    preserve_boundary=True,     # Keep boundary edges intact
    edge_flow_optimization=False,  # Post-process edge alignment
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | 'trimesh' | Remeshing algorithm: 'trimesh', 'hybrid', 'guided_quad', 'isotropic' |
| `target_faces` | int | None | Target face count (None = 25% of input) |
| `neural_weight` | float | 0.5 | Weight for neural guidance (0.0-1.0) |
| `feature_weight` | float | 0.5 | Weight for feature preservation (0.0-1.0) |
| `preserve_boundary` | bool | True | Preserve boundary edges |
| `edge_flow_optimization` | bool | False | Enable edge flow post-processing |

### Recommended Configurations

#### General Use (Best Overall Quality)
```python
pipeline = RetopoPipeline(
    backend='trimesh',
    target_faces=int(input_mesh.num_faces * 0.3),
    neural_weight=0.6,
    feature_weight=0.4,
)
```

#### For Animation (Quad Output)
```python
pipeline = RetopoPipeline(
    backend='hybrid',
    target_faces=int(input_mesh.num_faces * 0.25),
    neural_weight=0.7,
    feature_weight=0.5,
    edge_flow_optimization=True,
)
```

#### For Game Assets (Fast, Good Fidelity)
```python
pipeline = RetopoPipeline(
    backend='trimesh',
    target_faces=500,  # Fixed target
    neural_weight=0.4,
    feature_weight=0.6,
)
```

---

## AutoTuner Configuration

```python
tuner = AutoTuner(
    backends=['trimesh', 'hybrid'],
    neural_weight_range=(0.3, 0.9),
    feature_weight_range=(0.1, 0.7),
    reduction_range=(0.1, 0.4),
    max_iterations=20,
    early_stop_score=75.0,
    time_limit=120.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backends` | list | ['trimesh', 'hybrid'] | Backends to try |
| `neural_weight_range` | tuple | (0.3, 0.9) | Range for neural weight |
| `feature_weight_range` | tuple | (0.1, 0.7) | Range for feature weight |
| `reduction_range` | tuple | (0.1, 0.4) | Range for face reduction |
| `max_iterations` | int | 20 | Max configurations to try |
| `early_stop_score` | float | 75.0 | Stop if score exceeds this |
| `time_limit` | float | 120.0 | Max seconds |

---

## Backend-Specific Options

### TrimeshRemesher
No additional options - uses trimesh's quadric decimation.

### HybridRemesher
```python
remesher = HybridRemesher(
    quad_ratio=0.8,           # Target % quads (vs triangles)
    optimization_passes=5,    # Vertex smoothing iterations
    preserve_boundary=True,   # Keep boundary vertices
)
```

### GuidedQuadRemesher
```python
remesher = GuidedQuadRemesher(
    use_direction_field=True,
    smoothing_iterations=3,
)
```

### IsotropicRemesher
```python
remesher = IsotropicRemesher(
    iterations=5,
    target_edge_length=None,  # Auto-compute from target faces
    adaptive=True,            # Use curvature-adaptive sizing
    smoothing_weight=0.5,
    preserve_boundary=True,
)
```

---

## Scoring Weights

The overall score is computed as:
```
overall = 0.4 * quad + 0.4 * fidelity + 0.2 * topology
```

To modify weights:
```python
score.weights = {
    "quad": 0.3,      # Less emphasis on quad quality
    "fidelity": 0.5,  # More emphasis on fidelity
    "topology": 0.2,
}
score.compute_scores(reference_diagonal)
```

---

## Feature Detection Thresholds

```python
from meshretopo.analysis.features import FeatureDetector

detector = FeatureDetector(
    mesh,
    angle_threshold=30.0,   # Degrees - edges sharper than this are features
    corner_threshold=60.0,  # Degrees - corners sharper than this are preserved
)
features = detector.detect()
```

---

## Experiment Configuration (YAML)

```yaml
# experiments/config.yaml
name: "baseline_experiment"

pipeline:
  backend: trimesh
  target_reduction: 0.3
  neural_weight: 0.5
  feature_weight: 0.5

meshes:
  - path: "test_meshes/sphere.obj"
  - path: "test_meshes/bunny.obj"

evaluation:
  sample_count: 10000
  
output:
  directory: "outputs/baseline"
  save_meshes: true
  save_metrics: true
```

Run with:
```bash
meshretopo experiment --config experiments/config.yaml
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MESHRETOPO_CACHE_DIR` | `~/.cache/meshretopo` | Cache directory |
| `MESHRETOPO_LOG_LEVEL` | `INFO` | Logging level |
| `MESHRETOPO_NUM_THREADS` | CPU count | Parallel workers |
