# MeshRetopo: AI-Assisted Retopology

Neural-guided, deterministically-controlled mesh retopology for production pipelines.

## Current Performance (v0.3)

| Mesh | Best Backend | Reduction | Overall | Quad | Fidelity |
|------|--------------|-----------|---------|------|----------|
| Sphere | trimesh | 50% | **64.7** | 45.0 | 66.7 |
| Cube | trimesh | 60% | **64.7** | 53.8 | 57.8 |
| Torus | trimesh | 40% | **60.5** | 40.5 | 60.7 |
| Cylinder | trimesh | 40% | 52.7 | 27.0 | 54.8 |
| Cone | trimesh | 50% | 53.2 | 14.2 | 68.7 |
| Bunny | trimesh | 50% | **65.9** | 47.3 | 67.4 |
| Mechanical | trimesh | 40% | 53.1 | 24.8 | 57.8 |
| **Average** | - | - | **59.2** | 36.1 | 61.9 |

*Benchmark suite on procedural test meshes*

## Quick Start

```bash
# Install
pip install -e .

# Run retopology on a mesh
meshretopo process input.obj --target-faces 500 --output output.obj

# Use hybrid backend for quad output
meshretopo process input.obj -t 500 -b hybrid -o output.obj

# Auto-tune parameters for best quality
python -c "
from meshretopo import auto_retopo, load_mesh
mesh = load_mesh('input.obj')
output, score = auto_retopo(mesh, time_budget=60.0)
print(f'Score: {score.overall_score:.1f}')
"

# Run benchmark suite
python scripts/run_benchmark.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT MESH                               │
│                    (High-poly scan/sculpt)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS PIPELINE                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Curvature  │  │  Feature    │  │  Neural Edge Flow       │  │
│  │  Analysis   │  │  Detection  │  │  Prediction             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GUIDANCE GENERATION                            │
│  • Quad size field (curvature-adaptive)                         │
│  • Edge flow direction field                                     │
│  • Feature importance map                                        │
│  • Symmetry detection                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               DETERMINISTIC REMESHING                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  QuadriFlow │  │  Instant    │  │  Custom Guided          │  │
│  │  Backend    │  │  Meshes     │  │  Remesher               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                               │
│  • Feature snapping                                              │
│  • Vertex optimization                                           │
│  • UV-aware adjustments                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION METRICS                            │
│  • Quad quality (aspect ratio, skewness)                        │
│  • Hausdorff distance to original                               │
│  • Edge flow score                                               │
│  • Pole placement analysis                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run retopology on a mesh
meshretopo process input.obj --target-faces 5000 --output output.obj

# Run evaluation
meshretopo evaluate original.obj retopo.obj

# Run experiment suite
meshretopo experiment --config experiments/baseline.yaml

# Interactive iteration mode
meshretopo iterate input.obj --cycles 5
```

## Project Structure

```
src/meshretopo/
├── core/                    # Core mesh data structures
│   ├── mesh.py             # Unified mesh representation
│   └── fields.py           # Scalar/vector fields on meshes
├── analysis/               # Neural + classical analysis
│   ├── curvature.py        # Curvature computation
│   ├── features.py         # Feature detection (edges, corners)
│   └── neural/             # Neural network modules
│       ├── edge_flow.py    # Edge flow prediction
│       └── sizing.py       # Adaptive sizing prediction
├── guidance/               # Guidance field generation
│   ├── size_field.py       # Quad size guidance
│   ├── direction_field.py  # Edge direction guidance
│   └── composer.py         # Combine multiple guidance sources
├── remesh/                 # Remeshing backends
│   ├── base.py             # Abstract remesher interface
│   ├── trimesh_backend.py  # Trimesh decimation (triangles)
│   ├── hybrid.py           # Triangle-to-quad conversion
│   ├── guided.py           # Guidance-driven quad remesher
│   ├── isotropic.py        # Isotropic edge-based remeshing
│   ├── feature_aware.py    # Feature-preserving remeshing
│   └── tri_to_quad.py      # Triangle pairing algorithms
├── postprocess/            # Post-processing operations
│   └── optimizer.py        # Quad shape optimization
├── evaluation/             # Quality metrics
│   ├── metrics.py          # Quad quality, fidelity, topology
│   └── scorer.py           # Combined scoring
├── visualization/          # Result visualization
│   └── viewer.py           # 3D mesh visualization
├── tuning/                 # Parameter optimization
│   └── autotuner.py        # Grid search, adaptive tuning
├── experiments/            # Experiment framework
│   ├── runner.py           # Experiment runner
│   └── config.py           # Configuration system
├── pipeline.py             # High-level API
├── test_meshes.py          # Procedural test meshes
└── cli.py                  # Command-line interface
```

## Key Features

### Auto-Tuning
```python
from meshretopo import auto_retopo, load_mesh

mesh = load_mesh('input.obj')
output, score = auto_retopo(mesh, time_budget=60.0)
print(f'Score: {score.overall_score:.1f}')
```

### Pipeline API
```python
from meshretopo import RetopoPipeline, load_mesh

mesh = load_mesh('input.obj')
pipeline = RetopoPipeline(
    backend='hybrid',          # 'trimesh', 'hybrid', 'guided_quad'
    target_faces=1000,
    neural_weight=0.6,         # Blend neural vs classical guidance
)
output, score = pipeline.process(mesh, evaluate=True)
print(score.summary())
```

### Quality Metrics
- **Quad Quality**: Aspect ratio, angle deviation, valence distribution
- **Fidelity**: Hausdorff distance, mean error, normal deviation  
- **Topology**: Manifold check, boundary count, genus
