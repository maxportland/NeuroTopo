# System Architecture

## Overview

NeuroTopo follows a pipeline architecture:

```
Input Mesh → Analysis → Guidance → Remeshing → Post-Processing → Evaluation → Output
```

## Module Hierarchy

```
neurotopo/
├── core/               # Foundation layer
│   ├── mesh.py        # Mesh data structure (vertices, faces, properties)
│   ├── fields.py      # Scalar/vector fields on mesh surfaces
│   └── io.py          # File I/O (OBJ, FBX, etc.)
│
├── analysis/          # Mesh analysis
│   ├── curvature.py   # Principal curvature computation
│   ├── features.py    # Sharp edge/corner detection
│   └── neural/        # Neural network analyzers
│       ├── edge_flow.py    # Predict optimal edge directions
│       └── sizing.py       # Predict adaptive sizing
│
├── guidance/          # Guidance field generation
│   ├── size_field.py       # Target quad size per-vertex
│   ├── direction_field.py  # Target edge directions
│   └── composer.py         # Blend multiple guidance sources
│
├── remesh/            # Remeshing backends
│   ├── base.py             # Abstract interface
│   ├── trimesh_backend.py  # Trimesh quadric decimation
│   ├── hybrid.py           # Triangle→quad conversion
│   ├── guided.py           # Direction-field guided quads
│   ├── isotropic.py        # Edge-based isotropic remeshing
│   ├── feature_aware.py    # Feature-preserving operations
│   └── tri_to_quad.py      # Triangle pairing algorithms
│
├── postprocess/       # Quality improvement
│   └── optimizer.py   # QuadOptimizer, EdgeFlowOptimizer
│
├── evaluation/        # Quality metrics
│   ├── metrics.py     # QuadQuality, Fidelity, Topology
│   └── scorer.py      # Combined scoring
│
├── visualization/     # Visual output
│   └── viewer.py      # Matplotlib-based 3D viewing
│
├── tuning/            # Parameter optimization
│   └── autotuner.py   # Grid search, adaptive tuning
│
├── experiments/       # Experiment tracking
│   ├── runner.py      # Run experiments
│   └── config.py      # YAML configuration
│
├── pipeline.py        # High-level RetopoPipeline API
├── test_meshes.py     # Procedural test mesh generators
└── cli.py             # Command-line interface
```

## Data Flow

### 1. Input Stage
- Load mesh from file (OBJ, FBX, etc.)
- Normalize/validate mesh topology
- Compute vertex normals if missing

### 2. Analysis Stage
- **Curvature Analysis**: Compute principal curvatures k1, k2
- **Feature Detection**: Find sharp edges (dihedral angle > threshold)
- **Neural Analysis**: Predict edge flow directions (when available)

### 3. Guidance Generation
- **Size Field**: Smaller quads in high-curvature regions
- **Direction Field**: Align edges with principal curvature
- **Feature Weights**: Preserve important edges/corners

### 4. Remeshing
- Select backend based on requirements
- Apply guidance fields to control output
- Generate target face count

### 5. Post-Processing
- **Quad Optimization**: Laplacian smoothing, surface projection
- **Feature Snapping**: Align vertices to original features
- **Edge Flow Alignment**: Adjust for animation deformation

### 6. Evaluation
- **Quad Quality**: Aspect ratio, angle deviation, valence
- **Fidelity**: Hausdorff distance, normal deviation
- **Topology**: Manifold check, boundary count

## Backend Comparison

| Backend | Output | Speed | Quality | Best For |
|---------|--------|-------|---------|----------|
| trimesh | Triangles | Fast | Good fidelity | General use |
| hybrid | Quads | Medium | Mixed | Quad requirements |
| guided_quad | Quads | Slow | Direction-aware | Animation |
| isotropic | Triangles | Slow | Uniform | Re-triangulation |

## Key Classes

### Mesh (core/mesh.py)
```python
class Mesh:
    vertices: np.ndarray  # Nx3 positions
    faces: list[list[int]]  # Variable-size faces
    normals: np.ndarray  # Per-vertex normals
    
    @property
    def num_faces(self) -> int
    @property
    def is_manifold(self) -> bool
    def triangulate(self) -> Mesh
```

### RetopoPipeline (pipeline.py)
```python
class RetopoPipeline:
    def __init__(self, backend, target_faces, neural_weight, ...)
    def process(self, mesh, evaluate=True) -> (Mesh, RetopologyScore)
```

### RetopologyScore (evaluation/metrics.py)
```python
class RetopologyScore:
    overall_score: float  # 0-100
    quad_score: float
    fidelity_score: float
    topology_score: float
```
