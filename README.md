# NeuroTopo: AI-Assisted Retopology

Neural-guided, deterministically-controlled mesh retopology for production pipelines with GPT-4o visual quality assessment.

## Current Performance (v0.4)

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
neurotopo process input.obj --target-faces 500 --output output.obj

# Use hybrid backend for quad output
neurotopo process input.obj -t 500 -b hybrid -o output.obj

# Auto-tune parameters for best quality
python -c "
from neurotopo import auto_retopo, load_mesh
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
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Semantic Segmentation (GPT-4o Vision)                  │    │
│  │  • Region detection (face, body, clothing, etc.)        │    │
│  │  • Mesh type classification                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GUIDANCE GENERATION                            │
│  • Quad size field (curvature-adaptive)                         │
│  • Edge flow direction field                                     │
│  • Feature importance map                                        │
│  • Symmetry detection                                            │
│  • AI-guided region constraints                                  │
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
│                    EVALUATION & QA                               │
│  ┌─────────────────────────────┐  ┌───────────────────────────┐ │
│  │  Geometric Metrics          │  │  AI Quality Assessment    │ │
│  │  • Quad quality (aspect)    │  │  (GPT-4o Vision Analysis) │ │
│  │  • Hausdorff distance       │  │  • Edge flow quality      │ │
│  │  • Edge flow score          │  │  • Pole placement review  │ │
│  │  • Pole placement           │  │  • Deformation prediction │ │
│  └─────────────────────────────┘  └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run retopology on a mesh
neurotopo process input.obj --target-faces 5000 --output output.obj

# Run evaluation
neurotopo evaluate original.obj retopo.obj

# Run experiment suite
neurotopo experiment --config experiments/baseline.yaml

# Interactive iteration mode
neurotopo iterate input.obj --cycles 5
```

## Project Structure

```
src/neurotopo/
├── core/                    # Core mesh data structures
│   ├── mesh.py             # Unified mesh representation
│   └── fields.py           # Scalar/vector fields on meshes
├── analysis/               # Neural + classical analysis
│   ├── curvature.py        # Curvature computation
│   ├── features.py         # Feature detection (edges, corners)
│   ├── semantic.py         # AI semantic segmentation (GPT-4o)
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
│   ├── scorer.py           # Combined scoring
│   └── ai_quality.py       # AI visual quality assessment (GPT-4o)
├── visualization/          # Result visualization
│   ├── viewer.py           # 3D mesh visualization
│   └── blender_render.py   # Blender-based rendering
├── utils/                  # Utilities
│   └── keychain.py         # macOS Keychain integration
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
from neurotopo import auto_retopo, load_mesh

mesh = load_mesh('input.obj')
output, score = auto_retopo(mesh, time_budget=60.0)
print(f'Score: {score.overall_score:.1f}')
```

### Pipeline API
```python
from neurotopo import RetopoPipeline, load_mesh

mesh = load_mesh('input.obj')
pipeline = RetopoPipeline(
    backend='hybrid',          # 'trimesh', 'hybrid', 'guided_quad'
    target_faces=1000,
    neural_weight=0.6,         # Blend neural vs classical guidance
)
output, score = pipeline.process(mesh, evaluate=True)
print(score.summary())
```

### AI Quality Assessment (NEW in v0.4)

Visual quality analysis using GPT-4o to evaluate retopology results:

```python
from neurotopo.evaluation import AIQualityAssessor, assess_mesh_quality

# Quick assessment
report = assess_mesh_quality(
    mesh_path='retopo.obj',
    mesh_type='character',
    use_cache=True
)
print(f'AI Score: {report.overall_score}/100')
for issue in report.issues:
    print(f'  [{issue.severity.value}] {issue.description}')

# Detailed assessment with custom renderer
assessor = AIQualityAssessor(provider='openai')  # or 'anthropic'
report = assessor.assess(
    images=render_images,  # Multi-angle renders
    mesh_type='character',
    context={'target_use': 'game_asset', 'deformation': True}
)
```

Features:
- **Edge Flow Analysis**: Detects poor loop placement around joints and deformation areas
- **Pole Placement Review**: Identifies problematic pole positions
- **Density Evaluation**: Checks for appropriate polygon distribution
- **Deformation Prediction**: Assesses how well topology will animate

### Semantic Segmentation (NEW in v0.4)

AI-powered region detection for intelligent retopology constraints:

```python
from neurotopo.analysis import SemanticSegmenter

segmenter = SemanticSegmenter()
regions = segmenter.analyze(mesh_path='character.obj')
# Returns: face, body, hands, clothing regions with constraints
```

### Quality Metrics
- **Quad Quality**: Aspect ratio, angle deviation, valence distribution
- **Fidelity**: Hausdorff distance, mean error, normal deviation  
- **Topology**: Manifold check, boundary count, genus
- **AI Assessment**: Visual quality score with specific issue detection

## Configuration

### API Key Setup (macOS)

Store your OpenAI API key securely in macOS Keychain:

```bash
python scripts/store_api_key.py
```

Or manually:
```bash
security add-generic-password -a "$USER" -s "NeuroTopo" -w "your-api-key"
```

### Caching

Results are cached to `~/.cache/neurotopo/` to reduce API costs:
- `renders/` - Blender-rendered mesh images
- `api_responses/` - AI analysis results

Clear cache:
```bash
rm -rf ~/.cache/neurotopo/
```

## Testing

### Test Harness

JSON-config driven test runner for comprehensive testing:

```bash
# Run all tests
python tests/test_harness.py

# Run specific suite
python tests/test_harness.py --suite ai_quality

# Available suites: quick, full, ai_quality, full_with_ai
```

Configure tests in `tests/test_config.json`:
```json
{
  "test_suites": {
    "ai_quality": ["retopology", "ai_quality_assessment"]
  },
  "thresholds": {
    "min_ai_quality_score": 50.0
  }
}
```
## Requirements

### Core
- Python 3.10+
- trimesh, numpy, scipy

### AI Features
- OpenAI API key (for GPT-4o vision)
- Blender 4.0+ (for mesh rendering)
- macOS Keychain (for secure API key storage)

### Optional
- anthropic (for Claude as alternative AI provider)

## Documentation

See [docs/AI_INTEGRATION.md](docs/AI_INTEGRATION.md) for the full AI integration roadmap including:
- AI Quality Assessment (implemented)
- Edge Flow Direction Guidance (planned)
- Mesh Type Auto-Classification (planned)
- Iterative Refinement Loop (planned)

## License

MIT