# MeshRetopo Documentation

AI-assisted retopology system for generating clean, production-grade low-poly meshes from high-poly inputs.

## Documentation Index

- [Architecture](architecture.md) - System design and module overview
- [API Reference](api.md) - Core APIs and usage examples
- [Benchmarks](benchmarks.md) - Performance tracking and results
- [Development Notes](dev_notes.md) - Ongoing development log and ideas
- [Configuration](configuration.md) - Pipeline parameters and tuning

## Quick Links

### Getting Started
```python
from meshretopo import RetopoPipeline, auto_retopo
from meshretopo.core.io import load_mesh

# Load a mesh
mesh = load_mesh('input.obj')

# Option 1: Auto-tune for best results
output, score = auto_retopo(mesh, time_budget=60.0)

# Option 2: Manual pipeline
pipeline = RetopoPipeline(
    backend='trimesh',
    target_faces=1000,
)
output, score = pipeline.process(mesh, evaluate=True)
print(score.summary())
```

### Current Best Performance
- **Average Score: 59.2/100** (as of 2026-01-19)
- Best on organic meshes (Bunny: 65.9)
- Trimesh backend currently outperforms hybrid on most cases

### Key Concepts
1. **Backends**: Different remeshing algorithms (trimesh, hybrid, guided_quad, isotropic)
2. **Guidance Fields**: Neural + classical analysis guides remeshing decisions
3. **Quality Metrics**: Quad quality, geometric fidelity, topology scores
4. **Auto-tuning**: Automatic parameter optimization for best results
