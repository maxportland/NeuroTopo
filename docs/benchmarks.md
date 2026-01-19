# Benchmark Results

## Current Performance (2026-01-19)

### Procedural Test Meshes

| Mesh | Faces | Best Backend | Reduction | Overall | Quad | Fidelity | Topology |
|------|-------|--------------|-----------|---------|------|----------|----------|
| Sphere | 1280 | trimesh | 50% | **64.7** | 45.0 | 66.7 | 100 |
| Cube | 48 | trimesh | 60% | **64.7** | 53.8 | 57.8 | 100 |
| Torus | 1024 | trimesh | 40% | **60.5** | 40.5 | 60.7 | 100 |
| Cylinder | 320 | trimesh | 40% | 52.7 | 27.0 | 54.8 | 100 |
| Cone | 512 | trimesh | 50% | 53.2 | 14.2 | 68.7 | 100 |
| Bunny | 1280 | trimesh | 50% | **65.9** | 47.3 | 67.4 | 100 |
| Mechanical | 384 | trimesh | 40% | 53.1 | 24.8 | 57.8 | 100 |

**Average Score: 59.2/100**

### Real-World Meshes

| Mesh | Input Faces | Output Faces | Reduction | Score | Time |
|------|-------------|--------------|-----------|-------|------|
| test_mesh.fbx | 256,108 | 2,876 | 1% | 47.5 | **7.4s** |

**Performance Improvements (v0.3.1):**
- Vectorized curvature computation: 180s → 0.05s
- Vectorized feature detection: 170s → 0.64s
- Total pipeline speedup: **25.7x** (189s → 7.4s)

**Quality Observations:**
- Real mesh scores ~10 points lower than procedural meshes
- Fidelity remains good (66-67) but quad quality is ~27
- Output is triangular - quad conversion needed for higher scores

---

## Historical Progress

### Version 0.3.0 (2026-01-19)
- Average: **59.2** (+4.6 from v0.2)
- Added auto-tuning system
- Improved scoring calibration
- Better triangle angle handling (60° ideal)
- Smarter reduction ratios based on mesh size

### Version 0.2.0 (Previous)
- Average: **54.6**
- Added hybrid backend
- Added isotropic remesher
- Basic quad optimizer

### Version 0.1.0 (Initial)
- Average: ~45
- Trimesh backend only
- Basic pipeline

---

## Score Breakdown

### Overall Score Formula
```
overall = 0.4 * quad + 0.4 * fidelity + 0.2 * topology
```

### Quad Quality Score
- **Aspect Ratio** (35%): 1.0 ideal, penalized up to 2.0+
- **Angle Deviation** (45%): 0° ideal (90° for quads, 60° for tris)
- **Irregular Vertices** (20%): % of non-valence-4 vertices

### Fidelity Score
- **Hausdorff Distance** (30%): Max surface deviation
- **Mean Distance** (30%): Average surface deviation
- **Normal Deviation** (15%): Angle between normals
- **Coverage** (25%): % of original surface covered

### Topology Score
- **Manifold** (50%): 100 if manifold, 0 otherwise
- **Boundaries** (50%): 100 if ≤1 boundary, penalized otherwise

---

## Backend Comparison

### On Sphere (1280 faces → 640 faces)

| Backend | Overall | Quad | Fidelity | Time |
|---------|---------|------|----------|------|
| trimesh | 64.7 | 45.0 | 66.7 | 0.2s |
| hybrid | 48.4 | 21.3 | 49.7 | 0.5s |
| guided_quad | ~55 | ~30 | ~60 | 1.2s |
| isotropic | ~40 | ~20 | ~45 | 2.0s |

### Observations
1. **Trimesh** consistently best for overall score
2. **Hybrid** produces quads but lower fidelity
3. **Guided_quad** good for animation use cases
4. **Isotropic** needs more iterations for quality

---

## Known Issues

1. **Cone quad quality**: Low (14.2) due to radial topology near tip
2. **Hybrid fidelity**: Often lower than trimesh
3. **Small meshes**: Need higher reduction ratios (50-60%)

---

## Improvement Targets

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Average Overall | 59.2 | 70+ | High |
| Quad Score | 36.1 | 50+ | Medium |
| Fidelity Score | 61.9 | 70+ | High |
| Real-world mesh | TBD | 60+ | High |
