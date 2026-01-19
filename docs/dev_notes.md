# Development Notes

Ongoing development log and improvement ideas for MeshRetopo.

---

## 2026-01-19: Session 2 - Timing & Logging

### Added
1. ✅ Created `utils/timing.py` module with:
   - `timed_operation` context manager
   - `TimeoutError` exception for long-running ops
   - `TimingLog` for accumulating timing info
   - `ProgressTimer` for iterative operations
   - Configurable timeouts per operation type
   - `run_with_timeout()` for thread-based timeout enforcement

2. ✅ Integrated timing into all major components:
   - Pipeline: analysis, feature_detection, guidance, remeshing, evaluation
   - AutoTuner: per-iteration timing with optional timeout enforcement
   - Curvature analyzer: per-computation logging
   - Feature detector: detection timing
   - Remeshers: trimesh, hybrid backends
   - Evaluator: quad quality, fidelity, topology timing

3. ✅ Added timeout enforcement capability:
   - Pipeline supports `enforce_timeouts` parameter
   - AutoTuner supports `enforce_iteration_timeouts`
   - Uses thread-based timeout (portable, works on all platforms)
   - Configurable per-instance or globally via `configure_timeouts()`

### Default Timeouts
```python
TIMEOUT_CURVATURE = 30.0s
TIMEOUT_FEATURES = 30.0s
TIMEOUT_REMESH = 120.0s
TIMEOUT_EVALUATION = 60.0s
TIMEOUT_OPTIMIZATION = 60.0s
TIMEOUT_AUTOTUNER_ITERATION = 30.0s
```

### Usage
```python
# Enable timing logs
import logging
logging.basicConfig(level=logging.INFO)

# Run with timing (logs timing, doesn't kill on timeout)
from meshretopo import RetopoPipeline
pipeline = RetopoPipeline(backend='trimesh', target_faces=1000)
output, score = pipeline.process(mesh, enable_timing=True)

# Run with timeout enforcement (kills operations that exceed timeout)
pipeline.enforce_timeouts = True
pipeline.timeout_analysis = 10.0  # Custom 10s timeout for analysis
output, score = pipeline.process(mesh)

# Configure global timeouts
from meshretopo.utils.timing import configure_timeouts
configure_timeouts(curvature=15.0, features=15.0, remesh=60.0)

# Get timing summary
from meshretopo.utils.timing import get_timing_log
print(get_timing_log().summary())
```

---

## 2026-01-19: Session 1 Summary

### Completed Today
1. ✅ Created auto-tuning system (`tuning/autotuner.py`)
   - Grid search over backends, neural_weight, feature_weight, reduction
   - Local refinement around best config
   - `auto_retopo()` convenience function

2. ✅ Improved scoring calibration (`evaluation/metrics.py`)
   - Fixed triangle angle deviation (60° ideal, was 90°)
   - Better aspect ratio scaling
   - Recalibrated fidelity thresholds

3. ✅ Added feature-aware remeshing (`remesh/feature_aware.py`)
   - FeaturePreservingRemesher
   - AdaptiveDensityRemesher
   - BoundaryPreservingRemesher

4. ✅ Improved tri-to-quad conversion (`remesh/tri_to_quad.py`)
   - Better quality scoring with planarity
   - Direction field alignment support

5. ✅ Created documentation structure

### Current Scores
- **Average: 59.2/100** (up from 54.6)
- Best: Bunny at 65.9
- Weakest: Cone at 53.2 (radial topology issue)

### Next Steps
- [ ] Test with real high-poly mesh (test_mesh.fbx)
- [ ] Improve hybrid backend fidelity
- [ ] Add adaptive decimation based on curvature
- [ ] Implement better quad vertex ordering

---

## Improvement Ideas

### High Priority
1. **Curvature-Adaptive Decimation**
   - Preserve more detail in high-curvature areas
   - Use curvature analysis to weight decimation error

2. **Better Quad Conversion**
   - Current hybrid has lower fidelity than trimesh
   - Consider subdivision + simplification approach
   - Look into Catmull-Clark + decimation

3. **Feature Edge Preservation**
   - Sharp edges often lost during decimation
   - Lock feature vertices during optimization
   - Snap to feature edges after remeshing

### Medium Priority
4. **Direction Field Integration**
   - Edge flow optimizer exists but not well integrated
   - Use curvature directions for guidance

5. **UV-Aware Remeshing**
   - Preserve UV seams as feature edges
   - Important for textured assets

6. **Symmetry Detection**
   - Detect and preserve mesh symmetry
   - Important for characters/mechanical parts

### Low Priority (Future)
7. **Neural Network Training**
   - Train edge flow predictor on professional retopo data
   - Requires curated training dataset

8. **Multi-Resolution**
   - Support different detail levels
   - LOD generation

---

## Technical Debt

1. **Degenerate Quad Handling**
   - Some quads have repeated vertices (triangles stored as quads)
   - Need cleaner face type handling

2. **Error Handling**
   - Some backends fail silently
   - Need better error propagation

3. **Memory Usage**
   - Large meshes may cause issues
   - Need streaming/chunked processing

---

## Architecture Notes

### Why Trimesh Outperforms Hybrid
1. Trimesh uses quadric error metrics - optimizes for surface fidelity
2. Hybrid converts to quads after decimation - may create poor aspect ratios
3. Quad optimization can't fully recover from bad initial conversion

### Potential Fixes
1. Use isotropic remeshing BEFORE quad conversion
2. Better triangle pairing scoring
3. Multiple passes: decimate → optimize → decimate

---

## Test Results Log

### test_mesh.fbx Results (2026-01-19)

**After Performance Optimization:**
```
Input: 256,108 faces, 128,056 vertices
Diagonal: 2.73 units

1% reduction → 2,876 faces
  Score: 47.5
  Time: 7.4s  (was 189s - 25.7x speedup!)
```

**Optimization Details:**
1. Curvature computation: 180s → 0.05s (vectorized)
2. Feature detection: 170s → 0.64s (vectorized)
3. Total pipeline: 189s → 7.4s (25.7x speedup)

**Before Optimization:**
```
5% reduction → 11,342 faces
  Score: 47.6 (Q:26.4 F:67.5)
  Time: 192s

2% reduction → 4,494 faces  
  Score: 47.4 (Q:27.0 F:66.5)
  Time: 192s

1% reduction → 2,876 faces
  Score: 47.5 (Q:26.6 F:67.1)
  Time: 189s
```

**Analysis:**
- Fidelity is good (66-67) indicating geometry preserved well
- Quad quality is low (26-27) - triangles have poor aspect ratios
- ~~Processing time dominated by evaluation~~ FIXED: Analysis was bottleneck
- Performance now suitable for interactive use

**Remaining issues:**
1. Quad quality score is low (~27 vs 40+ target)
2. Output is triangles, not quads - need better quad conversion

---

## References

- Botsch & Kobbelt, "A Remeshing Approach to Multiresolution Modeling"
- Instant Meshes paper (quad-dominant remeshing)
- QuadriFlow (scalable quad meshing)
