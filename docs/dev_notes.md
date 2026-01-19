# Development Notes

Ongoing development log and improvement ideas for MeshRetopo.

---

## 2026-01-19: Session Summary

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

### test_mesh.fbx (TBD)
```
Input faces: TBD
Output faces: TBD
Score: TBD
Notes: TBD
```

---

## References

- Botsch & Kobbelt, "A Remeshing Approach to Multiresolution Modeling"
- Instant Meshes paper (quad-dominant remeshing)
- QuadriFlow (scalable quad meshing)
