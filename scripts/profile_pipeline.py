#!/usr/bin/env python
"""Profile the retopology pipeline to find bottlenecks."""

import time
from meshretopo.core.io import load_mesh
from meshretopo.analysis import detect_features, create_default_analyzer
from meshretopo.guidance import GuidanceComposer
from meshretopo.remesh import get_remesher


def profile_pipeline(mesh_path: str, reduction: float = 0.02):
    """Profile each stage of the pipeline."""
    
    print("=" * 60)
    print("Pipeline Profiling")
    print("=" * 60)
    
    # Load mesh
    print("\nLoading mesh...")
    start = time.time()
    mesh = load_mesh(mesh_path)
    load_time = time.time() - start
    print(f"Load time: {load_time:.1f}s")
    print(f"Input: {mesh.num_faces} faces, {mesh.num_vertices} vertices")
    
    target = int(mesh.num_faces * reduction)
    print(f"Target: {target} faces ({reduction*100:.0f}% reduction)")
    
    # Analysis
    print("\nRunning analysis...")
    start = time.time()
    analyzer = create_default_analyzer()
    prediction = analyzer.predict(mesh)
    analysis_time = time.time() - start
    print(f"Analysis time: {analysis_time:.1f}s")
    
    # Feature detection
    print("\nDetecting features...")
    start = time.time()
    features = detect_features(mesh)
    feature_time = time.time() - start
    print(f"Feature time: {feature_time:.1f}s")
    print(f"Feature edges: {features.num_feature_edges}")
    
    # Guidance composition
    print("\nComposing guidance...")
    start = time.time()
    composer = GuidanceComposer()
    guidance = composer.compose(mesh, prediction, features, target_faces=target)
    compose_time = time.time() - start
    print(f"Compose time: {compose_time:.1f}s")
    
    # Remeshing
    print("\nRemeshing...")
    start = time.time()
    remesher = get_remesher('trimesh')
    result = remesher.remesh(mesh, guidance)
    remesh_time = time.time() - start
    print(f"Remesh time: {remesh_time:.1f}s")
    print(f"Output: {result.mesh.num_faces} faces")
    
    # Summary
    total = load_time + analysis_time + feature_time + compose_time + remesh_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Load:      {load_time:6.1f}s  ({load_time/total*100:5.1f}%)")
    print(f"  Analysis:  {analysis_time:6.1f}s  ({analysis_time/total*100:5.1f}%)")
    print(f"  Features:  {feature_time:6.1f}s  ({feature_time/total*100:5.1f}%)")
    print(f"  Guidance:  {compose_time:6.1f}s  ({compose_time/total*100:5.1f}%)")
    print(f"  Remeshing: {remesh_time:6.1f}s  ({remesh_time/total*100:5.1f}%)")
    print(f"  TOTAL:     {total:6.1f}s")


if __name__ == '__main__':
    import sys
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/maxdavis/Projects/MeshRepair/test_mesh.fbx'
    profile_pipeline(mesh_path)
