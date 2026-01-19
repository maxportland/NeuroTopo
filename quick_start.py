#!/usr/bin/env python3
"""
Quick Start Script for NeuroTopo

This script demonstrates the full pipeline and helps verify installation.
Run this after installing the package to test everything works.
"""

import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("NeuroTopo Quick Start")
    print("=" * 60)
    
    # Add src to path for development
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    from neurotopo import Mesh, RetopoPipeline
    from neurotopo.test_meshes import create_sphere, create_torus, get_test_meshes
    from neurotopo.evaluation import evaluate_retopology
    
    # Create output directory
    output_dir = Path("quick_start_output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n1. Generating test meshes...")
    print("-" * 40)
    
    # Create test meshes
    sphere = create_sphere(subdivisions=3)
    torus = create_torus(major_segments=48, minor_segments=24)
    
    print(f"   Sphere: {sphere.num_vertices} vertices, {sphere.num_faces} faces")
    print(f"   Torus:  {torus.num_vertices} vertices, {torus.num_faces} faces")
    
    # Save originals
    sphere.to_file(output_dir / "sphere_original.obj")
    torus.to_file(output_dir / "torus_original.obj")
    
    print("\n2. Running retopology pipeline...")
    print("-" * 40)
    
    # Create pipeline
    pipeline = RetopoPipeline(
        backend="trimesh",
        target_faces=500,  # Aggressive reduction for demo
        neural_weight=0.7,
        feature_weight=0.3
    )
    
    # Process sphere
    print("\n   Processing sphere...")
    try:
        sphere_out, sphere_score = pipeline.process(sphere, evaluate=True)
        print(f"   Output: {sphere_out.num_vertices} vertices, {sphere_out.num_faces} faces")
        print(f"   Score:  {sphere_score.overall_score:.1f}/100")
        sphere_out.to_file(output_dir / "sphere_retopo.obj")
    except Exception as e:
        print(f"   Error processing sphere: {e}")
        sphere_score = None
    
    # Process torus
    print("\n   Processing torus...")
    try:
        torus_out, torus_score = pipeline.process(torus, evaluate=True)
        print(f"   Output: {torus_out.num_vertices} vertices, {torus_out.num_faces} faces")
        print(f"   Score:  {torus_score.overall_score:.1f}/100")
        torus_out.to_file(output_dir / "torus_retopo.obj")
    except Exception as e:
        print(f"   Error processing torus: {e}")
        torus_score = None
    
    print("\n3. Quality Analysis...")
    print("-" * 40)
    
    if sphere_score:
        print("\n   Sphere Quality Breakdown:")
        print(f"     - Quad Quality:  {sphere_score.quad_score:.1f}/100")
        print(f"     - Fidelity:      {sphere_score.fidelity_score:.1f}/100")
        print(f"     - Topology:      {sphere_score.topology_score:.1f}/100")
    
    if torus_score:
        print("\n   Torus Quality Breakdown:")
        print(f"     - Quad Quality:  {torus_score.quad_score:.1f}/100")
        print(f"     - Fidelity:      {torus_score.fidelity_score:.1f}/100")
        print(f"     - Topology:      {torus_score.topology_score:.1f}/100")
    
    print("\n4. Testing iterative improvement...")
    print("-" * 40)
    
    try:
        small_sphere = create_sphere(subdivisions=2)
        print(f"   Input: {small_sphere.num_faces} faces")
        
        best_mesh, best_score = pipeline.iterate(
            small_sphere,
            target_score=60.0,  # Lower target for demo
            max_iterations=3,
            target_faces=200
        )
        
        print(f"   Best result: {best_score.overall_score:.1f}/100")
        best_mesh.to_file(output_dir / "sphere_iterative.obj")
    except Exception as e:
        print(f"   Iteration test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review the output meshes in a 3D viewer")
    print("  2. Try different configurations in experiments/")
    print("  3. Run: neurotopo --help for CLI options")
    print("  4. Modify neural analyzers in src/neurotopo/analysis/neural/")


if __name__ == "__main__":
    main()
