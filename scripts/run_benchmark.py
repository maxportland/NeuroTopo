#!/usr/bin/env python
"""Comprehensive benchmark script for MeshRetopo system."""

from meshretopo.test_meshes import (
    create_sphere, create_cube, create_torus, create_cylinder,
    create_cone, create_bunny_like, create_mechanical_part
)
from meshretopo.pipeline import RetopoPipeline


def run_benchmark():
    """Run comprehensive benchmark on all test meshes."""
    # Test meshes
    meshes = {
        'Sphere': create_sphere(subdivisions=3),
        'Cube': create_cube(subdivisions=2),
        'Torus': create_torus(),
        'Cylinder': create_cylinder(),
        'Cone': create_cone(),
        'Bunny': create_bunny_like(),
        'Mechanical': create_mechanical_part(),
    }

    print('=' * 80)
    print('COMPREHENSIVE BENCHMARK')
    print('=' * 80)
    print()

    results = {}
    for name, mesh in meshes.items():
        print(f'Testing {name} ({mesh.num_faces} faces)...')
        
        best_score = 0
        best_config = None
        
        # Use smarter reduction ratios based on mesh size
        if mesh.num_faces < 100:
            reductions = [0.5, 0.6, 0.7, 0.8]  # Less reduction for small meshes
        else:
            reductions = [0.2, 0.3, 0.4, 0.5]
        
        for backend in ['trimesh', 'hybrid']:
            for reduction in reductions:
                try:
                    target = int(mesh.num_faces * reduction)
                    if target < 10:
                        target = 10
                    
                    pipeline = RetopoPipeline(backend=backend, target_faces=target)
                    out, score = pipeline.process(mesh, evaluate=True)
                    
                    if score.overall_score > best_score:
                        best_score = score.overall_score
                        best_config = (backend, reduction, score)
                except Exception as e:
                    print(f'    Error with {backend} @ {reduction}: {e}')
        
        results[name] = best_config
        if best_config:
            b, r, s = best_config
            print(f'  Best: {b} @ {r:.0%} -> Score: {s.overall_score:.1f} '
                  f'(Q:{s.quad_score:.1f} F:{s.fidelity_score:.1f})')
        print()

    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f"{'Mesh':<12} {'Backend':<10} {'Reduce':<8} {'Overall':<10} {'Quad':<8} {'Fidelity':<10}")
    print('-' * 60)

    total_score = 0
    count = 0
    for name, config in results.items():
        if config:
            b, r, s = config
            print(f'{name:<12} {b:<10} {r:<8.0%} {s.overall_score:<10.1f} '
                  f'{s.quad_score:<8.1f} {s.fidelity_score:<10.1f}')
            total_score += s.overall_score
            count += 1

    if count > 0:
        avg_score = total_score / count
        print('-' * 60)
        print(f'Average Score: {avg_score:.1f}')


if __name__ == '__main__':
    run_benchmark()
