#!/usr/bin/env python
"""NeuroTopo system validation script."""

def main():
    print('=' * 60)
    print('NeuroTopo System Validation')
    print('=' * 60)
    print()

    # Test imports
    print('Testing imports...')
    from neurotopo import RetopoPipeline, auto_retopo
    from neurotopo.core.mesh import Mesh
    from neurotopo.evaluation import MeshEvaluator
    from neurotopo.tuning import AutoTuner
    from neurotopo.remesh import get_remesher, HybridRemesher
    from neurotopo.postprocess import QuadOptimizer
    from neurotopo.visualization import visualize_mesh
    print('  All imports OK')
    print()

    # Test mesh creation
    print('Testing mesh creation...')
    from neurotopo.test_meshes import create_sphere, create_cube, create_bunny_like
    sphere = create_sphere(subdivisions=3)
    cube = create_cube(subdivisions=2)
    bunny = create_bunny_like()
    print(f'  Sphere: {sphere.num_faces} faces')
    print(f'  Cube: {cube.num_faces} faces')
    print(f'  Bunny: {bunny.num_faces} faces')
    print()

    # Test pipeline
    print('Testing pipeline...')
    pipeline = RetopoPipeline(backend='trimesh', target_faces=300)
    output, score = pipeline.process(sphere, evaluate=True)
    print(f'  Trimesh: {sphere.num_faces} -> {output.num_faces} faces, score={score.overall_score:.1f}')

    pipeline = RetopoPipeline(backend='hybrid', target_faces=200)
    output, score = pipeline.process(sphere, evaluate=True)
    print(f'  Hybrid: {sphere.num_faces} -> {output.num_faces} faces, score={score.overall_score:.1f}')
    print()

    # Test auto-tuner
    print('Testing auto-tuner...')
    tuner = AutoTuner(max_iterations=5, time_limit=20.0)
    result = tuner.tune(cube, verbose=False)
    print(f'  Best config: {result.best_config}')
    print(f'  Best score: {result.best_score:.1f}')
    print()

    # Test evaluator
    print('Testing evaluator...')
    evaluator = MeshEvaluator()
    score = evaluator.evaluate(sphere, sphere)
    print(f'  Self-comparison score: {score.overall_score:.1f}')
    print()

    print('=' * 60)
    print('ALL TESTS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()
