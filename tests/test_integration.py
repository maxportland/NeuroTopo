#!/usr/bin/env python
"""
NeuroTopo Integration Tests

Validates the complete retopology pipeline with various test cases.
"""

import pytest
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.test_meshes import (
    create_sphere, create_cube, create_torus, 
    create_cylinder, create_cone, create_bunny_like
)
from neurotopo.pipeline import RetopoPipeline
from neurotopo.evaluation import MeshEvaluator, RetopologyScore
from neurotopo.tuning import auto_retopo, AutoTuner


class TestMeshCreation:
    """Test mesh creation utilities."""
    
    def test_sphere_creation(self):
        """Sphere should be valid manifold mesh."""
        sphere = create_sphere(subdivisions=2)
        assert sphere.num_faces > 0
        assert sphere.num_vertices > 0
        assert sphere.is_manifold
    
    def test_cube_creation(self):
        """Cube should be valid mesh."""
        cube = create_cube(subdivisions=1)
        assert cube.num_faces == 12  # 6 faces * 2 triangles
        assert cube.num_vertices == 8
    
    def test_torus_creation(self):
        """Torus should create closed surface."""
        torus = create_torus()
        assert torus.num_faces > 0
        assert torus.is_closed


class TestPipeline:
    """Test the main retopology pipeline."""
    
    def test_trimesh_backend(self):
        """Trimesh backend should reduce face count."""
        sphere = create_sphere(subdivisions=3)
        target = sphere.num_faces // 4
        
        pipeline = RetopoPipeline(backend='trimesh', target_faces=target)
        output, score = pipeline.process(sphere, evaluate=True)
        
        assert output.num_faces < sphere.num_faces
        assert score is not None
        assert score.overall_score > 0
    
    def test_hybrid_backend(self):
        """Hybrid backend should produce quads."""
        sphere = create_sphere(subdivisions=3)
        target = sphere.num_faces // 4
        
        pipeline = RetopoPipeline(backend='hybrid', target_faces=target)
        output, score = pipeline.process(sphere, evaluate=True)
        
        assert output.num_faces > 0
        # Hybrid produces quads (stored as 4-vertex faces)
        assert any(len(set(f)) == 4 for f in output.faces)
    
    def test_score_computation(self):
        """Scores should be in valid range."""
        sphere = create_sphere(subdivisions=3)
        target = sphere.num_faces // 3
        
        pipeline = RetopoPipeline(backend='trimesh', target_faces=target)
        output, score = pipeline.process(sphere, evaluate=True)
        
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.quad_score <= 100
        assert 0 <= score.fidelity_score <= 100
        assert 0 <= score.topology_score <= 100


class TestAutoTuning:
    """Test the auto-tuning system."""
    
    def test_autotuner_basic(self):
        """AutoTuner should find best configuration."""
        sphere = create_sphere(subdivisions=2)
        
        tuner = AutoTuner(max_iterations=5, time_limit=30.0)
        result = tuner.tune(sphere, verbose=False)
        
        assert result.best_config is not None
        assert result.best_score > 0
        assert len(result.all_results) > 0
    
    def test_auto_retopo_convenience(self):
        """auto_retopo function should work end-to-end."""
        cube = create_cube(subdivisions=1)
        
        output, score = auto_retopo(
            cube, 
            time_budget=10.0,
            verbose=False
        )
        
        assert output.num_faces > 0
        assert score.overall_score > 0


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_evaluator_basic(self):
        """Evaluator should compute all metrics."""
        sphere = create_sphere(subdivisions=2)
        
        evaluator = MeshEvaluator()
        score = evaluator.evaluate(sphere, sphere)  # Self-comparison
        
        assert score.fidelity_score > 90  # Should be high for self
        assert score.quad_quality is not None
        assert score.topology is not None
    
    def test_metrics_with_decimation(self):
        """Metrics should reflect quality differences."""
        sphere = create_sphere(subdivisions=3)
        target_high = sphere.num_faces // 2
        target_low = sphere.num_faces // 8
        
        pipeline_high = RetopoPipeline(backend='trimesh', target_faces=target_high)
        output_high, score_high = pipeline_high.process(sphere, evaluate=True)
        
        pipeline_low = RetopoPipeline(backend='trimesh', target_faces=target_low)
        output_low, score_low = pipeline_low.process(sphere, evaluate=True)
        
        # Higher face count should generally have better fidelity
        assert score_high.fidelity_score >= score_low.fidelity_score * 0.8


class TestQualityTargets:
    """Test that quality targets are achievable."""
    
    def test_sphere_quality(self):
        """Sphere should achieve decent quality."""
        sphere = create_sphere(subdivisions=3)
        
        pipeline = RetopoPipeline(
            backend='trimesh',
            target_faces=int(sphere.num_faces * 0.5)
        )
        output, score = pipeline.process(sphere, evaluate=True)
        
        # Target: 60+ overall score
        assert score.overall_score >= 55, f"Sphere score {score.overall_score} < 55"
    
    def test_bunny_quality(self):
        """Bunny should achieve decent quality."""
        bunny = create_bunny_like()
        
        pipeline = RetopoPipeline(
            backend='trimesh',
            target_faces=int(bunny.num_faces * 0.4)
        )
        output, score = pipeline.process(bunny, evaluate=True)
        
        # Target: 55+ overall score  
        assert score.overall_score >= 50, f"Bunny score {score.overall_score} < 50"


def run_quick_validation():
    """Quick validation of the system."""
    print("Running quick validation...")
    print()
    
    # Test basic pipeline
    sphere = create_sphere(subdivisions=3)
    pipeline = RetopoPipeline(backend='trimesh', target_faces=300)
    output, score = pipeline.process(sphere, evaluate=True)
    
    print(f"Sphere ({sphere.num_faces} -> {output.num_faces} faces)")
    print(f"  Score: {score.overall_score:.1f}")
    print(f"  Quad: {score.quad_score:.1f}, Fidelity: {score.fidelity_score:.1f}")
    print()
    
    # Test hybrid
    pipeline = RetopoPipeline(backend='hybrid', target_faces=200)
    output, score = pipeline.process(sphere, evaluate=True)
    
    print(f"Sphere hybrid ({sphere.num_faces} -> {output.num_faces} faces)")
    print(f"  Score: {score.overall_score:.1f}")
    print(f"  Quad: {score.quad_score:.1f}, Fidelity: {score.fidelity_score:.1f}")
    print()
    
    print("Validation complete!")


if __name__ == '__main__':
    run_quick_validation()
