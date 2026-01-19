#!/usr/bin/env python3
"""
Development Iteration Script

This script is designed for rapid iteration during development.
It runs the pipeline, evaluates results, and suggests improvements.

Usage:
    python dev_iterate.py [mesh_path] [--cycles N]
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class IterationResult:
    """Result from one iteration cycle."""
    cycle: int
    score: float
    quad_score: float
    fidelity_score: float
    topology_score: float
    faces: int
    time: float
    config: dict


def run_iteration_cycle(
    mesh_path: Optional[str] = None,
    num_cycles: int = 5,
    save_all: bool = True
) -> list[IterationResult]:
    """
    Run multiple iteration cycles with different configurations.
    
    Each cycle tests a different configuration to explore the parameter space.
    """
    from meshretopo import Mesh
    from meshretopo.test_meshes import create_sphere, create_torus
    from meshretopo.pipeline import RetopoPipeline
    from meshretopo.experiments import ExperimentConfig
    
    # Load or create input mesh
    if mesh_path:
        input_mesh = Mesh.from_file(mesh_path)
        print(f"Loaded: {mesh_path}")
    else:
        # Default test mesh
        input_mesh = create_sphere(subdivisions=3)
        print("Using default sphere mesh")
    
    print(f"Input: {input_mesh.num_vertices} vertices, {input_mesh.num_faces} faces")
    print(f"Running {num_cycles} iteration cycles...\n")
    
    # Different configurations to try
    configs = [
        {
            "name": "baseline",
            "backend": "trimesh",
            "neural_weight": 0.7,
            "feature_weight": 0.3,
            "target_reduction": 0.1,
        },
        {
            "name": "hybrid_balanced",
            "backend": "hybrid", 
            "neural_weight": 0.7,
            "feature_weight": 0.3,
            "target_reduction": 0.25,
        },
        {
            "name": "hybrid_conservative",
            "backend": "hybrid",
            "neural_weight": 0.6,
            "feature_weight": 0.4,
            "target_reduction": 0.2,
        },
        {
            "name": "aggressive_reduction",
            "backend": "trimesh",
            "neural_weight": 0.7,
            "feature_weight": 0.3,
            "target_reduction": 0.05,
        },
        {
            "name": "conservative",
            "backend": "trimesh",
            "neural_weight": 0.6,
            "feature_weight": 0.4,
            "target_reduction": 0.2,
        },
        {
            "name": "guided_quad",
            "backend": "guided_quad",
            "neural_weight": 0.7,
            "feature_weight": 0.3,
            "target_reduction": 0.1,
        },
        {
            "name": "hybrid_aggressive",
            "backend": "hybrid",
            "neural_weight": 0.8,
            "feature_weight": 0.2,
            "target_reduction": 0.1,
        },
    ]
    
    results = []
    output_dir = Path("dev_iterations")
    output_dir.mkdir(exist_ok=True)
    
    for i in range(min(num_cycles, len(configs))):
        config = configs[i]
        print(f"Cycle {i+1}/{num_cycles}: {config['name']}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            pipeline = RetopoPipeline(
                backend=config["backend"],
                target_faces=int(input_mesh.num_faces * config["target_reduction"]),
                neural_weight=config["neural_weight"],
                feature_weight=config["feature_weight"],
            )
            
            output_mesh, score = pipeline.process(input_mesh, evaluate=True)
            
            elapsed = time.time() - start_time
            
            result = IterationResult(
                cycle=i + 1,
                score=score.overall_score,
                quad_score=score.quad_score,
                fidelity_score=score.fidelity_score,
                topology_score=score.topology_score,
                faces=output_mesh.num_faces,
                time=elapsed,
                config=config,
            )
            results.append(result)
            
            print(f"  Score: {score.overall_score:.1f}/100")
            print(f"  Quad: {score.quad_score:.1f} | Fidelity: {score.fidelity_score:.1f} | Topology: {score.topology_score:.1f}")
            print(f"  Faces: {output_mesh.num_faces} | Time: {elapsed:.2f}s")
            
            if save_all:
                output_mesh.to_file(output_dir / f"cycle_{i+1}_{config['name']}.obj")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(IterationResult(
                cycle=i + 1,
                score=0,
                quad_score=0,
                fidelity_score=0,
                topology_score=0,
                faces=0,
                time=0,
                config=config,
            ))
        
        print()
    
    return results


def analyze_results(results: list[IterationResult]) -> None:
    """Analyze iteration results and suggest improvements."""
    print("=" * 60)
    print("ITERATION ANALYSIS")
    print("=" * 60)
    
    if not results:
        print("No results to analyze")
        return
    
    # Find best result
    valid_results = [r for r in results if r.score > 0]
    if not valid_results:
        print("All iterations failed!")
        return
    
    best = max(valid_results, key=lambda r: r.score)
    
    print(f"\nBest configuration: {best.config['name']}")
    print(f"  Overall Score: {best.score:.1f}/100")
    print(f"  Config: {best.config}")
    
    # Identify weak points
    print("\nWeak Points Analysis:")
    
    if best.quad_score < 60:
        print("  ⚠️  QUAD QUALITY is low ({:.1f})".format(best.quad_score))
        print("      Suggestions:")
        print("      - Try 'guided_quad' backend for better quad generation")
        print("      - Increase optimization_iterations")
        print("      - Reduce target face count for simpler topology")
    
    if best.fidelity_score < 60:
        print("  ⚠️  GEOMETRIC FIDELITY is low ({:.1f})".format(best.fidelity_score))
        print("      Suggestions:")
        print("      - Reduce target_reduction (keep more faces)")
        print("      - Increase feature_weight to preserve details")
        print("      - Enable preserve_boundary")
    
    if best.topology_score < 60:
        print("  ⚠️  TOPOLOGY QUALITY is low ({:.1f})".format(best.topology_score))
        print("      Suggestions:")
        print("      - Check for non-manifold edges in input")
        print("      - Try different remeshing backend")
    
    # Score improvement over cycles
    print("\nScore Progression:")
    for r in results:
        bar_len = int(r.score / 2)  # Scale to 50 chars max
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  Cycle {r.cycle}: [{bar}] {r.score:.1f}")
    
    # Suggest next experiments
    print("\nSuggested Next Experiments:")
    
    if best.config["neural_weight"] > 0.6:
        print("  1. Try LOWER neural_weight ({:.1f} → {:.1f})".format(
            best.config["neural_weight"], best.config["neural_weight"] - 0.2))
    else:
        print("  1. Try HIGHER neural_weight ({:.1f} → {:.1f})".format(
            best.config["neural_weight"], best.config["neural_weight"] + 0.2))
    
    if best.config["target_reduction"] > 0.1:
        print("  2. Try more aggressive reduction (current: {:.0f}%)".format(
            best.config["target_reduction"] * 100))
    else:
        print("  2. Try less aggressive reduction (current: {:.0f}%)".format(
            best.config["target_reduction"] * 100))
    
    print("  3. Try 'guided_quad' backend for native quad output")
    print("  4. Add curvature_sensitivity sweep (0.5, 1.0, 1.5, 2.0)")


def main():
    parser = argparse.ArgumentParser(description="Development iteration script")
    parser.add_argument("mesh", nargs="?", help="Input mesh path (optional)")
    parser.add_argument("--cycles", "-c", type=int, default=5, help="Number of cycles")
    parser.add_argument("--no-save", action="store_true", help="Don't save intermediate meshes")
    
    args = parser.parse_args()
    
    results = run_iteration_cycle(
        mesh_path=args.mesh,
        num_cycles=args.cycles,
        save_all=not args.no_save
    )
    
    analyze_results(results)


if __name__ == "__main__":
    main()
