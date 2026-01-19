#!/usr/bin/env python3
"""
Benchmark Suite for MeshRetopo

Comprehensive benchmarking across multiple test shapes, backends,
and configurations. Tracks results over time for regression detection.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    timestamp: str
    mesh_name: str
    backend: str
    config_name: str
    input_faces: int
    target_faces: int
    output_faces: int
    overall_score: float
    quad_score: float
    fidelity_score: float
    topology_score: float
    time_seconds: float
    success: bool
    error: Optional[str] = None


class BenchmarkSuite:
    """Runs and tracks benchmarks."""
    
    def __init__(self, results_file: str = "benchmark_results.json"):
        self.results_file = Path(results_file)
        self.results: list[BenchmarkResult] = []
        self._load_history()
    
    def _load_history(self):
        """Load historical results."""
        if self.results_file.exists():
            try:
                with open(self.results_file) as f:
                    data = json.load(f)
                    self.results = [BenchmarkResult(**r) for r in data]
            except Exception:
                self.results = []
    
    def _save_results(self):
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
    
    def run_benchmark(
        self,
        mesh_name: str,
        mesh,
        backend: str,
        config_name: str,
        target_faces: int,
        **config
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        from meshretopo.pipeline import RetopoPipeline
        
        timestamp = datetime.now().isoformat()
        
        try:
            pipeline = RetopoPipeline(
                backend=backend,
                target_faces=target_faces,
                **config
            )
            
            start = time.time()
            output_mesh, score = pipeline.process(mesh, evaluate=True)
            elapsed = time.time() - start
            
            result = BenchmarkResult(
                timestamp=timestamp,
                mesh_name=mesh_name,
                backend=backend,
                config_name=config_name,
                input_faces=mesh.num_faces,
                target_faces=target_faces,
                output_faces=output_mesh.num_faces,
                overall_score=score.overall_score if score else 0,
                quad_score=score.quad_score if score else 0,
                fidelity_score=score.fidelity_score if score else 0,
                topology_score=score.topology_score if score else 0,
                time_seconds=elapsed,
                success=True
            )
            
        except Exception as e:
            result = BenchmarkResult(
                timestamp=timestamp,
                mesh_name=mesh_name,
                backend=backend,
                config_name=config_name,
                input_faces=mesh.num_faces,
                target_faces=target_faces,
                output_faces=0,
                overall_score=0,
                quad_score=0,
                fidelity_score=0,
                topology_score=0,
                time_seconds=0,
                success=False,
                error=str(e)
            )
        
        self.results.append(result)
        self._save_results()
        return result
    
    def run_full_suite(self):
        """Run the complete benchmark suite."""
        from meshretopo.test_meshes import (
            create_sphere, create_torus, create_cube,
            create_cylinder, create_cone
        )
        
        # Test meshes
        meshes = {
            "sphere_low": create_sphere(subdivisions=2),
            "sphere_high": create_sphere(subdivisions=4),
            "torus": create_torus(major_segments=32, minor_segments=16),
            "cube": create_cube(subdivisions=3),
            "cylinder": create_cylinder(radial_segments=32, height_segments=8),
            "cone": create_cone(segments=32, height_segments=8),
        }
        
        # Backend configurations
        configs = [
            {"backend": "trimesh", "config_name": "trimesh_default"},
            {"backend": "isotropic", "config_name": "isotropic_default"},
            {"backend": "guided_quad", "config_name": "guided_quad_default"},
            {"backend": "hybrid", "config_name": "hybrid_default"},
        ]
        
        # Target face ratios
        ratios = [0.1, 0.2, 0.3]
        
        print("=" * 70)
        print("MESHRETOPO BENCHMARK SUITE")
        print("=" * 70)
        print(f"Meshes: {len(meshes)}")
        print(f"Backends: {len(configs)}")
        print(f"Ratios: {ratios}")
        print(f"Total runs: {len(meshes) * len(configs) * len(ratios)}")
        print("=" * 70)
        
        run_results = []
        
        for mesh_name, mesh in meshes.items():
            print(f"\n{mesh_name} ({mesh.num_faces} faces)")
            print("-" * 50)
            
            for config in configs:
                for ratio in ratios:
                    target = max(20, int(mesh.num_faces * ratio))
                    
                    result = self.run_benchmark(
                        mesh_name=mesh_name,
                        mesh=mesh,
                        backend=config["backend"],
                        config_name=config["config_name"],
                        target_faces=target,
                        neural_weight=0.7,
                        feature_weight=0.3
                    )
                    
                    run_results.append(result)
                    
                    status = "✓" if result.success else "✗"
                    print(f"  {status} {config['config_name']:20} r={ratio:.1f} "
                          f"→ {result.overall_score:5.1f} "
                          f"(Q:{result.quad_score:4.1f} F:{result.fidelity_score:4.1f})")
        
        # Summary
        self._print_summary(run_results)
        
        return run_results
    
    def _print_summary(self, results: list[BenchmarkResult]):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        # Group by backend
        by_backend = {}
        for r in results:
            if r.backend not in by_backend:
                by_backend[r.backend] = []
            by_backend[r.backend].append(r)
        
        print("\nAverage Scores by Backend:")
        print("-" * 50)
        
        best_backend = None
        best_score = 0
        
        for backend, backend_results in by_backend.items():
            successful = [r for r in backend_results if r.success]
            if not successful:
                print(f"  {backend:20} - ALL FAILED")
                continue
            
            avg_overall = sum(r.overall_score for r in successful) / len(successful)
            avg_quad = sum(r.quad_score for r in successful) / len(successful)
            avg_fidelity = sum(r.fidelity_score for r in successful) / len(successful)
            avg_time = sum(r.time_seconds for r in successful) / len(successful)
            
            print(f"  {backend:20} Overall: {avg_overall:5.1f} | "
                  f"Quad: {avg_quad:5.1f} | Fidelity: {avg_fidelity:5.1f} | "
                  f"Time: {avg_time:.2f}s")
            
            if avg_overall > best_score:
                best_score = avg_overall
                best_backend = backend
        
        print(f"\n🏆 Best Backend: {best_backend} (avg score: {best_score:.1f})")
        
        # Best individual result
        successful_results = [r for r in results if r.success]
        if successful_results:
            best = max(successful_results, key=lambda r: r.overall_score)
            print(f"\n🎯 Best Single Result:")
            print(f"   Mesh: {best.mesh_name}")
            print(f"   Backend: {best.backend}")
            print(f"   Score: {best.overall_score:.1f} "
                  f"(Q:{best.quad_score:.1f} F:{best.fidelity_score:.1f})")
    
    def compare_to_baseline(self, baseline_date: Optional[str] = None):
        """Compare current results to a baseline."""
        if len(self.results) < 2:
            print("Not enough historical data for comparison")
            return
        
        # Get latest run
        latest = [r for r in self.results if r.timestamp.startswith(datetime.now().strftime("%Y-%m-%d"))]
        
        if not latest:
            latest = self.results[-20:]  # Last 20 results
        
        # Get baseline (first results or specified date)
        if baseline_date:
            baseline = [r for r in self.results if r.timestamp.startswith(baseline_date)]
        else:
            baseline = self.results[:20]
        
        if not baseline:
            print("No baseline data found")
            return
        
        # Compare averages
        latest_avg = sum(r.overall_score for r in latest if r.success) / max(1, len([r for r in latest if r.success]))
        baseline_avg = sum(r.overall_score for r in baseline if r.success) / max(1, len([r for r in baseline if r.success]))
        
        diff = latest_avg - baseline_avg
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        
        print(f"\nBaseline comparison:")
        print(f"  Baseline avg: {baseline_avg:.1f}")
        print(f"  Current avg:  {latest_avg:.1f}")
        print(f"  Change: {direction} {abs(diff):.1f}")


def main():
    """Run benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MeshRetopo Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tests)")
    parser.add_argument("--compare", action="store_true", help="Compare to baseline")
    args = parser.parse_args()
    
    suite = BenchmarkSuite()
    
    if args.compare:
        suite.compare_to_baseline()
    else:
        suite.run_full_suite()


if __name__ == "__main__":
    main()
