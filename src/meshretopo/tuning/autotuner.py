"""
Auto-tuning system for optimal retopology parameters.

Uses grid search and Bayesian-like optimization to find
the best configuration for a given mesh.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.pipeline import RetopoPipeline
from meshretopo.evaluation import RetopologyScore


@dataclass
class TuningResult:
    """Result from auto-tuning."""
    best_config: dict
    best_score: float
    all_results: list[tuple[dict, float]]
    total_time: float
    iterations: int


class AutoTuner:
    """
    Automatic parameter tuning for retopology.
    
    Searches the parameter space to find optimal settings
    for a given input mesh.
    """
    
    def __init__(
        self,
        backends: list[str] = None,
        neural_weight_range: tuple[float, float] = (0.3, 0.9),
        feature_weight_range: tuple[float, float] = (0.1, 0.7),
        reduction_range: tuple[float, float] = (0.1, 0.4),
        max_iterations: int = 20,
        early_stop_score: float = 75.0,
        time_limit: float = 120.0,  # seconds
    ):
        self.backends = backends or ["trimesh", "hybrid"]
        self.neural_weight_range = neural_weight_range
        self.feature_weight_range = feature_weight_range
        self.reduction_range = reduction_range
        self.max_iterations = max_iterations
        self.early_stop_score = early_stop_score
        self.time_limit = time_limit
    
    def tune(
        self,
        mesh: Mesh,
        objective: str = "overall",  # overall, quad, fidelity
        verbose: bool = True,
    ) -> TuningResult:
        """
        Find optimal parameters for the given mesh.
        
        Args:
            mesh: Input mesh to optimize for
            objective: Which score to optimize
            verbose: Print progress
            
        Returns:
            TuningResult with best configuration
        """
        start_time = time.time()
        results = []
        best_score = 0
        best_config = None
        
        # Phase 1: Grid search over key parameters
        if verbose:
            print("Phase 1: Grid search...")
        
        grid_configs = self._generate_grid_configs()
        
        for i, config in enumerate(grid_configs):
            if time.time() - start_time > self.time_limit:
                if verbose:
                    print(f"  Time limit reached after {i} iterations")
                break
            
            score = self._evaluate_config(mesh, config, objective)
            results.append((config, score))
            
            if score > best_score:
                best_score = score
                best_config = config
                if verbose:
                    print(f"  New best: {score:.1f} - {config['backend']} "
                          f"(nw={config['neural_weight']:.2f}, r={config['reduction']:.2f})")
            
            if score >= self.early_stop_score:
                if verbose:
                    print(f"  Early stop: reached {score:.1f}")
                break
        
        # Phase 2: Local refinement around best config
        if best_config and time.time() - start_time < self.time_limit * 0.8:
            if verbose:
                print("\nPhase 2: Local refinement...")
            
            refined_configs = self._generate_refined_configs(best_config)
            
            for config in refined_configs:
                if time.time() - start_time > self.time_limit:
                    break
                
                score = self._evaluate_config(mesh, config, objective)
                results.append((config, score))
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    if verbose:
                        print(f"  Refined: {score:.1f}")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTuning complete in {total_time:.1f}s")
            print(f"Best score: {best_score:.1f}")
            print(f"Best config: {best_config}")
        
        return TuningResult(
            best_config=best_config,
            best_score=best_score,
            all_results=results,
            total_time=total_time,
            iterations=len(results),
        )
    
    def _generate_grid_configs(self) -> list[dict]:
        """Generate grid of configurations to try."""
        configs = []
        
        neural_weights = np.linspace(
            self.neural_weight_range[0],
            self.neural_weight_range[1],
            3
        )
        reductions = np.linspace(
            self.reduction_range[0],
            self.reduction_range[1],
            3
        )
        
        for backend in self.backends:
            for nw in neural_weights:
                for reduction in reductions:
                    fw = 1.0 - nw
                    if fw >= self.feature_weight_range[0] and fw <= self.feature_weight_range[1]:
                        configs.append({
                            "backend": backend,
                            "neural_weight": float(nw),
                            "feature_weight": float(fw),
                            "reduction": float(reduction),
                        })
        
        return configs
    
    def _generate_refined_configs(self, base_config: dict) -> list[dict]:
        """Generate configurations around a base config."""
        configs = []
        
        # Small variations
        deltas = [-0.1, -0.05, 0.05, 0.1]
        
        for delta in deltas:
            # Vary neural weight
            nw = base_config["neural_weight"] + delta
            if self.neural_weight_range[0] <= nw <= self.neural_weight_range[1]:
                configs.append({
                    **base_config,
                    "neural_weight": nw,
                    "feature_weight": 1.0 - nw,
                })
            
            # Vary reduction
            r = base_config["reduction"] + delta
            if self.reduction_range[0] <= r <= self.reduction_range[1]:
                configs.append({
                    **base_config,
                    "reduction": r,
                })
        
        return configs
    
    def _evaluate_config(
        self,
        mesh: Mesh,
        config: dict,
        objective: str,
    ) -> float:
        """Evaluate a configuration and return score."""
        try:
            target_faces = int(mesh.num_faces * config["reduction"])
            
            pipeline = RetopoPipeline(
                backend=config["backend"],
                target_faces=target_faces,
                neural_weight=config["neural_weight"],
                feature_weight=config["feature_weight"],
            )
            
            _, score = pipeline.process(mesh, evaluate=True)
            
            if score is None:
                return 0.0
            
            if objective == "quad":
                return score.quad_score
            elif objective == "fidelity":
                return score.fidelity_score
            else:
                return score.overall_score
                
        except Exception as e:
            return 0.0


class AdaptiveRetopology:
    """
    High-level adaptive retopology that automatically
    selects the best approach for each mesh.
    """
    
    def __init__(self, time_budget: float = 30.0):
        self.time_budget = time_budget
        self.tuner = AutoTuner(
            max_iterations=15,
            time_limit=time_budget * 0.8,
        )
    
    def process(
        self,
        mesh: Mesh,
        target_faces: Optional[int] = None,
        quality_priority: str = "balanced",  # balanced, fidelity, quads
    ) -> tuple[Mesh, RetopologyScore, dict]:
        """
        Process mesh with automatic parameter selection.
        
        Args:
            mesh: Input mesh
            target_faces: Target face count (optional)
            quality_priority: What to optimize for
            
        Returns:
            (output_mesh, score, config_used)
        """
        # Determine objective
        objective = "overall"
        if quality_priority == "fidelity":
            objective = "fidelity"
        elif quality_priority == "quads":
            objective = "quad"
        
        # Override reduction range if target specified
        if target_faces is not None:
            reduction = target_faces / mesh.num_faces
            self.tuner.reduction_range = (
                max(0.05, reduction - 0.1),
                min(0.9, reduction + 0.1)
            )
        
        # Run tuning
        result = self.tuner.tune(mesh, objective=objective, verbose=False)
        
        # Run final pass with best config
        pipeline = RetopoPipeline(
            backend=result.best_config["backend"],
            target_faces=int(mesh.num_faces * result.best_config["reduction"]),
            neural_weight=result.best_config["neural_weight"],
            feature_weight=result.best_config["feature_weight"],
        )
        
        output, score = pipeline.process(mesh, evaluate=True)
        
        return output, score, result.best_config


def auto_retopo(
    mesh: Mesh,
    target_faces: Optional[int] = None,
    time_budget: float = 30.0,
    verbose: bool = True,
) -> tuple[Mesh, RetopologyScore]:
    """
    Convenience function for automatic retopology.
    
    Args:
        mesh: Input mesh
        target_faces: Target face count
        time_budget: Maximum time in seconds
        verbose: Print progress
        
    Returns:
        (output_mesh, score)
    """
    adaptive = AdaptiveRetopology(time_budget=time_budget)
    
    if verbose:
        print(f"Auto-tuning retopology for {mesh.num_faces} faces...")
    
    output, score, config = adaptive.process(mesh, target_faces=target_faces)
    
    if verbose:
        print(f"Best config: {config}")
        print(f"Final score: {score.overall_score:.1f}")
    
    return output, score
