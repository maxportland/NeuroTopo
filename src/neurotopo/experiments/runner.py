"""
Experiment runner with logging and result tracking.

Orchestrates the full retopology pipeline and tracks results
for comparison and iteration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.analysis import detect_features, create_default_analyzer
from neurotopo.analysis.neural import CurvatureBasedSizingPredictor, PrincipalDirectionPredictor, HybridNeuralAnalyzer
from neurotopo.guidance import GuidanceComposer
from neurotopo.remesh import get_remesher, RemeshResult
from neurotopo.evaluation import MeshEvaluator, RetopologyScore
from neurotopo.experiments.config import ExperimentConfig


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    config: ExperimentConfig
    input_mesh_name: str
    output_mesh: Optional[Mesh]
    score: Optional[RetopologyScore]
    remesh_result: Optional[RemeshResult]
    
    # Timing
    total_time: float = 0.0
    analysis_time: float = 0.0
    remesh_time: float = 0.0
    evaluation_time: float = 0.0
    
    # Status
    success: bool = False
    error: Optional[str] = None
    
    # Metadata
    timestamp: str = ""
    iteration: int = 0
    
    def to_dict(self) -> dict:
        return {
            "config_name": self.config.name,
            "input_mesh": self.input_mesh_name,
            "success": self.success,
            "error": self.error,
            "total_time": self.total_time,
            "analysis_time": self.analysis_time,
            "remesh_time": self.remesh_time,
            "evaluation_time": self.evaluation_time,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "score": self.score.to_dict() if self.score else None,
            "output_faces": self.output_mesh.num_faces if self.output_mesh else None,
        }


@dataclass
class ExperimentLog:
    """Log of all experiment runs."""
    results: list[ExperimentResult] = field(default_factory=list)
    
    def add(self, result: ExperimentResult) -> None:
        self.results.append(result)
    
    def best_result(self) -> Optional[ExperimentResult]:
        """Get result with highest overall score."""
        valid = [r for r in self.results if r.success and r.score]
        if not valid:
            return None
        return max(valid, key=lambda r: r.score.overall_score)
    
    def summary_df(self):
        """Return results as pandas DataFrame."""
        import pandas as pd
        
        records = []
        for r in self.results:
            record = {
                "config": r.config.name,
                "mesh": r.input_mesh_name,
                "success": r.success,
                "overall_score": r.score.overall_score if r.score else None,
                "quad_score": r.score.quad_score if r.score else None,
                "fidelity_score": r.score.fidelity_score if r.score else None,
                "topology_score": r.score.topology_score if r.score else None,
                "output_faces": r.output_mesh.num_faces if r.output_mesh else None,
                "total_time": r.total_time,
                "iteration": r.iteration,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save log to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [r.to_dict() for r in self.results]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class ExperimentRunner:
    """
    Run retopology experiments with tracking.
    
    Supports single runs, parameter sweeps, and iterative improvement.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.log = ExperimentLog()
        
        # Set random seed for reproducibility
        if config.seed is not None:
            np.random.seed(config.seed)
    
    def run(
        self,
        input_mesh: Union[str, Path, Mesh],
        iteration: int = 0
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            input_mesh: Path to mesh file or Mesh object
            iteration: Iteration number (for auto-iterate mode)
            
        Returns:
            ExperimentResult with all metrics
        """
        start_time = time.time()
        
        # Load mesh if needed
        if isinstance(input_mesh, (str, Path)):
            mesh_path = Path(input_mesh)
            mesh = Mesh.from_file(mesh_path)
            mesh_name = mesh_path.stem
        else:
            mesh = input_mesh
            mesh_name = mesh.name
        
        result = ExperimentResult(
            config=self.config,
            input_mesh_name=mesh_name,
            output_mesh=None,
            score=None,
            remesh_result=None,
            timestamp=datetime.now().isoformat(),
            iteration=iteration
        )
        
        try:
            # Step 1: Analysis
            analysis_start = time.time()
            
            # Create neural analyzer based on config
            sizing_predictor = CurvatureBasedSizingPredictor(
                curvature_sensitivity=self.config.analysis.curvature_sensitivity,
                smoothing_iterations=self.config.analysis.smoothing_iterations
            )
            direction_predictor = PrincipalDirectionPredictor()
            analyzer = HybridNeuralAnalyzer([
                (sizing_predictor, self.config.analysis.neural_weight),
                (direction_predictor, 1.0),
            ])
            
            prediction = analyzer.predict(mesh)
            
            # Feature detection
            features = detect_features(
                mesh,
                angle_threshold=self.config.analysis.feature_angle_threshold
            )
            
            # Compose guidance
            composer = GuidanceComposer(
                neural_weight=self.config.analysis.neural_weight,
                feature_weight=self.config.analysis.feature_weight
            )
            
            # Determine target faces
            target_faces = self.config.remesh.target_faces
            if target_faces is None and self.config.remesh.target_reduction is not None:
                target_faces = int(mesh.num_faces * self.config.remesh.target_reduction)
            
            guidance = composer.compose(
                mesh, prediction, features,
                target_faces=target_faces
            )
            
            result.analysis_time = time.time() - analysis_start
            
            # Step 2: Remeshing
            remesh_start = time.time()
            
            remesher = get_remesher(
                self.config.remesh.backend,
                iterations=self.config.remesh.iterations,
                adaptive=self.config.remesh.adaptive,
                preserve_boundary=self.config.remesh.preserve_boundary
            )
            
            remesh_result = remesher.remesh(mesh, guidance)
            result.remesh_result = remesh_result
            result.output_mesh = remesh_result.mesh
            
            result.remesh_time = time.time() - remesh_start
            
            # Step 3: Evaluation
            eval_start = time.time()
            
            evaluator = MeshEvaluator(sample_count=self.config.evaluation.sample_count)
            score = evaluator.evaluate(remesh_result.mesh, mesh)
            
            # Update weights from config
            score.weights = {
                "quad": self.config.evaluation.quad_weight,
                "fidelity": self.config.evaluation.fidelity_weight,
                "topology": self.config.evaluation.topology_weight
            }
            score.compute_scores(mesh.diagonal)
            
            result.score = score
            result.evaluation_time = time.time() - eval_start
            
            result.success = remesh_result.success
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        result.total_time = time.time() - start_time
        
        # Log result
        self.log.add(result)
        
        return result
    
    def run_iterative(
        self,
        input_mesh: Union[str, Path, Mesh],
        target_score: Optional[float] = None,
        max_iterations: Optional[int] = None
    ) -> ExperimentResult:
        """
        Run iterative improvement until target score is reached.
        
        Args:
            input_mesh: Input mesh
            target_score: Stop when this score is reached
            max_iterations: Maximum iterations
            
        Returns:
            Best result achieved
        """
        target = target_score or self.config.target_score
        max_iter = max_iterations or self.config.max_iterations
        
        best_result = None
        current_mesh = input_mesh
        
        for i in range(max_iter):
            result = self.run(current_mesh, iteration=i)
            
            if best_result is None or (result.score and result.score.overall_score > best_result.score.overall_score):
                best_result = result
            
            # Check if target reached
            if result.score and result.score.overall_score >= target:
                print(f"Target score {target} reached at iteration {i}")
                break
            
            # Use output as input for next iteration
            if result.output_mesh is not None:
                current_mesh = result.output_mesh
        
        return best_result
    
    def run_all_inputs(self) -> list[ExperimentResult]:
        """Run experiment on all configured input meshes."""
        results = []
        
        for mesh_path in self.config.input_meshes:
            if self.config.auto_iterate:
                result = self.run_iterative(mesh_path)
            else:
                result = self.run(mesh_path)
            results.append(result)
        
        return results
    
    def save_results(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Save all results and outputs."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save log
        self.log.save(output_dir / "experiment_log.json")
        
        # Save config
        self.config.save(output_dir / "config.yaml")
        
        # Save output meshes
        for result in self.log.results:
            if result.output_mesh is not None:
                mesh_path = output_dir / f"{result.input_mesh_name}_retopo_{result.iteration}.obj"
                result.output_mesh.to_file(mesh_path)
        
        # Save summary
        if len(self.log.results) > 0:
            summary_path = output_dir / "summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Experiment: {self.config.name}\n")
                f.write(f"Description: {self.config.description}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total runs: {len(self.log.results)}\n\n")
                
                best = self.log.best_result()
                if best and best.score:
                    f.write("Best Result:\n")
                    f.write(best.score.summary())


def run_experiment(config: ExperimentConfig, input_mesh: Union[str, Path, Mesh]) -> ExperimentResult:
    """Convenience function to run a single experiment."""
    runner = ExperimentRunner(config)
    return runner.run(input_mesh)
