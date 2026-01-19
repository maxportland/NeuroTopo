"""
Experiment configuration system.

Supports YAML-based configuration with inheritance and overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import yaml


@dataclass
class AnalysisConfig:
    """Configuration for the analysis pipeline."""
    neural_weight: float = 0.7
    feature_weight: float = 0.3
    curvature_sensitivity: float = 1.0
    smoothing_iterations: int = 3
    feature_angle_threshold: float = 30.0  # degrees
    
    def to_dict(self) -> dict:
        return {
            "neural_weight": self.neural_weight,
            "feature_weight": self.feature_weight,
            "curvature_sensitivity": self.curvature_sensitivity,
            "smoothing_iterations": self.smoothing_iterations,
            "feature_angle_threshold": self.feature_angle_threshold,
        }


@dataclass
class RemeshConfig:
    """Configuration for the remeshing backend."""
    backend: str = "trimesh"  # trimesh, pymeshlab, pymeshlab_quad, guided_quad
    target_faces: Optional[int] = None
    target_reduction: Optional[float] = None  # e.g., 0.1 = 10% of original
    iterations: int = 3
    adaptive: bool = True
    preserve_boundary: bool = True
    optimization_iterations: int = 10
    
    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "target_faces": self.target_faces,
            "target_reduction": self.target_reduction,
            "iterations": self.iterations,
            "adaptive": self.adaptive,
            "preserve_boundary": self.preserve_boundary,
            "optimization_iterations": self.optimization_iterations,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    sample_count: int = 10000
    quad_weight: float = 0.4
    fidelity_weight: float = 0.4
    topology_weight: float = 0.2
    
    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "quad_weight": self.quad_weight,
            "fidelity_weight": self.fidelity_weight,
            "topology_weight": self.topology_weight,
        }


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "experiment"
    description: str = ""
    
    # Input/output
    input_meshes: list[str] = field(default_factory=list)
    output_dir: str = "outputs"
    
    # Pipeline configs
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    remesh: RemeshConfig = field(default_factory=RemeshConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Iteration settings
    auto_iterate: bool = False
    max_iterations: int = 5
    target_score: float = 80.0
    
    # Reproducibility
    seed: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_meshes": self.input_meshes,
            "output_dir": self.output_dir,
            "analysis": self.analysis.to_dict(),
            "remesh": self.remesh.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "auto_iterate": self.auto_iterate,
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "seed": self.seed,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        """Create config from dictionary."""
        analysis = AnalysisConfig(**data.get("analysis", {}))
        remesh = RemeshConfig(**data.get("remesh", {}))
        evaluation = EvaluationConfig(**data.get("evaluation", {}))
        
        return cls(
            name=data.get("name", "experiment"),
            description=data.get("description", ""),
            input_meshes=data.get("input_meshes", []),
            output_dir=data.get("output_dir", "outputs"),
            analysis=analysis,
            remesh=remesh,
            evaluation=evaluation,
            auto_iterate=data.get("auto_iterate", False),
            max_iterations=data.get("max_iterations", 5),
            target_score=data.get("target_score", 80.0),
            seed=data.get("seed"),
        )
    
    def with_overrides(self, **kwargs) -> ExperimentConfig:
        """Create new config with overrides."""
        data = self.to_dict()
        
        for key, value in kwargs.items():
            if "." in key:
                # Nested key like "remesh.backend"
                parts = key.split(".")
                d = data
                for part in parts[:-1]:
                    d = d[part]
                d[parts[-1]] = value
            else:
                data[key] = value
        
        return ExperimentConfig.from_dict(data)


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig()


def create_sweep_configs(
    base_config: ExperimentConfig,
    sweep_params: dict[str, list]
) -> list[ExperimentConfig]:
    """
    Create multiple configs for parameter sweep.
    
    Args:
        base_config: Base configuration
        sweep_params: Dict of param_name -> list of values to sweep
        
    Returns:
        List of configs, one for each combination
    """
    import itertools
    
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    configs = []
    for values in itertools.product(*param_values):
        overrides = dict(zip(param_names, values))
        config = base_config.with_overrides(**overrides)
        
        # Update name to reflect sweep parameters
        suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
        config.name = f"{base_config.name}_{suffix}"
        
        configs.append(config)
    
    return configs
