"""Evaluation and metrics modules."""

from meshretopo.evaluation.metrics import (
    MeshEvaluator,
    RetopologyScore,
    QuadQualityMetrics,
    GeometricFidelityMetrics,
    TopologyMetrics,
    evaluate_retopology,
)
from meshretopo.evaluation.visual import (
    VisualEvaluator,
    VisualQualityMetrics,
)
from meshretopo.evaluation.visualize import (
    MeshVisualizer,
    visualize_comparison,
)
from meshretopo.evaluation.ai_quality import (
    AIQualityAssessor,
    AIQualityReport,
    TopologyIssue,
    IssueSeverity,
    IssueCategory,
    assess_mesh_quality,
)

__all__ = [
    "MeshEvaluator",
    "RetopologyScore",
    "QuadQualityMetrics",
    "GeometricFidelityMetrics",
    "TopologyMetrics",
    "VisualQualityMetrics",
    "VisualEvaluator",
    "evaluate_retopology",
    "MeshVisualizer",
    "visualize_comparison",
    # AI quality assessment
    "AIQualityAssessor",
    "AIQualityReport",
    "TopologyIssue",
    "IssueSeverity",
    "IssueCategory",
    "assess_mesh_quality",
]
