"""Evaluation and metrics modules."""

from neurotopo.evaluation.metrics import (
    MeshEvaluator,
    RetopologyScore,
    QuadQualityMetrics,
    GeometricFidelityMetrics,
    TopologyMetrics,
    evaluate_retopology,
)
from neurotopo.evaluation.visual import (
    VisualEvaluator,
    VisualQualityMetrics,
)
from neurotopo.evaluation.visualize import (
    MeshVisualizer,
    visualize_comparison,
)
from neurotopo.evaluation.ai_quality import (
    AIQualityAssessor,
    AIQualityReport,
    TopologyIssue,
    IssueSeverity,
    IssueCategory,
    assess_mesh_quality,
)
from neurotopo.evaluation.manifold_test import (
    test_manifold,
    test_manifold_blender,
    test_manifold_python,
    ManifoldTestResult,
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
    # Manifold testing
    "test_manifold",
    "test_manifold_blender",
    "test_manifold_python",
    "ManifoldTestResult",
]
