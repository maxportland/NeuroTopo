"""Mesh analysis modules."""

from meshretopo.analysis.curvature import (
    CurvatureAnalyzer,
    CurvatureType,
    compute_curvature,
)
from meshretopo.analysis.features import (
    FeatureDetector,
    FeatureSet,
    FeatureEdge,
    FeaturePoint,
    detect_features,
)
from meshretopo.analysis.neural import (
    NeuralAnalyzer,
    NeuralPrediction,
    create_default_analyzer,
)
from meshretopo.analysis.semantic import (
    SemanticAnalyzer,
    SemanticSegmentation,
    SemanticSegment,
    SemanticRegion,
    REGION_TOPOLOGY_RULES,
    analyze_mesh_semantics,
)

__all__ = [
    "CurvatureAnalyzer",
    "CurvatureType", 
    "compute_curvature",
    "FeatureDetector",
    "FeatureSet",
    "FeatureEdge",
    "FeaturePoint",
    "detect_features",
    "NeuralAnalyzer",
    "NeuralPrediction",
    "create_default_analyzer",
    # Semantic analysis
    "SemanticAnalyzer",
    "SemanticSegmentation",
    "SemanticSegment",
    "SemanticRegion",
    "REGION_TOPOLOGY_RULES",
    "analyze_mesh_semantics",
]
