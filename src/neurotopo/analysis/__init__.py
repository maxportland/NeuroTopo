"""Mesh analysis modules."""

from neurotopo.analysis.curvature import (
    CurvatureAnalyzer,
    CurvatureType,
    compute_curvature,
    compute_principal_directions,
)
from neurotopo.analysis.features import (
    FeatureDetector,
    FeatureSet,
    FeatureEdge,
    FeaturePoint,
    detect_features,
)
from neurotopo.analysis.neural import (
    NeuralAnalyzer,
    NeuralPrediction,
    create_default_analyzer,
)
from neurotopo.analysis.semantic import (
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
    "compute_principal_directions",
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
