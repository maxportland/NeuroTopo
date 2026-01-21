"""Guidance field generation."""

from neurotopo.guidance.composer import GuidanceComposer, GuidanceFields
from neurotopo.guidance.semantic import (
    SemanticGuidanceFields,
    SemanticGuidanceGenerator,
    generate_semantic_guidance,
)
from neurotopo.guidance.contours import (
    TubularRegion,
    ContourLoop,
    ContourSeeding,
    TubularRegionDetector,
    ContourLoopGenerator,
    ContourGuidanceGenerator,
    detect_tubular_regions,
    generate_contour_seeding,
)

__all__ = [
    # Core guidance
    "GuidanceComposer",
    "GuidanceFields",
    # Semantic guidance
    "SemanticGuidanceFields",
    "SemanticGuidanceGenerator",
    "generate_semantic_guidance",
    # Contour-based guidance (for tubes/limbs)
    "TubularRegion",
    "ContourLoop", 
    "ContourSeeding",
    "TubularRegionDetector",
    "ContourLoopGenerator",
    "ContourGuidanceGenerator",
    "detect_tubular_regions",
    "generate_contour_seeding",
]
