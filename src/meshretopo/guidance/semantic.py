"""
Semantic-aware guidance generation for topology-informed remeshing.

Converts semantic segmentation into guidance fields that the remesher
can use to apply topology rules like:
- Concentric loops around eyes/mouth
- Higher density at joints
- Pole placement in flat areas
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.core.fields import ScalarField, DirectionField, FieldLocation
from meshretopo.analysis.semantic import (
    SemanticSegmentation,
    SemanticRegion,
    REGION_TOPOLOGY_RULES,
)

logger = logging.getLogger("meshretopo.guidance.semantic")


@dataclass
class SemanticGuidanceFields:
    """Guidance fields derived from semantic analysis."""
    
    # Core guidance
    density_field: ScalarField  # Per-vertex density multiplier
    pole_penalty_field: ScalarField  # Per-vertex pole placement penalty
    deformation_priority: ScalarField  # Per-vertex deformation importance
    loop_requirement: ScalarField  # Per-vertex concentric loop need
    
    # Feature centers (for radial topology)
    feature_centers: list[tuple[np.ndarray, SemanticRegion]]  # (position, type)
    
    # Segmentation reference
    segmentation: SemanticSegmentation
    
    def get_combined_sizing_weight(self, base_field: ScalarField) -> ScalarField:
        """
        Combine semantic density with a base sizing field.
        
        Higher density -> smaller quads (inverse of size field)
        """
        # Density is a multiplier, so invert for sizing
        # density=1.5 means quads should be 1/1.5 = 0.67x the base size
        density_factor = 1.0 / np.clip(self.density_field.values, 0.5, 2.0)
        
        combined = base_field.values * density_factor
        return ScalarField(combined, FieldLocation.VERTEX, "semantic_weighted_size")


class SemanticGuidanceGenerator:
    """
    Generate remeshing guidance from semantic segmentation.
    
    This translates semantic understanding of the mesh into concrete
    guidance fields that influence the remeshing algorithm.
    """
    
    def __init__(
        self,
        density_smoothing: int = 3,  # Smoothing iterations for density
        blend_radius: float = 0.1,  # Blend radius for transitions (relative to mesh)
    ):
        self.density_smoothing = density_smoothing
        self.blend_radius = blend_radius
    
    def generate(
        self,
        mesh: Mesh,
        segmentation: SemanticSegmentation,
    ) -> SemanticGuidanceFields:
        """
        Generate guidance fields from semantic segmentation.
        
        Args:
            mesh: The input mesh
            segmentation: Semantic segmentation results
            
        Returns:
            SemanticGuidanceFields with all guidance information
        """
        logger.info("Generating semantic guidance fields")
        
        # Generate per-vertex fields from segmentation
        density_field = self._generate_density_field(mesh, segmentation)
        pole_penalty = self._generate_pole_penalty_field(mesh, segmentation)
        deformation = self._generate_deformation_field(mesh, segmentation)
        loop_req = self._generate_loop_requirement_field(mesh, segmentation)
        
        # Smooth the fields for better gradients
        density_field = density_field.smooth(mesh, self.density_smoothing)
        pole_penalty = pole_penalty.smooth(mesh, 2)
        loop_req = loop_req.smooth(mesh, 2)
        
        # Extract feature centers for radial topology
        feature_centers = self._extract_feature_centers(segmentation)
        
        return SemanticGuidanceFields(
            density_field=density_field,
            pole_penalty_field=pole_penalty,
            deformation_priority=deformation,
            loop_requirement=loop_req,
            feature_centers=feature_centers,
            segmentation=segmentation,
        )
    
    def _generate_density_field(
        self,
        mesh: Mesh,
        segmentation: SemanticSegmentation
    ) -> ScalarField:
        """Generate density field from segment rules."""
        densities = np.ones(mesh.num_vertices)
        
        for segment in segmentation.segments:
            rules = segment.rules
            weight = rules.relative_density * segment.confidence
            
            # Apply density to segment vertices
            current = densities[segment.vertex_indices]
            densities[segment.vertex_indices] = np.maximum(current, weight)
        
        return ScalarField(densities, FieldLocation.VERTEX, "semantic_density")
    
    def _generate_pole_penalty_field(
        self,
        mesh: Mesh,
        segmentation: SemanticSegmentation
    ) -> ScalarField:
        """Generate pole penalty field - high values = bad for poles."""
        penalties = np.zeros(mesh.num_vertices)
        
        for segment in segmentation.segments:
            rules = segment.rules
            
            if not rules.allow_poles:
                # Strong penalty - no poles here
                penalty = 1.0 * segment.confidence
            elif rules.ideal_pole_placement:
                # Negative penalty (bonus) - good spot for poles
                penalty = -0.3 * segment.confidence
            else:
                # Neutral
                penalty = 0.3 * segment.confidence
            
            current = penalties[segment.vertex_indices]
            # Use max for penalties, allow negatives for ideal spots
            if penalty > 0:
                penalties[segment.vertex_indices] = np.maximum(current, penalty)
            else:
                penalties[segment.vertex_indices] = np.minimum(current, penalty)
        
        # Normalize to 0-1 range
        penalties = (penalties - penalties.min()) / (penalties.max() - penalties.min() + 1e-10)
        
        return ScalarField(penalties, FieldLocation.VERTEX, "pole_penalty")
    
    def _generate_deformation_field(
        self,
        mesh: Mesh,
        segmentation: SemanticSegmentation
    ) -> ScalarField:
        """Generate deformation priority field."""
        priorities = np.full(mesh.num_vertices, 0.5)  # Default neutral
        
        for segment in segmentation.segments:
            priority = segment.rules.deformation_priority * segment.confidence
            current = priorities[segment.vertex_indices]
            priorities[segment.vertex_indices] = np.maximum(current, priority)
        
        return ScalarField(priorities, FieldLocation.VERTEX, "deformation_priority")
    
    def _generate_loop_requirement_field(
        self,
        mesh: Mesh,
        segmentation: SemanticSegmentation
    ) -> ScalarField:
        """Generate field indicating need for concentric loops."""
        requirements = np.zeros(mesh.num_vertices)
        
        for segment in segmentation.segments:
            rules = segment.rules
            
            if rules.needs_concentric_loops:
                # Higher value = more loops needed
                value = (rules.ideal_loop_count / 5.0) * segment.confidence
                current = requirements[segment.vertex_indices]
                requirements[segment.vertex_indices] = np.maximum(current, value)
        
        return ScalarField(requirements, FieldLocation.VERTEX, "loop_requirement")
    
    def _extract_feature_centers(
        self,
        segmentation: SemanticSegmentation
    ) -> list[tuple[np.ndarray, SemanticRegion]]:
        """Extract centers of features needing radial topology."""
        centers = []
        
        radial_regions = {
            SemanticRegion.EYE,
            SemanticRegion.MOUTH,
            SemanticRegion.NOSE,
            SemanticRegion.SHOULDER,
            SemanticRegion.HIP,
        }
        
        for segment in segmentation.segments:
            if segment.region_type in radial_regions:
                centers.append((segment.centroid.copy(), segment.region_type))
        
        return centers


def generate_semantic_guidance(
    mesh: Mesh,
    segmentation: SemanticSegmentation,
) -> SemanticGuidanceFields:
    """
    Convenience function to generate semantic guidance.
    
    Args:
        mesh: Input mesh
        segmentation: Semantic segmentation results
        
    Returns:
        SemanticGuidanceFields for remeshing
    """
    generator = SemanticGuidanceGenerator()
    return generator.generate(mesh, segmentation)
