"""
Guidance field generation.

Combines neural predictions with classical analysis and user constraints
to generate the final guidance fields for remeshing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.core.fields import ScalarField, DirectionField, FieldLocation
from meshretopo.analysis.neural import NeuralPrediction
from meshretopo.analysis.features import FeatureSet


@dataclass
class GuidanceFields:
    """Combined guidance information for remeshing."""
    size_field: ScalarField  # Target quad size at each vertex
    direction_field: Optional[DirectionField]  # Preferred edge directions
    importance_field: ScalarField  # Feature importance weights
    
    # Constraints
    target_face_count: Optional[int] = None
    symmetry_plane: Optional[np.ndarray] = None  # (point, normal) for symmetry
    
    def scale_to_target(self, mesh: Mesh) -> GuidanceFields:
        """Adjust size field to achieve target face count."""
        if self.target_face_count is None:
            return self
        
        # Estimate current face count from size field
        # Average size -> average quad area -> estimated face count
        avg_size = self.size_field.mean
        estimated_area = avg_size ** 2
        
        # Compute mesh surface area
        surface_area = self._compute_surface_area(mesh)
        estimated_faces = surface_area / estimated_area
        
        # Scale factor to hit target
        scale = np.sqrt(estimated_faces / self.target_face_count)
        
        scaled_sizes = self.size_field.values * scale
        scaled_field = ScalarField(
            scaled_sizes, 
            self.size_field.location,
            "scaled_size"
        )
        
        return GuidanceFields(
            size_field=scaled_field,
            direction_field=self.direction_field,
            importance_field=self.importance_field,
            target_face_count=self.target_face_count,
            symmetry_plane=self.symmetry_plane
        )
    
    @staticmethod
    def _compute_surface_area(mesh: Mesh) -> float:
        """Compute total surface area of mesh."""
        if not mesh.is_triangular:
            mesh = mesh.triangulate()
        
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        return float(np.sum(areas))


class GuidanceComposer:
    """
    Compose guidance fields from multiple sources.
    
    Blends neural predictions, feature detection, and user constraints
    into unified guidance for the remesher.
    """
    
    def __init__(
        self,
        neural_weight: float = 0.7,
        feature_weight: float = 0.3,
        min_size_ratio: float = 0.5,  # Min size as ratio of max
    ):
        self.neural_weight = neural_weight
        self.feature_weight = feature_weight
        self.min_size_ratio = min_size_ratio
    
    def compose(
        self,
        mesh: Mesh,
        neural_prediction: NeuralPrediction,
        features: FeatureSet,
        target_faces: Optional[int] = None,
        user_size_field: Optional[ScalarField] = None,
    ) -> GuidanceFields:
        """
        Compose final guidance fields.
        
        Args:
            mesh: Input mesh
            neural_prediction: Predictions from neural analyzer
            features: Detected features
            target_faces: Target face count (optional)
            user_size_field: User-provided size hints (optional)
        """
        # Compose size field
        size_field = self._compose_size_field(
            mesh, neural_prediction, features, user_size_field
        )
        
        # Get direction field (use neural if available)
        direction_field = neural_prediction.direction_field
        
        # Compose importance field
        importance_field = self._compose_importance(
            mesh, neural_prediction, features
        )
        
        guidance = GuidanceFields(
            size_field=size_field,
            direction_field=direction_field,
            importance_field=importance_field,
            target_face_count=target_faces
        )
        
        # Scale to target face count if specified
        if target_faces is not None:
            guidance = guidance.scale_to_target(mesh)
        
        return guidance
    
    def _compose_size_field(
        self,
        mesh: Mesh,
        prediction: NeuralPrediction,
        features: FeatureSet,
        user_field: Optional[ScalarField]
    ) -> ScalarField:
        """Blend size fields from different sources."""
        components = []
        weights = []
        
        # Neural sizing
        if prediction.sizing_field is not None:
            components.append(prediction.sizing_field.values)
            weights.append(self.neural_weight * prediction.confidence)
        
        # Feature-based sizing (smaller near features)
        if features.vertex_importance is not None:
            # Invert importance: high importance -> small size
            feature_sizing = 1.0 - 0.5 * features.vertex_importance.values
            components.append(feature_sizing)
            weights.append(self.feature_weight)
        
        # User field
        if user_field is not None:
            components.append(user_field.values)
            weights.append(1.0)  # User input has full weight
        
        if not components:
            # Default: uniform sizing
            uniform = np.ones(mesh.num_vertices) * 0.05  # 5% of bbox diagonal
            return ScalarField(uniform * mesh.diagonal, FieldLocation.VERTEX, "size")
        
        # Weighted blend
        total_weight = sum(weights)
        blended = sum(c * w for c, w in zip(components, weights)) / total_weight
        
        # Enforce min/max ratio
        max_size = blended.max()
        min_size = max_size * self.min_size_ratio
        blended = np.clip(blended, min_size, max_size)
        
        return ScalarField(blended, FieldLocation.VERTEX, "composed_size")
    
    def _compose_importance(
        self,
        mesh: Mesh,
        prediction: NeuralPrediction,
        features: FeatureSet
    ) -> ScalarField:
        """Compose importance field for feature preservation."""
        importance = features.vertex_importance.values.copy()
        
        # Boost importance from neural prediction if available
        if prediction.importance_field is not None:
            importance = np.maximum(importance, prediction.importance_field.values)
        
        return ScalarField(importance, FieldLocation.VERTEX, "importance")
