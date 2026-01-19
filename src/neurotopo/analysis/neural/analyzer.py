"""
Neural network modules for mesh analysis.

These modules predict guidance information that would traditionally
require artist intuition: edge flow directions, adaptive sizing, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.core.fields import ScalarField, VectorField, DirectionField, FieldLocation


@dataclass
class NeuralPrediction:
    """Container for neural network predictions."""
    sizing_field: Optional[ScalarField] = None
    direction_field: Optional[DirectionField] = None
    importance_field: Optional[ScalarField] = None
    confidence: float = 0.0
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NeuralAnalyzer(ABC):
    """Abstract base class for neural mesh analyzers."""
    
    @abstractmethod
    def predict(self, mesh: Mesh) -> NeuralPrediction:
        """Generate predictions for the input mesh."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return analyzer name."""
        pass


class CurvatureBasedSizingPredictor(NeuralAnalyzer):
    """
    Predict adaptive sizing field based on curvature.
    
    This is a "neural-style" predictor that uses classical curvature
    analysis but could be replaced with a learned model later.
    
    The interface remains stable for easy swapping.
    """
    
    def __init__(
        self,
        min_size: float = 0.01,
        max_size: float = 0.1,
        curvature_sensitivity: float = 1.0,
        smoothing_iterations: int = 3
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.curvature_sensitivity = curvature_sensitivity
        self.smoothing_iterations = smoothing_iterations
    
    def name(self) -> str:
        return "curvature_sizing"
    
    def predict(self, mesh: Mesh) -> NeuralPrediction:
        """Predict sizing field from curvature."""
        from neurotopo.analysis.curvature import CurvatureAnalyzer, CurvatureType
        
        # Compute curvature
        analyzer = CurvatureAnalyzer(mesh)
        mean_curv = analyzer.compute(CurvatureType.MEAN)
        
        # Convert curvature to sizing: high curvature -> small quads
        # Use inverse relationship with sensitivity control
        curv_abs = np.abs(mean_curv.values)
        
        # Normalize curvature to [0, 1] range with outlier handling
        curv_95 = np.percentile(curv_abs, 95)
        curv_normalized = np.clip(curv_abs / (curv_95 + 1e-10), 0, 1)
        
        # Apply sensitivity
        curv_weighted = curv_normalized ** self.curvature_sensitivity
        
        # Invert: high curvature -> low size value
        size_values = 1.0 - curv_weighted
        
        # Scale to target range
        size_values = size_values * (self.max_size - self.min_size) + self.min_size
        
        sizing_field = ScalarField(size_values, FieldLocation.VERTEX, "neural_sizing")
        
        # Smooth the field
        sizing_field = sizing_field.smooth(mesh, self.smoothing_iterations)
        
        return NeuralPrediction(
            sizing_field=sizing_field,
            confidence=0.7,  # Classical method, moderate confidence
            metadata={"method": "curvature_based"}
        )


class PrincipalDirectionPredictor(NeuralAnalyzer):
    """
    Predict edge flow directions from principal curvature directions.
    
    Principal curvature directions provide natural edge flow for
    organic surfaces (muscles, cloth folds, etc.).
    """
    
    def __init__(self, smoothing_iterations: int = 5):
        self.smoothing_iterations = smoothing_iterations
    
    def name(self) -> str:
        return "principal_direction"
    
    def predict(self, mesh: Mesh) -> NeuralPrediction:
        """Predict direction field from principal curvatures."""
        # Compute principal directions using covariance analysis
        directions = self._compute_principal_directions(mesh)
        
        direction_field = DirectionField(
            directions=directions,
            location=FieldLocation.VERTEX,
            name="neural_direction"
        )
        
        return NeuralPrediction(
            direction_field=direction_field,
            confidence=0.6,
            metadata={"method": "principal_curvature"}
        )
    
    def _compute_principal_directions(self, mesh: Mesh) -> np.ndarray:
        """Compute principal curvature directions via local covariance (vectorized)."""
        if mesh.normals is None:
            mesh.compute_normals()
        
        n_verts = mesh.num_vertices
        faces = np.array(mesh.faces)
        
        # For large meshes, use vectorized batch processing
        if n_verts > 10000:
            return self._compute_principal_directions_fast(mesh)
        
        directions = np.zeros((n_verts, 3))
        
        # Build vertex adjacency
        adjacency = [set() for _ in range(n_verts)]
        for face in mesh.faces:
            for i, vi in enumerate(face):
                for j, vj in enumerate(face):
                    if i != j:
                        adjacency[vi].add(vj)
        
        for vi in range(n_verts):
            neighbors = list(adjacency[vi])
            if len(neighbors) < 3:
                # Not enough neighbors, use arbitrary tangent
                directions[vi] = self._arbitrary_tangent(mesh.normals[vi])
                continue
            
            # Get neighbor positions relative to vertex
            center = mesh.vertices[vi]
            rel_positions = mesh.vertices[neighbors] - center
            
            # Project onto tangent plane
            normal = mesh.normals[vi]
            proj_positions = rel_positions - np.outer(
                np.dot(rel_positions, normal), normal
            )
            
            # Compute covariance in tangent plane
            cov = np.dot(proj_positions.T, proj_positions)
            
            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Principal direction is eigenvector with largest eigenvalue
            # Project to ensure it's in tangent plane
            principal = eigenvectors[:, -1]
            principal = principal - np.dot(principal, normal) * normal
            
            norm = np.linalg.norm(principal)
            if norm > 1e-10:
                directions[vi] = principal / norm
            else:
                directions[vi] = self._arbitrary_tangent(normal)
        
        return directions
    
    def _compute_principal_directions_fast(self, mesh: Mesh) -> np.ndarray:
        """Fast principal direction computation using sampling and interpolation."""
        n_verts = mesh.num_vertices
        normals = mesh.normals
        verts = mesh.vertices
        faces = np.array(mesh.faces)
        
        # For very large meshes, sample a subset and interpolate
        sample_rate = max(1, n_verts // 20000)  # Aim for ~20k samples
        
        if sample_rate > 1:
            # Sample vertices uniformly
            sample_indices = np.arange(0, n_verts, sample_rate)
            n_samples = len(sample_indices)
        else:
            sample_indices = np.arange(n_verts)
            n_samples = n_verts
        
        # Build sparse adjacency for samples
        from scipy import sparse
        
        rows = []
        cols = []
        for i in range(len(faces[0])):
            for j in range(len(faces[0])):
                if i != j:
                    rows.extend(faces[:, i])
                    cols.extend(faces[:, j])
        
        adj_matrix = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), 
            shape=(n_verts, n_verts)
        )
        
        # Compute directions only for sampled vertices
        sample_directions = np.zeros((n_samples, 3))
        
        for idx, vi in enumerate(sample_indices):
            neighbors = adj_matrix[vi].indices
            if len(neighbors) < 3:
                sample_directions[idx] = self._arbitrary_tangent(normals[vi])
                continue
            
            center = verts[vi]
            rel_positions = verts[neighbors] - center
            normal = normals[vi]
            dots = rel_positions @ normal
            proj_positions = rel_positions - np.outer(dots, normal)
            
            cov = proj_positions.T @ proj_positions
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                principal = eigenvectors[:, -1]
                principal = principal - np.dot(principal, normal) * normal
                
                norm = np.linalg.norm(principal)
                if norm > 1e-10:
                    sample_directions[idx] = principal / norm
                else:
                    sample_directions[idx] = self._arbitrary_tangent(normal)
            except:
                sample_directions[idx] = self._arbitrary_tangent(normal)
        
        # If we sampled, interpolate to all vertices
        if sample_rate > 1:
            directions = np.zeros((n_verts, 3))
            
            # Use nearest-neighbor from samples
            from scipy.spatial import cKDTree
            tree = cKDTree(verts[sample_indices])
            
            # Query in batches
            batch_size = 10000
            for start in range(0, n_verts, batch_size):
                end = min(start + batch_size, n_verts)
                _, nearest_idx = tree.query(verts[start:end], k=1)
                directions[start:end] = sample_directions[nearest_idx]
            
            # Project to tangent plane and normalize
            dots = np.sum(directions * normals, axis=1)
            directions = directions - dots[:, np.newaxis] * normals
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            directions = directions / norms
            
            return directions
        else:
            return sample_directions
        
        return directions
    
    @staticmethod
    def _arbitrary_tangent(normal: np.ndarray) -> np.ndarray:
        """Generate an arbitrary tangent vector."""
        if abs(normal[0]) < 0.9:
            tangent = np.cross(normal, [1, 0, 0])
        else:
            tangent = np.cross(normal, [0, 1, 0])
        return tangent / np.linalg.norm(tangent)


class HybridNeuralAnalyzer(NeuralAnalyzer):
    """
    Combines multiple analyzers with weighted blending.
    
    This allows mixing classical and neural methods, gradually
    increasing neural influence as models improve.
    """
    
    def __init__(self, analyzers: list[tuple[NeuralAnalyzer, float]]):
        """
        Args:
            analyzers: List of (analyzer, weight) tuples
        """
        self.analyzers = analyzers
    
    def name(self) -> str:
        names = [a.name() for a, _ in self.analyzers]
        return f"hybrid({'+'.join(names)})"
    
    def predict(self, mesh: Mesh) -> NeuralPrediction:
        """Blend predictions from all analyzers."""
        predictions = [(a.predict(mesh), w) for a, w in self.analyzers]
        
        # Blend sizing fields
        sizing_fields = [
            (p.sizing_field, w) for p, w in predictions 
            if p.sizing_field is not None
        ]
        blended_sizing = self._blend_scalar_fields(sizing_fields)
        
        # For direction field, use highest confidence
        direction_field = None
        best_conf = 0
        for p, w in predictions:
            if p.direction_field is not None and p.confidence * w > best_conf:
                direction_field = p.direction_field
                best_conf = p.confidence * w
        
        return NeuralPrediction(
            sizing_field=blended_sizing,
            direction_field=direction_field,
            confidence=sum(p.confidence * w for p, w in predictions) / sum(w for _, w in predictions)
        )
    
    def _blend_scalar_fields(
        self, 
        fields: list[tuple[ScalarField, float]]
    ) -> Optional[ScalarField]:
        """Blend multiple scalar fields with weights."""
        if not fields:
            return None
        
        total_weight = sum(w for _, w in fields)
        blended = sum(f.values * w for f, w in fields) / total_weight
        
        return ScalarField(blended, fields[0][0].location, "blended_sizing")


# Factory function for easy creation
def create_default_analyzer() -> NeuralAnalyzer:
    """Create the default neural analyzer configuration."""
    return HybridNeuralAnalyzer([
        (CurvatureBasedSizingPredictor(), 1.0),
        (PrincipalDirectionPredictor(), 1.0),
    ])
