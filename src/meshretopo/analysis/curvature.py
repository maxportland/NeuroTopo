"""
Curvature analysis for meshes.

Provides various curvature measures used to guide adaptive sizing
and feature detection.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional
import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.core.fields import ScalarField, FieldLocation

logger = logging.getLogger("meshretopo.analysis.curvature")


class CurvatureType(Enum):
    """Types of curvature measures."""
    GAUSSIAN = "gaussian"
    MEAN = "mean"
    PRINCIPAL_MAX = "principal_max"
    PRINCIPAL_MIN = "principal_min"
    SHAPE_INDEX = "shape_index"


class CurvatureAnalyzer:
    """
    Compute various curvature measures on meshes.
    
    Used to determine where the mesh needs more or fewer quads.
    Optimized for large meshes using vectorized operations.
    """
    
    # Use fast approximation for meshes larger than this
    FAST_THRESHOLD = 50000
    
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self._ensure_triangular()
        self._cache = {}
        self._use_fast = mesh.num_faces > self.FAST_THRESHOLD
    
    def _ensure_triangular(self):
        """Ensure we're working with a triangular mesh."""
        if not self.mesh.is_triangular:
            self.mesh = self.mesh.triangulate()
    
    def compute(self, curvature_type: CurvatureType) -> ScalarField:
        """Compute the specified curvature type."""
        if curvature_type in self._cache:
            return self._cache[curvature_type]
        
        start_time = time.time()
        method = "fast" if self._use_fast else "standard"
        
        if curvature_type == CurvatureType.GAUSSIAN:
            result = self._compute_gaussian_fast() if self._use_fast else self._compute_gaussian()
        elif curvature_type == CurvatureType.MEAN:
            result = self._compute_mean_fast() if self._use_fast else self._compute_mean()
        elif curvature_type == CurvatureType.PRINCIPAL_MAX:
            result = self._compute_principal()[0]
        elif curvature_type == CurvatureType.PRINCIPAL_MIN:
            result = self._compute_principal()[1]
        elif curvature_type == CurvatureType.SHAPE_INDEX:
            result = self._compute_shape_index()
        else:
            raise ValueError(f"Unknown curvature type: {curvature_type}")
        
        elapsed = time.time() - start_time
        logger.debug(f"Curvature {curvature_type.value} ({method}): {elapsed:.3f}s")
        
        self._cache[curvature_type] = result
        return result
    
    def _compute_gaussian_fast(self) -> ScalarField:
        """Fast vectorized Gaussian curvature computation."""
        faces = np.array(self.mesh.faces)
        verts = self.mesh.vertices
        
        # Get all triangle vertices at once
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        # Edge vectors
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2
        
        # Compute angles using dot products
        def compute_angles(ea, eb):
            dot = np.sum(-ea * eb, axis=1)
            norm_a = np.linalg.norm(ea, axis=1)
            norm_b = np.linalg.norm(eb, axis=1)
            cos_angle = np.clip(dot / (norm_a * norm_b + 1e-10), -1, 1)
            return np.arccos(cos_angle)
        
        angles0 = compute_angles(-e2, e0)
        angles1 = compute_angles(-e0, e1)
        angles2 = compute_angles(-e1, e2)
        
        # Triangle areas
        cross = np.cross(e0, -e2)
        tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Accumulate using np.add.at for speed
        n_verts = len(verts)
        angle_sum = np.zeros(n_verts)
        area = np.zeros(n_verts)
        
        np.add.at(angle_sum, faces[:, 0], angles0)
        np.add.at(angle_sum, faces[:, 1], angles1)
        np.add.at(angle_sum, faces[:, 2], angles2)
        
        area_per_vert = tri_areas / 3
        np.add.at(area, faces[:, 0], area_per_vert)
        np.add.at(area, faces[:, 1], area_per_vert)
        np.add.at(area, faces[:, 2], area_per_vert)
        
        # Gaussian curvature
        area = np.maximum(area, 1e-10)
        gaussian = (2 * np.pi - angle_sum) / area
        
        return ScalarField(gaussian, FieldLocation.VERTEX, "gaussian_curvature")
    
    def _compute_mean_fast(self) -> ScalarField:
        """Fast vectorized mean curvature computation."""
        faces = np.array(self.mesh.faces)
        verts = self.mesh.vertices
        n_verts = len(verts)
        
        # Get all triangle vertices
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        # Compute cotangent weights for each edge
        def cotangent(v1, v2):
            dot = np.sum(v1 * v2, axis=1)
            cross_norm = np.linalg.norm(np.cross(v1, v2), axis=1)
            return dot / (cross_norm + 1e-10)
        
        cot0 = cotangent(v1 - v0, v2 - v0)
        cot1 = cotangent(v0 - v1, v2 - v1)
        cot2 = cotangent(v0 - v2, v1 - v2)
        
        # Laplacian contributions
        laplacian = np.zeros((n_verts, 3))
        weights_sum = np.zeros(n_verts)
        
        # Contribution to v0
        contrib0 = cot2[:, np.newaxis] * (v1 - v0) + cot1[:, np.newaxis] * (v2 - v0)
        # Contribution to v1
        contrib1 = cot2[:, np.newaxis] * (v0 - v1) + cot0[:, np.newaxis] * (v2 - v1)
        # Contribution to v2
        contrib2 = cot1[:, np.newaxis] * (v0 - v2) + cot0[:, np.newaxis] * (v1 - v2)
        
        np.add.at(laplacian, faces[:, 0], contrib0)
        np.add.at(laplacian, faces[:, 1], contrib1)
        np.add.at(laplacian, faces[:, 2], contrib2)
        
        np.add.at(weights_sum, faces[:, 0], cot1 + cot2)
        np.add.at(weights_sum, faces[:, 1], cot0 + cot2)
        np.add.at(weights_sum, faces[:, 2], cot0 + cot1)
        
        # Mean curvature
        weights_sum = np.maximum(weights_sum, 1e-10)
        mean_curvature = 0.5 * np.linalg.norm(laplacian, axis=1) / weights_sum
        
        return ScalarField(mean_curvature, FieldLocation.VERTEX, "mean_curvature")
    
    def _compute_gaussian(self) -> ScalarField:
        """
        Compute Gaussian curvature using angle defect method.
        
        K = (2π - sum of angles) / A_mixed
        """
        n_verts = self.mesh.num_vertices
        angle_sum = np.zeros(n_verts)
        area = np.zeros(n_verts)
        
        for face in self.mesh.faces:
            v0, v1, v2 = self.mesh.vertices[face]
            
            # Compute edge vectors
            e0 = v1 - v0
            e1 = v2 - v1
            e2 = v0 - v2
            
            # Compute angles at each vertex
            angles = [
                self._angle_between(-e2, e0),
                self._angle_between(-e0, e1),
                self._angle_between(-e1, e2)
            ]
            
            # Triangle area
            tri_area = 0.5 * np.linalg.norm(np.cross(e0, -e2))
            
            for i, (vi, angle) in enumerate(zip(face, angles)):
                angle_sum[vi] += angle
                area[vi] += tri_area / 3  # Distribute area equally
        
        # Gaussian curvature via angle defect
        area = np.where(area > 1e-10, area, 1e-10)
        gaussian = (2 * np.pi - angle_sum) / area
        
        return ScalarField(gaussian, FieldLocation.VERTEX, "gaussian_curvature")
    
    def _compute_mean(self) -> ScalarField:
        """
        Compute mean curvature using cotangent Laplacian.
        
        H = 0.5 * |Δx| (magnitude of Laplacian)
        """
        n_verts = self.mesh.num_vertices
        laplacian = np.zeros((n_verts, 3))
        weights_sum = np.zeros(n_verts)
        
        for face in self.mesh.faces:
            v0, v1, v2 = face
            p0, p1, p2 = self.mesh.vertices[face]
            
            # Cotangent weights
            cot0 = self._cotangent(p1 - p0, p2 - p0)
            cot1 = self._cotangent(p0 - p1, p2 - p1)
            cot2 = self._cotangent(p0 - p2, p1 - p2)
            
            # Add contributions
            laplacian[v0] += cot2 * (p1 - p0) + cot1 * (p2 - p0)
            laplacian[v1] += cot2 * (p0 - p1) + cot0 * (p2 - p1)
            laplacian[v2] += cot1 * (p0 - p2) + cot0 * (p1 - p2)
            
            weights_sum[v0] += cot1 + cot2
            weights_sum[v1] += cot0 + cot2
            weights_sum[v2] += cot0 + cot1
        
        # Mean curvature is half the magnitude of the normalized Laplacian
        weights_sum = np.where(weights_sum > 1e-10, weights_sum, 1e-10)
        mean_curvature = 0.5 * np.linalg.norm(laplacian, axis=1) / weights_sum
        
        return ScalarField(mean_curvature, FieldLocation.VERTEX, "mean_curvature")
    
    def _compute_principal(self) -> tuple[ScalarField, ScalarField]:
        """Compute principal curvatures from Gaussian and mean curvature."""
        H = self.compute(CurvatureType.MEAN).values
        K = self.compute(CurvatureType.GAUSSIAN).values
        
        # k1, k2 = H ± sqrt(H² - K)
        discriminant = np.maximum(H**2 - K, 0)  # Clamp numerical errors
        sqrt_disc = np.sqrt(discriminant)
        
        k_max = H + sqrt_disc
        k_min = H - sqrt_disc
        
        return (
            ScalarField(k_max, FieldLocation.VERTEX, "principal_max"),
            ScalarField(k_min, FieldLocation.VERTEX, "principal_min")
        )
    
    def _compute_shape_index(self) -> ScalarField:
        """
        Compute shape index (normalized curvature descriptor).
        
        SI = (2/π) * arctan((k1 + k2) / (k1 - k2))
        
        Range: [-1, 1] where -1 = cup, 0 = saddle, 1 = cap
        """
        k_max, k_min = self._compute_principal()
        k1, k2 = k_max.values, k_min.values
        
        denom = k1 - k2
        denom = np.where(np.abs(denom) > 1e-10, denom, 1e-10)
        
        shape_index = (2 / np.pi) * np.arctan((k1 + k2) / denom)
        
        return ScalarField(shape_index, FieldLocation.VERTEX, "shape_index")
    
    def compute_all(self) -> dict[CurvatureType, ScalarField]:
        """Compute all curvature measures."""
        return {ct: self.compute(ct) for ct in CurvatureType}
    
    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        return float(np.arccos(cos_angle))
    
    @staticmethod
    def _cotangent(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cotangent of angle between two vectors."""
        cos_angle = np.dot(v1, v2)
        sin_angle = np.linalg.norm(np.cross(v1, v2))
        if abs(sin_angle) < 1e-10:
            return 0.0
        return cos_angle / sin_angle


def compute_curvature(mesh: Mesh, curvature_type: CurvatureType = CurvatureType.MEAN) -> ScalarField:
    """Convenience function to compute curvature."""
    analyzer = CurvatureAnalyzer(mesh)
    return analyzer.compute(curvature_type)
