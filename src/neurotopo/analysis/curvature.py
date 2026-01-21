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

from neurotopo.core.mesh import Mesh
from neurotopo.core.fields import ScalarField, FieldLocation

logger = logging.getLogger("neurotopo.analysis.curvature")


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
    
    def compute_principal_directions(self, sample_ratio: float = 0.1) -> np.ndarray:
        """
        Compute principal curvature directions for each vertex.
        
        Returns Nx3 array of principal direction vectors (max curvature direction).
        These directions are useful for aligning quad edges with the natural
        flow of the surface - a key technique from RetopoFlow and Instant Meshes.
        
        OPTIMIZED: Uses sampling + interpolation for large meshes.
        - For small meshes (<1000 verts): compute all vertices
        - For large meshes: sample vertices, compute directions, interpolate
        
        Args:
            sample_ratio: Fraction of vertices to sample for large meshes (default 0.1)
        
        Uses local quadric fitting approach.
        """
        start_time = time.time()
        
        faces = np.array(self.mesh.faces)
        verts = self.mesh.vertices
        n_verts = len(verts)
        
        # Compute vertex normals
        if self.mesh.normals is None:
            self.mesh.compute_normals()
        normals = self.mesh.normals
        
        # Build adjacency using vectorized operations
        adjacency = self._build_adjacency_fast(faces, n_verts)
        
        # Determine if we should use sampling
        use_sampling = n_verts > 1000
        
        if use_sampling:
            # Sample vertices for direction computation
            n_samples = max(100, int(n_verts * sample_ratio))
            sampled_indices = self._select_sample_vertices(n_verts, n_samples, adjacency)
            directions = self._compute_directions_for_vertices(
                sampled_indices, verts, normals, adjacency
            )
            # Interpolate to all vertices
            all_directions = self._interpolate_directions(
                sampled_indices, directions, verts, adjacency
            )
        else:
            # Compute for all vertices directly
            all_indices = np.arange(n_verts)
            all_directions = self._compute_directions_for_vertices(
                all_indices, verts, normals, adjacency
            )
        
        elapsed = time.time() - start_time
        logger.debug(f"Principal directions: {elapsed:.3f}s (sampling={'yes' if use_sampling else 'no'})")
        
        return all_directions
    
    def _build_adjacency_fast(self, faces: np.ndarray, n_verts: int) -> list:
        """Build adjacency list using vectorized operations."""
        adjacency = [set() for _ in range(n_verts)]
        
        # Vectorized edge extraction
        n_face_verts = faces.shape[1]
        for i in range(n_face_verts):
            j = (i + 1) % n_face_verts
            v0s = faces[:, i]
            v1s = faces[:, j]
            
            for v0, v1 in zip(v0s, v1s):
                if v0 < n_verts and v1 < n_verts:
                    adjacency[v0].add(v1)
                    adjacency[v1].add(v0)
        
        return adjacency
    
    def _select_sample_vertices(
        self, 
        n_verts: int, 
        n_samples: int, 
        adjacency: list
    ) -> np.ndarray:
        """
        Select well-distributed sample vertices using approximate farthest point sampling.
        
        Uses a greedy approach with local distance tracking for efficiency.
        """
        # For small sample counts, just use evenly spaced indices
        if n_samples > n_verts // 2:
            # Sample too large, just use all vertices
            return np.arange(n_verts)
        
        # Use simple strided sampling + random perturbation for speed
        # This is much faster than true farthest point sampling
        # and provides reasonable distribution
        stride = n_verts // n_samples
        base_indices = np.arange(0, n_verts, stride)[:n_samples]
        
        # Add some randomness to avoid regular patterns
        rng = np.random.default_rng(42)
        perturbation = rng.integers(-stride // 4, stride // 4 + 1, size=len(base_indices))
        sampled = np.clip(base_indices + perturbation, 0, n_verts - 1)
        
        # Ensure unique
        sampled = np.unique(sampled)
        
        # If we don't have enough, fill with random vertices
        if len(sampled) < n_samples:
            remaining = np.setdiff1d(np.arange(n_verts), sampled)
            n_need = min(n_samples - len(sampled), len(remaining))
            extra = rng.choice(remaining, size=n_need, replace=False)
            sampled = np.concatenate([sampled, extra])
        
        return sampled[:n_samples]
    
    def _compute_directions_for_vertices(
        self,
        indices: np.ndarray,
        verts: np.ndarray,
        normals: np.ndarray,
        adjacency: list,
    ) -> np.ndarray:
        """Compute principal directions for specified vertex indices."""
        directions = np.zeros((len(indices), 3))
        
        for i, vi in enumerate(indices):
            neighbors = list(adjacency[vi])
            if len(neighbors) < 3:
                continue
            
            normal = normals[vi]
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue
            normal = normal / norm_len
            
            # Build local coordinate frame
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(normal, up)) > 0.9:
                up = np.array([1.0, 0.0, 0.0])
            
            t1 = np.cross(normal, up)
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            
            # Project neighbors to local tangent plane (vectorized)
            center = verts[vi]
            neighbor_verts = verts[neighbors]
            offsets = neighbor_verts - center
            
            # Local coordinates
            u = np.dot(offsets, t1)
            v = np.dot(offsets, t2)
            h = np.dot(offsets, normal)
            
            # Fit quadric
            A = np.column_stack([
                u**2, v**2, u*v, u, v, np.ones(len(u))
            ])
            
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
                a, b, c = coeffs[:3]
                
                # Shape operator (2x2)
                S = np.array([[2*a, c], [c, 2*b]])
                eigenvalues, eigenvectors = np.linalg.eigh(S)
                
                max_idx = np.argmax(np.abs(eigenvalues))
                local_dir = eigenvectors[:, max_idx]
                
                directions[i] = local_dir[0] * t1 + local_dir[1] * t2
            except Exception:
                continue
        
        return directions
    
    def _interpolate_directions(
        self,
        sampled_indices: np.ndarray,
        sampled_directions: np.ndarray,
        verts: np.ndarray,
        adjacency: list,
    ) -> np.ndarray:
        """Interpolate directions from sampled vertices to all vertices."""
        n_verts = len(verts)
        all_directions = np.zeros((n_verts, 3))
        
        # Copy sampled directions
        for i, vi in enumerate(sampled_indices):
            all_directions[vi] = sampled_directions[i]
        
        # Build KD-tree of sampled vertex positions for fast nearest-neighbor
        sampled_positions = verts[sampled_indices]
        
        # For non-sampled vertices, interpolate from nearest sampled neighbors
        sampled_set = set(sampled_indices)
        
        for vi in range(n_verts):
            if vi in sampled_set:
                continue
            
            # Find nearest sampled vertices by graph distance
            # Use BFS to find closest sampled neighbors
            nearest_sampled = []
            weights = []
            
            queue = [(vi, 0)]
            visited = {vi}
            
            while queue and len(nearest_sampled) < 4:
                current, dist = queue.pop(0)
                
                if current in sampled_set and current != vi:
                    nearest_sampled.append(current)
                    weights.append(1.0 / (dist + 1))
                
                if dist < 5:  # Limit search depth
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
            
            if nearest_sampled:
                # Weighted average of directions
                total_weight = sum(weights)
                if total_weight > 0:
                    for j, si in enumerate(nearest_sampled):
                        # Find index in sampled_indices
                        try:
                            idx = np.where(sampled_indices == si)[0][0]
                            all_directions[vi] += sampled_directions[idx] * weights[j]
                        except IndexError:
                            continue
                    
                    # Normalize
                    dir_norm = np.linalg.norm(all_directions[vi])
                    if dir_norm > 1e-10:
                        all_directions[vi] /= dir_norm
        
        return all_directions
    
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


def compute_principal_directions(mesh: Mesh) -> np.ndarray:
    """
    Convenience function to compute principal curvature directions.
    
    Returns Nx3 array of direction vectors for quad edge alignment.
    """
    analyzer = CurvatureAnalyzer(mesh)
    return analyzer.compute_principal_directions()
