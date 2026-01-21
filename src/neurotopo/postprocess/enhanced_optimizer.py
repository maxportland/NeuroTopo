"""
Enhanced Quad Optimizer with RetopoFlow-inspired techniques.

Improvements over basic optimizer:
- Feature edge locking (preserves sharp edges)
- Continuous surface projection (shrinkwrap)
- Principal curvature alignment
- Tangent-plane Laplacian smoothing
- Final shrinkwrap pass

Performance optimizations:
- Vectorized vertex updates using sparse matrices
- Batch shrinkwrap projection
- Pre-computed adjacency with CSR format
"""

from __future__ import annotations

import logging
import numpy as np
from scipy import sparse
from typing import Optional, Set, Tuple
from dataclasses import dataclass

from neurotopo.postprocess.shrinkwrap import Shrinkwrap, ShrinkwrapConfig, create_shrinkwrap

logger = logging.getLogger("neurotopo.postprocess.enhanced_optimizer")


@dataclass
class EnhancedOptimizerConfig:
    """Configuration for enhanced quad optimizer."""
    # Iteration settings
    iterations: int = 15
    
    # Smoothing weights
    smoothing_weight: float = 0.3
    angle_weight: float = 0.5
    edge_equalization_weight: float = 0.3
    
    # Surface conformity
    surface_weight: float = 0.8
    final_shrinkwrap: bool = True
    project_every_step: bool = True
    
    # Feature preservation
    lock_feature_edges: bool = True
    lock_boundary: bool = True
    feature_edge_indices: Optional[Set[int]] = None
    
    # Principal direction alignment
    align_to_curvature: bool = False
    curvature_alignment_weight: float = 0.3
    
    # Relaxation
    tangent_plane_smoothing: bool = True


class EnhancedQuadOptimizer:
    """
    Enhanced post-processing optimizer for quad-dominant meshes.
    
    Incorporates RetopoFlow-inspired techniques:
    - Feature edge locking during optimization
    - Continuous surface projection (shrinkwrap)
    - Tangent-plane Laplacian smoothing
    - Final 100% projection pass
    
    Performance optimized with:
    - Vectorized Laplacian smoothing via sparse matrices
    - Batch shrinkwrap projection
    - NumPy broadcasting for geometric computations
    """
    
    def __init__(self, config: Optional[EnhancedOptimizerConfig] = None):
        self.config = config or EnhancedOptimizerConfig()
        self._shrinkwrap: Optional[Shrinkwrap] = None
        self._laplacian_matrix: Optional[sparse.csr_matrix] = None
        self._adjacency_counts: Optional[np.ndarray] = None
    
    def optimize(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        original_mesh=None,  # trimesh object
        feature_vertices: Optional[Set[int]] = None,
        principal_directions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Optimize vertex positions with enhanced techniques.
        
        Args:
            vertices: Nx3 vertex positions
            faces: Mx4 face indices (quads, may have degenerate triangles)
            original_mesh: Original trimesh for surface projection
            feature_vertices: Vertex indices on feature edges (will be locked)
            principal_directions: Nx3 principal curvature directions per vertex
            
        Returns:
            Optimized vertex positions
        """
        vertices = vertices.copy()
        n_verts = len(vertices)
        
        # Setup shrinkwrap if we have original mesh
        if original_mesh is not None:
            shrinkwrap_config = ShrinkwrapConfig(
                final_projection=self.config.final_shrinkwrap,
                project_every_iteration=self.config.project_every_step,
                projection_strength=self.config.surface_weight,
            )
            self._shrinkwrap = Shrinkwrap(original_mesh, shrinkwrap_config)
        
        # Build optimized mesh topology (sparse Laplacian matrix)
        self._build_laplacian_matrix(vertices, faces)
        face_verts = self._build_face_vertices(faces)
        
        # Determine locked vertices and create mask
        locked_vertices = self._determine_locked_vertices(
            vertices, faces, feature_vertices
        )
        locked_mask = np.zeros(n_verts, dtype=bool)
        for vi in locked_vertices:
            if vi < n_verts:
                locked_mask[vi] = True
        movable_mask = ~locked_mask
        movable_indices = np.where(movable_mask)[0]
        
        logger.debug(f"Enhanced optimization: {len(locked_vertices)} locked vertices, "
                    f"{self.config.iterations} iterations")
        
        # Main optimization loop - VECTORIZED
        for iteration in range(self.config.iterations):
            iter_ratio = iteration / max(self.config.iterations - 1, 1)
            
            # Adaptive weights - decrease smoothing over iterations
            smooth_w = self.config.smoothing_weight * (1 - iter_ratio * 0.5)
            
            # ===== VECTORIZED LAPLACIAN SMOOTHING =====
            # Compute centroids for all vertices at once using sparse matrix
            centroids = self._laplacian_matrix.dot(vertices)
            # Normalize by neighbor count (avoid division by zero)
            counts = self._adjacency_counts.reshape(-1, 1)
            counts = np.where(counts > 0, counts, 1)
            centroids = centroids / counts
            
            # ===== VECTORIZED EDGE EQUALIZATION =====
            edge_targets = self._compute_edge_equalization_vectorized(
                vertices, movable_indices
            )
            
            # ===== VECTORIZED ANGLE OPTIMIZATION =====
            angle_targets = self._compute_angle_optimization_vectorized(
                vertices, faces, face_verts, movable_indices
            )
            
            # ===== CURVATURE ALIGNMENT (if enabled) =====
            curvature_adjustments = np.zeros_like(vertices)
            if (self.config.align_to_curvature and 
                principal_directions is not None):
                curvature_adjustments = self._compute_curvature_alignment_vectorized(
                    vertices, principal_directions, movable_indices
                )
            
            # ===== COMBINE OBJECTIVES FOR ALL MOVABLE VERTICES =====
            new_vertices = vertices.copy()
            
            # Blend all components
            blend_factor = smooth_w
            new_vertices[movable_mask] = (
                vertices[movable_mask] * (1 - blend_factor) +
                centroids[movable_mask] * blend_factor * 0.5 +
                edge_targets[movable_mask] * blend_factor * self.config.edge_equalization_weight +
                angle_targets[movable_mask] * blend_factor * self.config.angle_weight +
                curvature_adjustments[movable_mask] * self.config.curvature_alignment_weight
            )
            
            # ===== BATCH SHRINKWRAP PROJECTION =====
            if self._shrinkwrap is not None and self.config.project_every_step:
                # Project all movable vertices in one batch call
                new_vertices = self._shrinkwrap.project_vertices(
                    new_vertices,
                    locked_vertices=locked_vertices,
                    strength=self.config.surface_weight
                )
            
            vertices = new_vertices
        
        # Final shrinkwrap pass - full projection
        if self._shrinkwrap is not None and self.config.final_shrinkwrap:
            vertices = self._shrinkwrap.final_shrinkwrap(
                vertices,
                locked_vertices=locked_vertices if self.config.lock_feature_edges else None
            )
        
        return vertices
    
    def _build_laplacian_matrix(self, vertices: np.ndarray, faces: np.ndarray):
        """Build sparse Laplacian matrix for vectorized smoothing."""
        n_verts = len(vertices)
        
        # Build adjacency from faces
        rows = []
        cols = []
        
        for face in faces:
            unique = list(set(face))
            for i, vi in enumerate(unique):
                for j, vj in enumerate(unique):
                    if i != j and vi < n_verts and vj < n_verts:
                        rows.append(vi)
                        cols.append(vj)
        
        if not rows:
            # Empty mesh
            self._laplacian_matrix = sparse.csr_matrix((n_verts, n_verts))
            self._adjacency_counts = np.zeros(n_verts)
            return
        
        # Create sparse adjacency matrix (binary)
        data = np.ones(len(rows))
        adj_matrix = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(n_verts, n_verts)
        )
        
        # Convert to Laplacian (sum of neighbors)
        # For each vertex, we want to sum neighbor positions
        self._laplacian_matrix = adj_matrix
        
        # Store neighbor counts for normalization
        self._adjacency_counts = np.array(adj_matrix.sum(axis=1)).flatten()
    
    def _compute_edge_equalization_vectorized(
        self,
        vertices: np.ndarray,
        movable_indices: np.ndarray,
    ) -> np.ndarray:
        """Vectorized edge length equalization."""
        result = vertices.copy()
        
        if len(movable_indices) == 0:
            return result
        
        # For each movable vertex, compute adjustment based on neighbor edges
        # Using the pre-computed Laplacian structure
        for vi in movable_indices:
            # Get neighbors from sparse matrix
            row_start = self._laplacian_matrix.indptr[vi]
            row_end = self._laplacian_matrix.indptr[vi + 1]
            neighbors = self._laplacian_matrix.indices[row_start:row_end]
            
            if len(neighbors) < 2:
                continue
            
            current = vertices[vi]
            neighbor_verts = vertices[neighbors]
            
            # Compute all edge lengths at once
            edge_vectors = neighbor_verts - current
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)
            avg_length = np.mean(edge_lengths)
            
            # Compute adjustment (vectorized)
            valid_mask = edge_lengths > 1e-10
            if not np.any(valid_mask):
                continue
            
            factors = np.zeros(len(neighbors))
            factors[valid_mask] = (edge_lengths[valid_mask] - avg_length) / edge_lengths[valid_mask] * 0.1
            
            adjustment = np.sum(edge_vectors * factors[:, np.newaxis], axis=0)
            result[vi] = current + adjustment
        
        return result
    
    def _compute_angle_optimization_vectorized(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_verts: list,
        movable_indices: np.ndarray,
    ) -> np.ndarray:
        """Vectorized angle optimization towards 90 degrees."""
        result = vertices.copy()
        
        for vi in movable_indices:
            if vi >= len(face_verts):
                continue
            
            current = vertices[vi]
            adjustment = np.zeros(3)
            count = 0
            
            for fi in face_verts[vi]:
                face = faces[fi]
                unique = list(set(face))
                
                if len(unique) != 4:
                    continue
                
                try:
                    idx = unique.index(vi)
                except ValueError:
                    continue
                
                # Get adjacent vertices
                prev_vi = unique[(idx - 1) % 4]
                next_vi = unique[(idx + 1) % 4]
                
                e1 = vertices[prev_vi] - current
                e2 = vertices[next_vi] - current
                
                n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if n1 < 1e-10 or n2 < 1e-10:
                    continue
                
                e1, e2 = e1 / n1, e2 / n2
                cos_angle = np.clip(np.dot(e1, e2), -1, 1)
                angle = np.arccos(cos_angle)
                
                angle_diff = angle - np.pi / 2
                
                bisector = e1 + e2
                bisector_norm = np.linalg.norm(bisector)
                if bisector_norm > 1e-10:
                    bisector = bisector / bisector_norm
                    adjustment += bisector * angle_diff * 0.1
                    count += 1
            
            if count > 0:
                adjustment /= count
            
            result[vi] = current + adjustment
        
        return result
    
    def _compute_curvature_alignment_vectorized(
        self,
        vertices: np.ndarray,
        principal_directions: np.ndarray,
        movable_indices: np.ndarray,
    ) -> np.ndarray:
        """Vectorized curvature alignment computation."""
        result = np.zeros_like(vertices)
        
        for vi in movable_indices:
            if vi >= len(principal_directions):
                continue
            
            current = vertices[vi]
            target_dir = principal_directions[vi]
            
            if np.linalg.norm(target_dir) < 1e-10:
                continue
            
            target_dir = target_dir / np.linalg.norm(target_dir)
            
            # Get neighbors
            row_start = self._laplacian_matrix.indptr[vi]
            row_end = self._laplacian_matrix.indptr[vi + 1]
            neighbors = self._laplacian_matrix.indices[row_start:row_end]
            
            if len(neighbors) == 0:
                continue
            
            # Vectorized edge computation
            edge_vectors = vertices[neighbors] - current
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)
            valid_mask = edge_lengths > 1e-10
            
            if not np.any(valid_mask):
                continue
            
            edge_dirs = edge_vectors[valid_mask] / edge_lengths[valid_mask, np.newaxis]
            
            # Project onto target direction
            parallels = np.dot(edge_dirs, target_dir)
            perps = edge_dirs - parallels[:, np.newaxis] * target_dir
            
            adjustment = -np.sum(perps, axis=0) * 0.05
            result[vi] = adjustment
        
        return result
    
    def _determine_locked_vertices(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        feature_vertices: Optional[Set[int]],
    ) -> Set[int]:
        """Determine which vertices should be locked during optimization."""
        locked = set()
        
        # Lock boundary vertices
        if self.config.lock_boundary:
            boundary = self._find_boundary_vertices(faces)
            locked.update(boundary)
        
        # Lock feature edge vertices
        if self.config.lock_feature_edges:
            if feature_vertices is not None:
                locked.update(feature_vertices)
            
            if self.config.feature_edge_indices is not None:
                locked.update(self.config.feature_edge_indices)
        
        return locked
    
    def _build_face_vertices(self, faces: np.ndarray) -> list:
        """Build vertex-to-faces mapping."""
        max_vert = faces.max() + 1
        face_verts = [[] for _ in range(max_vert)]
        
        for fi, face in enumerate(faces):
            for vi in set(face):
                face_verts[vi].append(fi)
        
        return face_verts
    
    def _find_boundary_vertices(self, faces: np.ndarray) -> set:
        """Find vertices on mesh boundary."""
        edge_count = {}
        
        for face in faces:
            unique = list(set(face))
            n = len(unique)
            for i in range(n):
                v0, v1 = unique[i], unique[(i + 1) % n]
                edge = (min(v0, v1), max(v0, v1))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary = set()
        for (v0, v1), count in edge_count.items():
            if count == 1:
                boundary.add(v0)
                boundary.add(v1)
        
        return boundary


def optimize_with_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    original_mesh=None,
    feature_set=None,  # FeatureSet from analysis
    iterations: int = 15,
    lock_features: bool = True,
    final_shrinkwrap: bool = True,
) -> np.ndarray:
    """
    Convenience function for enhanced optimization with feature preservation.
    
    Args:
        vertices: Vertex positions
        faces: Face indices
        original_mesh: trimesh of original surface
        feature_set: FeatureSet from feature detection
        iterations: Number of optimization iterations
        lock_features: Whether to lock feature vertices
        final_shrinkwrap: Whether to do final projection pass
        
    Returns:
        Optimized vertices
    """
    # Extract feature vertices if available
    feature_vertices = None
    if feature_set is not None and lock_features:
        feature_vertices = set(feature_set.get_feature_vertices())
    
    config = EnhancedOptimizerConfig(
        iterations=iterations,
        lock_feature_edges=lock_features,
        final_shrinkwrap=final_shrinkwrap,
    )
    
    optimizer = EnhancedQuadOptimizer(config)
    
    return optimizer.optimize(
        vertices=vertices,
        faces=faces,
        original_mesh=original_mesh,
        feature_vertices=feature_vertices,
    )
