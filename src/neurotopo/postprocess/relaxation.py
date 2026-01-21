"""
Iterative Relaxation with Surface Projection.

Implements RetopoFlow-inspired Relax Brush behavior for automated retopology:
- Laplacian-Beltrami smoothing in tangent space
- Continuous surface projection after every vertex move
- Feature-aware relaxation (preserves sharp edges)
- Adaptive relaxation strength based on local geometry

This ensures the retopologized mesh stays "on surface" at all times,
a key characteristic of RetopoFlow's workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Set, List
import numpy as np

logger = logging.getLogger("neurotopo.postprocess.relaxation")


@dataclass
class RelaxationConfig:
    """Configuration for iterative relaxation."""
    # Iteration control
    iterations: int = 10
    
    # Smoothing strength
    strength: float = 0.5  # 0-1, how much to move towards target
    falloff: float = 0.8  # How strength decreases per iteration
    
    # Surface projection
    project_every_step: bool = True
    projection_strength: float = 1.0  # 1.0 = full projection
    
    # Feature preservation
    preserve_features: bool = True
    feature_blend: float = 0.2  # How much to move feature vertices (0=locked)
    
    # Boundary handling
    preserve_boundary: bool = True
    boundary_strength: float = 0.3  # Reduced strength for boundary
    
    # Adaptive relaxation
    adaptive_strength: bool = True
    curvature_factor: float = 0.5  # Reduce strength in high-curvature areas


class ProjectedRelaxation:
    """
    Laplacian relaxation with continuous surface projection.
    
    Every vertex movement is immediately followed by projection
    back to the original surface, ensuring the mesh "hugs" the
    original geometry like RetopoFlow's behavior.
    """
    
    def __init__(
        self,
        original_mesh=None,  # trimesh object for projection
        config: Optional[RelaxationConfig] = None,
    ):
        self.original_mesh = original_mesh
        self.config = config or RelaxationConfig()
    
    def relax(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        feature_vertices: Optional[Set[int]] = None,
        vertex_normals: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Perform iterative relaxation with projection.
        
        Args:
            vertices: Nx3 vertex positions
            faces: Face indices
            feature_vertices: Vertices on feature edges (reduced movement)
            vertex_normals: Nx3 vertex normals (for tangent-plane smoothing)
            curvature: Per-vertex curvature values (for adaptive strength)
            
        Returns:
            Relaxed vertex positions
        """
        vertices = vertices.copy()
        n_verts = len(vertices)
        
        # Build topology
        adjacency = self._build_adjacency(vertices, faces)
        boundary = self._find_boundary_vertices(faces)
        
        # Precompute vertex categories
        locked = set()  # Completely locked
        reduced = set()  # Reduced movement (features, boundary)
        
        if self.config.preserve_boundary:
            reduced.update(boundary)
        
        if self.config.preserve_features and feature_vertices:
            reduced.update(feature_vertices)
        
        # Get surface normals if not provided
        if vertex_normals is None and self.original_mesh is not None:
            vertex_normals = self._compute_surface_normals(vertices)
        
        # Compute adaptive strength factors
        strength_factors = np.ones(n_verts)
        if self.config.adaptive_strength and curvature is not None:
            # Reduce strength in high-curvature areas
            max_curv = np.percentile(np.abs(curvature), 95)
            if max_curv > 1e-10:
                normalized_curv = np.clip(np.abs(curvature) / max_curv, 0, 1)
                strength_factors = 1.0 - normalized_curv * self.config.curvature_factor
        
        # Iterative relaxation
        for iteration in range(self.config.iterations):
            # Decay strength over iterations
            iter_strength = self.config.strength * (self.config.falloff ** iteration)
            
            new_vertices = vertices.copy()
            
            for vi in range(n_verts):
                if vi in locked or not adjacency[vi]:
                    continue
                
                # Determine effective strength for this vertex
                effective_strength = iter_strength * strength_factors[vi]
                
                if vi in reduced:
                    if vi in boundary:
                        effective_strength *= self.config.boundary_strength
                    elif feature_vertices and vi in feature_vertices:
                        effective_strength *= self.config.feature_blend
                
                # Compute Laplacian target
                neighbors = adjacency[vi]
                
                if vertex_normals is not None and np.linalg.norm(vertex_normals[vi]) > 1e-10:
                    # Tangent-plane Laplacian
                    target = self._tangent_plane_laplacian(
                        vi, vertices, neighbors, vertex_normals[vi]
                    )
                else:
                    # Standard Laplacian (centroid)
                    target = vertices[neighbors].mean(axis=0)
                
                # Blend towards target
                new_pos = vertices[vi] * (1 - effective_strength) + target * effective_strength
                
                # Project to surface immediately
                if self.config.project_every_step and self.original_mesh is not None:
                    new_pos = self._project_to_surface(
                        new_pos, 
                        strength=self.config.projection_strength
                    )
                
                new_vertices[vi] = new_pos
            
            vertices = new_vertices
            
            # Update normals for next iteration
            if vertex_normals is not None and self.original_mesh is not None:
                vertex_normals = self._compute_surface_normals(vertices)
        
        return vertices
    
    def _tangent_plane_laplacian(
        self,
        vi: int,
        vertices: np.ndarray,
        neighbors: list,
        normal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Laplacian smoothing target in the tangent plane.
        
        This is the Laplacian-Beltrami approach that preserves
        surface conformity better than standard Laplacian.
        """
        current = vertices[vi]
        neighbor_verts = vertices[neighbors]
        centroid = neighbor_verts.mean(axis=0)
        
        # Normalize normal
        normal = normal / np.linalg.norm(normal)
        
        # Project centroid offset to tangent plane
        offset = centroid - current
        tangent_offset = offset - np.dot(offset, normal) * normal
        
        return current + tangent_offset
    
    def _project_to_surface(
        self,
        position: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Project a point to the original surface."""
        try:
            closest, _, _ = self.original_mesh.nearest.on_surface([position])
            projected = closest[0]
            
            if strength < 1.0:
                return position * (1 - strength) + projected * strength
            return projected
        except Exception:
            return position
    
    def _compute_surface_normals(self, vertices: np.ndarray) -> np.ndarray:
        """Get surface normals at vertex positions by projecting and sampling."""
        n_verts = len(vertices)
        normals = np.zeros((n_verts, 3))
        
        try:
            _, _, face_indices = self.original_mesh.nearest.on_surface(vertices)
            
            for i, fi in enumerate(face_indices):
                if fi >= 0 and fi < len(self.original_mesh.face_normals):
                    normals[i] = self.original_mesh.face_normals[fi]
        except Exception:
            pass
        
        return normals
    
    def _build_adjacency(self, vertices: np.ndarray, faces: np.ndarray) -> list:
        """Build vertex adjacency list."""
        adjacency = [set() for _ in range(len(vertices))]
        
        for face in faces:
            unique = list(set(face))
            for i, vi in enumerate(unique):
                for j, vj in enumerate(unique):
                    if i != j:
                        adjacency[vi].add(vj)
        
        return [list(adj) for adj in adjacency]
    
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


class LocalRelaxBrush:
    """
    Apply relaxation to a local region (like RetopoFlow's Relax Brush).
    
    Useful for targeted smoothing of specific areas while preserving
    the rest of the mesh.
    """
    
    def __init__(
        self,
        original_mesh=None,
        radius: float = 1.0,
        strength: float = 0.5,
        falloff: str = "smooth",  # "smooth", "linear", "constant"
    ):
        self.original_mesh = original_mesh
        self.radius = radius
        self.strength = strength
        self.falloff = falloff
    
    def apply(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        center: np.ndarray,
        iterations: int = 3,
    ) -> np.ndarray:
        """
        Apply local relaxation around a center point.
        
        Args:
            vertices: Vertex positions
            faces: Face indices
            center: Center of brush application
            iterations: Number of relaxation iterations
            
        Returns:
            Modified vertices
        """
        vertices = vertices.copy()
        
        # Find vertices within brush radius
        distances = np.linalg.norm(vertices - center, axis=1)
        affected_mask = distances < self.radius
        affected_indices = np.where(affected_mask)[0]
        
        if len(affected_indices) == 0:
            return vertices
        
        # Compute per-vertex weights based on falloff
        weights = self._compute_weights(distances[affected_indices])
        
        # Build local adjacency
        adjacency = self._build_local_adjacency(vertices, faces, set(affected_indices))
        
        # Iterative local relaxation
        for _ in range(iterations):
            new_vertices = vertices.copy()
            
            for i, vi in enumerate(affected_indices):
                neighbors = adjacency.get(vi, [])
                if not neighbors:
                    continue
                
                # Compute target (centroid)
                target = vertices[neighbors].mean(axis=0)
                
                # Blend with weight
                effective_strength = self.strength * weights[i]
                new_pos = vertices[vi] * (1 - effective_strength) + target * effective_strength
                
                # Project to surface
                if self.original_mesh is not None:
                    try:
                        closest, _, _ = self.original_mesh.nearest.on_surface([new_pos])
                        new_pos = closest[0]
                    except Exception:
                        pass
                
                new_vertices[vi] = new_pos
            
            vertices = new_vertices
        
        return vertices
    
    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute falloff weights for affected vertices."""
        normalized = distances / self.radius
        
        if self.falloff == "smooth":
            # Smooth step falloff
            weights = 1.0 - normalized * normalized * (3 - 2 * normalized)
        elif self.falloff == "linear":
            weights = 1.0 - normalized
        else:  # constant
            weights = np.ones_like(normalized)
        
        return np.clip(weights, 0, 1)
    
    def _build_local_adjacency(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        affected: set,
    ) -> dict:
        """Build adjacency only for affected vertices."""
        adjacency = {vi: set() for vi in affected}
        
        for face in faces:
            unique = list(set(face))
            for i, vi in enumerate(unique):
                if vi not in affected:
                    continue
                for j, vj in enumerate(unique):
                    if i != j:
                        adjacency[vi].add(vj)
        
        return {k: list(v) for k, v in adjacency.items()}


def relax_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    original_mesh=None,
    iterations: int = 10,
    strength: float = 0.5,
    feature_vertices: Optional[Set[int]] = None,
) -> np.ndarray:
    """
    Convenience function for mesh relaxation with projection.
    
    Args:
        vertices: Vertex positions
        faces: Face indices
        original_mesh: trimesh for projection
        iterations: Number of iterations
        strength: Relaxation strength (0-1)
        feature_vertices: Feature edge vertices to preserve
        
    Returns:
        Relaxed vertices
    """
    config = RelaxationConfig(
        iterations=iterations,
        strength=strength,
        preserve_features=feature_vertices is not None,
    )
    
    relaxer = ProjectedRelaxation(original_mesh, config)
    return relaxer.relax(vertices, faces, feature_vertices)


def relax_with_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    original_mesh=None,
    feature_set=None,  # FeatureSet from analysis
    iterations: int = 10,
) -> np.ndarray:
    """
    Relax mesh while preserving detected features.
    
    Args:
        vertices: Vertex positions  
        faces: Face indices
        original_mesh: trimesh for projection
        feature_set: FeatureSet from feature detection
        iterations: Number of iterations
        
    Returns:
        Relaxed vertices
    """
    feature_vertices = None
    if feature_set is not None:
        feature_vertices = set(feature_set.get_feature_vertices())
    
    return relax_mesh(
        vertices=vertices,
        faces=faces,
        original_mesh=original_mesh,
        iterations=iterations,
        feature_vertices=feature_vertices,
    )
