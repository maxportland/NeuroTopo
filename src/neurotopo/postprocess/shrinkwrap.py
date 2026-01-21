"""
Shrinkwrap (Surface Snapping) Module.

Provides RetopoFlow-inspired surface projection techniques for 
ensuring the retopologized mesh tightly conforms to the original surface.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("neurotopo.postprocess.shrinkwrap")


@dataclass
class ShrinkwrapConfig:
    """Configuration for shrinkwrap operation."""
    # Final projection pass settings
    final_projection: bool = True  # Do 100% projection at end
    
    # Iterative projection during optimization
    project_every_iteration: bool = True
    projection_strength: float = 0.9  # How strongly to snap (0-1)
    
    # Offset from surface (for visibility during editing)
    surface_offset: float = 0.0  # Offset along normal
    
    # Smooth projection (avoid sharp changes)
    smooth_projection: bool = True
    smoothing_iterations: int = 2


class Shrinkwrap:
    """
    Surface projection (shrinkwrap) for retopology meshes.
    
    Ensures vertices stay on or very close to the original surface.
    Inspired by RetopoFlow's "always on surface" behavior.
    """
    
    def __init__(
        self,
        original_mesh,  # trimesh object
        config: Optional[ShrinkwrapConfig] = None,
    ):
        """
        Initialize shrinkwrap with the original high-poly mesh.
        
        Args:
            original_mesh: trimesh.Trimesh of the original surface
            config: Shrinkwrap configuration
        """
        self.original_mesh = original_mesh
        self.config = config or ShrinkwrapConfig()
        
        # Pre-compute spatial acceleration structure
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build spatial index for fast nearest-point queries."""
        # trimesh already has efficient nearest-point queries via
        # original_mesh.nearest which uses an RTree/BVH
        pass
    
    def project_vertices(
        self,
        vertices: np.ndarray,
        locked_vertices: Optional[set] = None,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Project all vertices to the original surface.
        
        OPTIMIZED: Fully vectorized projection without per-vertex loops.
        
        Args:
            vertices: Nx3 array of vertex positions
            locked_vertices: Set of vertex indices to skip (feature edges, etc.)
            strength: Override projection strength (0=no change, 1=full projection)
            
        Returns:
            Projected vertex positions
        """
        if self.original_mesh is None:
            return vertices
        
        strength = strength if strength is not None else self.config.projection_strength
        locked = locked_vertices or set()
        
        result = vertices.copy()
        
        # Get indices of vertices to project
        if locked:
            movable_mask = np.ones(len(vertices), dtype=bool)
            for vi in locked:
                if vi < len(vertices):
                    movable_mask[vi] = False
            movable_indices = np.where(movable_mask)[0]
        else:
            movable_indices = np.arange(len(vertices))
        
        if len(movable_indices) == 0:
            return result
        
        # Batch project for efficiency
        movable_verts = vertices[movable_indices]
        
        try:
            closest_points, distances, face_indices = self.original_mesh.nearest.on_surface(movable_verts)
            
            # Vectorized projection with strength blending
            projected = closest_points.copy()
            
            # Apply surface offset if configured (vectorized)
            if self.config.surface_offset != 0.0:
                valid_faces = face_indices >= 0
                if np.any(valid_faces):
                    normals = self.original_mesh.face_normals[face_indices[valid_faces]]
                    projected[valid_faces] += normals * self.config.surface_offset
            
            # Blend based on strength (vectorized)
            blended = movable_verts * (1 - strength) + projected * strength
            
            # Write back all at once
            result[movable_indices] = blended
                
        except Exception as e:
            logger.debug(f"Projection failed: {e}")
        
        return result
    
    def project_single(
        self,
        position: np.ndarray,
        return_normal: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project a single point to the surface.
        
        Args:
            position: 3D position to project
            return_normal: Also return surface normal at projection point
            
        Returns:
            Projected position, and optionally surface normal
        """
        try:
            closest, distance, face_idx = self.original_mesh.nearest.on_surface([position])
            projected = closest[0]
            
            if self.config.surface_offset != 0.0 and face_idx[0] >= 0:
                normal = self.original_mesh.face_normals[face_idx[0]]
                projected = projected + normal * self.config.surface_offset
            
            if return_normal and face_idx[0] >= 0:
                return projected, self.original_mesh.face_normals[face_idx[0]]
            
            return projected, None
            
        except Exception:
            return position, None
    
    def final_shrinkwrap(
        self,
        vertices: np.ndarray,
        locked_vertices: Optional[set] = None,
    ) -> np.ndarray:
        """
        Final 100% projection pass - snaps all movable vertices to surface.
        
        This is the "finishing" step that ensures the retopo mesh
        perfectly conforms to the original.
        
        Args:
            vertices: Vertex positions
            locked_vertices: Vertices to skip (boundary, features)
            
        Returns:
            Fully projected vertices
        """
        if not self.config.final_projection:
            return vertices
        
        logger.debug("Applying final shrinkwrap pass (100% projection)")
        
        # First do full projection
        result = self.project_vertices(vertices, locked_vertices, strength=1.0)
        
        # Optional smoothing pass to avoid harsh transitions
        if self.config.smooth_projection:
            result = self._smooth_projected_vertices(
                result, 
                locked_vertices,
                iterations=self.config.smoothing_iterations
            )
            # Re-project after smoothing
            result = self.project_vertices(result, locked_vertices, strength=1.0)
        
        return result
    
    def _smooth_projected_vertices(
        self,
        vertices: np.ndarray,
        locked_vertices: Optional[set],
        iterations: int = 2,
    ) -> np.ndarray:
        """
        Light tangent-space smoothing to avoid harsh transitions.
        
        Smooths along the surface tangent plane to preserve surface conformity.
        """
        # This is a simple uniform Laplacian - could be enhanced
        # For now, we just return as-is since projection handles the main work
        return vertices


class ProjectedVertexTracker:
    """
    Track vertices during optimization with continuous projection.
    
    Implements the RetopoFlow-style "always on surface" behavior
    where every vertex movement is immediately projected back.
    """
    
    def __init__(
        self,
        shrinkwrap: Shrinkwrap,
        vertices: np.ndarray,
        locked_vertices: Optional[set] = None,
    ):
        self.shrinkwrap = shrinkwrap
        self.vertices = vertices.copy()
        self.locked = locked_vertices or set()
        
        # Track surface normals at each vertex for tangent-plane operations
        self._update_surface_info()
    
    def _update_surface_info(self):
        """Update cached surface information for all vertices."""
        self.surface_normals = np.zeros_like(self.vertices)
        
        for i in range(len(self.vertices)):
            _, normal = self.shrinkwrap.project_single(
                self.vertices[i], 
                return_normal=True
            )
            if normal is not None:
                self.surface_normals[i] = normal
    
    def move_vertex(
        self,
        index: int,
        new_position: np.ndarray,
        project: bool = True,
    ) -> np.ndarray:
        """
        Move a vertex and optionally project to surface.
        
        Args:
            index: Vertex index
            new_position: Desired new position
            project: Whether to project to surface immediately
            
        Returns:
            Final vertex position (may differ from new_position if projected)
        """
        if index in self.locked:
            return self.vertices[index]
        
        if project:
            projected, normal = self.shrinkwrap.project_single(
                new_position, 
                return_normal=True
            )
            self.vertices[index] = projected
            if normal is not None:
                self.surface_normals[index] = normal
        else:
            self.vertices[index] = new_position
        
        return self.vertices[index]
    
    def move_in_tangent_plane(
        self,
        index: int,
        direction: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        """
        Move vertex along the surface tangent plane.
        
        This is useful for smoothing operations that should
        preserve the vertex's relationship to the surface.
        
        Args:
            index: Vertex index
            direction: Movement direction (will be projected to tangent)
            distance: How far to move
            
        Returns:
            New vertex position
        """
        if index in self.locked:
            return self.vertices[index]
        
        normal = self.surface_normals[index]
        if np.linalg.norm(normal) < 1e-10:
            # No normal info, just move and project
            new_pos = self.vertices[index] + direction * distance
            return self.move_vertex(index, new_pos)
        
        # Project direction to tangent plane
        normal = normal / np.linalg.norm(normal)
        tangent_dir = direction - np.dot(direction, normal) * normal
        
        if np.linalg.norm(tangent_dir) > 1e-10:
            tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
            new_pos = self.vertices[index] + tangent_dir * distance
            return self.move_vertex(index, new_pos)
        
        return self.vertices[index]
    
    def get_vertices(self) -> np.ndarray:
        """Get current vertex positions."""
        return self.vertices.copy()


def create_shrinkwrap(original_mesh, **config_kwargs) -> Shrinkwrap:
    """
    Convenience function to create a shrinkwrap instance.
    
    Args:
        original_mesh: trimesh.Trimesh of original surface
        **config_kwargs: Configuration overrides
        
    Returns:
        Configured Shrinkwrap instance
    """
    config = ShrinkwrapConfig(**config_kwargs)
    return Shrinkwrap(original_mesh, config)
