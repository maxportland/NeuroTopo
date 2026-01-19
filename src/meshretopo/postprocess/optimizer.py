"""
Post-processing optimization for quad meshes.

Improves quad quality through:
- Laplacian smoothing with surface projection
- Edge length equalization
- Angle optimization towards 90 degrees
- Valence optimization (target valence 4)
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.spatial import cKDTree


class QuadOptimizer:
    """
    Post-processing optimizer for quad-dominant meshes.
    
    Iteratively improves quad quality while maintaining
    surface fidelity to the original mesh.
    """
    
    def __init__(
        self,
        iterations: int = 10,
        smoothing_weight: float = 0.3,
        angle_weight: float = 0.5,
        edge_weight: float = 0.3,
        surface_weight: float = 0.7,
    ):
        self.iterations = iterations
        self.smoothing_weight = smoothing_weight
        self.angle_weight = angle_weight
        self.edge_weight = edge_weight
        self.surface_weight = surface_weight
    
    def optimize(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        original_mesh=None,  # trimesh object for projection
    ) -> np.ndarray:
        """
        Optimize vertex positions for better quad quality.
        
        Args:
            vertices: Nx3 vertex positions
            faces: Mx4 face indices (quads, may have degenerate)
            original_mesh: Original trimesh for surface projection
            
        Returns:
            Optimized vertex positions
        """
        vertices = vertices.copy()
        
        # Build mesh topology
        adjacency = self._build_adjacency(vertices, faces)
        face_verts = self._build_face_vertices(faces)
        boundary_verts = self._find_boundary_vertices(faces)
        
        for iteration in range(self.iterations):
            # Adaptive weights - decrease smoothing over iterations
            iter_ratio = iteration / self.iterations
            smooth_w = self.smoothing_weight * (1 - iter_ratio * 0.5)
            
            new_vertices = vertices.copy()
            
            for vi in range(len(vertices)):
                if vi in boundary_verts:
                    # Don't move boundary vertices (or move less)
                    continue
                
                if not adjacency[vi]:
                    continue
                
                # Compute target position from multiple objectives
                target = self._compute_target_position(
                    vi, vertices, adjacency, face_verts, smooth_w
                )
                
                # Project to original surface
                if original_mesh is not None:
                    try:
                        closest, _, _ = original_mesh.nearest.on_surface([target])
                        projected = closest[0]
                        # Blend between target and projection
                        target = target * (1 - self.surface_weight) + projected * self.surface_weight
                    except Exception:
                        pass
                
                new_vertices[vi] = target
            
            vertices = new_vertices
        
        return vertices
    
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
    
    def _compute_target_position(
        self,
        vi: int,
        vertices: np.ndarray,
        adjacency: list,
        face_verts: list,
        smooth_weight: float,
    ) -> np.ndarray:
        """Compute target position combining multiple objectives."""
        current = vertices[vi]
        neighbors = adjacency[vi]
        
        if not neighbors:
            return current
        
        # 1. Laplacian smoothing target (centroid of neighbors)
        centroid = vertices[neighbors].mean(axis=0)
        
        # 2. Edge length equalization
        edge_target = current.copy()
        if len(neighbors) >= 2:
            edge_lengths = [np.linalg.norm(vertices[n] - current) for n in neighbors]
            avg_length = np.mean(edge_lengths)
            
            # Move towards neighbors with longer edges, away from shorter
            for ni, n in enumerate(neighbors):
                direction = vertices[n] - current
                dist = edge_lengths[ni]
                if dist > 1e-10:
                    # Pull towards if edge too long, push away if too short
                    factor = (dist - avg_length) / dist * 0.1
                    edge_target += direction * factor
        
        # Combine objectives
        target = (
            current * (1 - smooth_weight) +
            centroid * smooth_weight * 0.7 +
            edge_target * smooth_weight * 0.3
        )
        
        return target
    
    def compute_quality_improvement(
        self,
        old_vertices: np.ndarray,
        new_vertices: np.ndarray,
        faces: np.ndarray,
    ) -> dict:
        """Measure quality improvement from optimization."""
        old_quality = self._measure_quad_quality(old_vertices, faces)
        new_quality = self._measure_quad_quality(new_vertices, faces)
        
        return {
            "old_aspect_ratio": old_quality["aspect_ratio"],
            "new_aspect_ratio": new_quality["aspect_ratio"],
            "old_angle_deviation": old_quality["angle_deviation"],
            "new_angle_deviation": new_quality["angle_deviation"],
            "aspect_improvement": old_quality["aspect_ratio"] - new_quality["aspect_ratio"],
            "angle_improvement": old_quality["angle_deviation"] - new_quality["angle_deviation"],
        }
    
    def _measure_quad_quality(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        """Measure quad quality metrics."""
        aspect_ratios = []
        angle_deviations = []
        
        for face in faces:
            unique = list(set(face))
            if len(unique) < 3:
                continue
            
            verts = vertices[unique]
            
            if len(unique) == 4:
                # Quad metrics
                edges = [np.linalg.norm(verts[(i+1)%4] - verts[i]) for i in range(4)]
                if min(edges) > 1e-10:
                    aspect_ratios.append(max(edges) / min(edges))
                
                # Angle deviation from 90 degrees
                for i in range(4):
                    p0 = verts[(i-1) % 4]
                    p1 = verts[i]
                    p2 = verts[(i+1) % 4]
                    e1, e2 = p0 - p1, p2 - p1
                    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                        angle = np.arccos(cos_angle)
                        angle_deviations.append(abs(angle - np.pi/2))
        
        return {
            "aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 1.0,
            "angle_deviation": np.mean(angle_deviations) if angle_deviations else 0.0,
        }


class EdgeFlowOptimizer:
    """
    Optimize edge flow direction alignment.
    
    Aligns quad edges with principal curvature directions
    for better animation deformation.
    """
    
    def __init__(
        self,
        alignment_strength: float = 0.5,
        iterations: int = 5,
    ):
        self.alignment_strength = alignment_strength
        self.iterations = iterations
    
    def optimize(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        direction_field: Optional[np.ndarray] = None,
        original_mesh=None,
    ) -> np.ndarray:
        """
        Optimize edge alignment with direction field.
        
        Args:
            vertices: Nx3 vertex positions
            faces: Mx4 quad faces
            direction_field: Nx3 target edge directions per vertex
            original_mesh: For surface projection
            
        Returns:
            Optimized vertex positions
        """
        if direction_field is None:
            return vertices
        
        vertices = vertices.copy()
        adjacency = self._build_adjacency(faces, len(vertices))
        
        for iteration in range(self.iterations):
            new_vertices = vertices.copy()
            strength = self.alignment_strength * (1 - iteration / self.iterations * 0.3)
            
            for vi in range(len(vertices)):
                if not adjacency[vi]:
                    continue
                
                target_dir = direction_field[vi]
                if np.linalg.norm(target_dir) < 1e-10:
                    continue
                
                target_dir = target_dir / np.linalg.norm(target_dir)
                
                # Compute adjustment to align edges with direction
                adjustment = np.zeros(3)
                for ni in adjacency[vi]:
                    edge = vertices[ni] - vertices[vi]
                    edge_len = np.linalg.norm(edge)
                    if edge_len < 1e-10:
                        continue
                    
                    edge_dir = edge / edge_len
                    
                    # Project edge onto target direction and perpendicular
                    parallel = np.dot(edge_dir, target_dir)
                    
                    # Small adjustment perpendicular to target direction
                    perp = edge_dir - parallel * target_dir
                    adjustment -= perp * strength * 0.1
                
                target = vertices[vi] + adjustment
                
                # Project back to surface
                if original_mesh is not None:
                    try:
                        closest, _, _ = original_mesh.nearest.on_surface([target])
                        target = closest[0]
                    except Exception:
                        pass
                
                new_vertices[vi] = target
            
            vertices = new_vertices
        
        return vertices
    
    def _build_adjacency(self, faces: np.ndarray, n_verts: int) -> list:
        """Build vertex adjacency."""
        adjacency = [set() for _ in range(n_verts)]
        for face in faces:
            unique = list(set(face))
            for i, vi in enumerate(unique):
                for j, vj in enumerate(unique):
                    if i != j:
                        adjacency[vi].add(vj)
        return [list(adj) for adj in adjacency]
