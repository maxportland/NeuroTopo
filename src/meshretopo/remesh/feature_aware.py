"""
Feature-aware remeshing with edge preservation.

Improves retopology by preserving important edges like:
- Sharp creases
- Silhouette edges  
- UV seams (if present)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.spatial import cKDTree

from meshretopo.core.mesh import Mesh
from meshretopo.analysis.features import FeatureDetector


class FeaturePreservingRemesher:
    """
    Remeshing with feature edge preservation.
    
    Ensures that important edges like creases, corners,
    and boundaries are preserved in the output mesh.
    """
    
    def __init__(
        self,
        crease_angle: float = 30.0,  # degrees
        feature_weight: float = 5.0,  # How much to weight feature edges
    ):
        self.crease_angle = crease_angle
        self.feature_weight = feature_weight
    
    def process(
        self,
        mesh: Mesh,
        target_faces: int,
        base_remesher: callable,
    ) -> Mesh:
        """
        Remesh with feature preservation.
        
        Args:
            mesh: Input mesh
            target_faces: Target face count
            base_remesher: Function to perform base remeshing
            
        Returns:
            Remeshed mesh with preserved features
        """
        # Extract feature edges
        detector = FeatureDetector(mesh, angle_threshold=self.crease_angle)
        feature_info = detector.detect()
        
        # Get feature vertices
        feature_vertices = set()
        for edge in feature_info.edges:
            feature_vertices.add(edge.v0)
            feature_vertices.add(edge.v1)
        
        # Create constraint weights
        vertex_weights = np.ones(mesh.num_vertices)
        for vi in feature_vertices:
            vertex_weights[vi] = self.feature_weight
        
        # Run base remesher with constraints
        result = base_remesher(mesh, target_faces)
        
        # Post-process: snap feature vertices back to closest feature points
        if len(feature_vertices) > 0:
            result = self._snap_features(
                result, mesh, list(feature_vertices)
            )
        
        return result
    
    def _snap_features(
        self,
        remeshed: Mesh,
        original: Mesh,
        feature_vertices: list[int],
    ) -> Mesh:
        """Snap remeshed vertices to original feature locations."""
        # Get feature point positions from original
        feature_positions = original.vertices[feature_vertices]
        
        if len(feature_positions) == 0:
            return remeshed
        
        # Build KD-tree of feature positions
        tree = cKDTree(feature_positions)
        
        # Find remeshed vertices near features
        snap_distance = original.diagonal * 0.02  # 2% of diagonal
        
        distances, indices = tree.query(remeshed.vertices, distance_upper_bound=snap_distance)
        
        # Snap close vertices
        new_vertices = remeshed.vertices.copy()
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < snap_distance and idx < len(feature_positions):
                # Blend towards feature position
                blend = max(0.3, 1.0 - dist / snap_distance)
                new_vertices[i] = (
                    (1 - blend) * new_vertices[i] + 
                    blend * feature_positions[idx]
                )
        
        return Mesh(vertices=new_vertices, faces=remeshed.faces)


class AdaptiveDensityRemesher:
    """
    Remeshing with adaptive vertex density based on curvature.
    
    Places more vertices in high-curvature regions and fewer
    in flat areas.
    """
    
    def __init__(
        self,
        curvature_weight: float = 2.0,
        min_density: float = 0.3,
        max_density: float = 3.0,
    ):
        self.curvature_weight = curvature_weight
        self.min_density = min_density
        self.max_density = max_density
    
    def compute_density_field(self, mesh: Mesh) -> np.ndarray:
        """Compute target vertex density from curvature."""
        from meshretopo.analysis.curvature import CurvatureAnalyzer
        
        analyzer = CurvatureAnalyzer()
        curv_info = analyzer.analyze(mesh)
        
        # Use mean curvature magnitude for density
        curvature_mag = np.abs(curv_info.mean_curvature)
        
        # Normalize to [0, 1]
        if curvature_mag.max() > curvature_mag.min():
            normalized = (curvature_mag - curvature_mag.min()) / (curvature_mag.max() - curvature_mag.min())
        else:
            normalized = np.zeros_like(curvature_mag)
        
        # Map to density range
        density = self.min_density + (self.max_density - self.min_density) * normalized ** self.curvature_weight
        
        return density
    
    def compute_target_edge_length(
        self,
        mesh: Mesh,
        target_faces: int,
    ) -> np.ndarray:
        """Compute per-vertex target edge length."""
        # Compute base edge length from target face count
        current_area = mesh.total_area
        area_per_face = current_area / target_faces
        base_edge = np.sqrt(2 * area_per_face)  # Approximate for triangles
        
        # Get density field
        density = self.compute_density_field(mesh)
        
        # Inverse density gives edge length
        target_lengths = base_edge / density
        
        return target_lengths


class BoundaryPreservingRemesher:
    """
    Ensures mesh boundaries are preserved during remeshing.
    """
    
    def __init__(self, boundary_smoothing: float = 0.1):
        self.boundary_smoothing = boundary_smoothing
    
    def find_boundary_vertices(self, mesh: Mesh) -> set[int]:
        """Find vertices on mesh boundary."""
        # Build edge-to-face mapping
        edge_faces = {}
        for fi, face in enumerate(mesh.faces):
            n = len(face)
            for i in range(n):
                edge = tuple(sorted([face[i], face[(i+1) % n]]))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)
        
        # Boundary edges have only one face
        boundary_vertices = set()
        for edge, faces in edge_faces.items():
            if len(faces) == 1:
                boundary_vertices.add(edge[0])
                boundary_vertices.add(edge[1])
        
        return boundary_vertices
    
    def preserve_boundary(
        self,
        original: Mesh,
        remeshed: Mesh,
    ) -> Mesh:
        """Snap remeshed boundary to original boundary."""
        orig_boundary = self.find_boundary_vertices(original)
        new_boundary = self.find_boundary_vertices(remeshed)
        
        if not orig_boundary or not new_boundary:
            return remeshed
        
        # Get boundary positions
        orig_positions = original.vertices[list(orig_boundary)]
        tree = cKDTree(orig_positions)
        
        # Snap new boundary vertices
        new_vertices = remeshed.vertices.copy()
        for vi in new_boundary:
            dist, idx = tree.query(new_vertices[vi])
            # Blend towards closest original boundary
            blend = 1.0 - self.boundary_smoothing
            new_vertices[vi] = (
                self.boundary_smoothing * new_vertices[vi] +
                blend * orig_positions[idx]
            )
        
        return Mesh(vertices=new_vertices, faces=remeshed.faces)


def feature_aware_remesh(
    mesh: Mesh,
    target_faces: int,
    base_remesher: callable,
    crease_angle: float = 30.0,
    preserve_boundaries: bool = True,
) -> Mesh:
    """
    Convenience function for feature-aware remeshing.
    
    Args:
        mesh: Input mesh
        target_faces: Target face count
        base_remesher: Base remeshing function
        crease_angle: Angle threshold for creases (degrees)
        preserve_boundaries: Whether to preserve mesh boundaries
        
    Returns:
        Remeshed mesh with preserved features
    """
    # Feature preservation
    feature_remesher = FeaturePreservingRemesher(crease_angle=crease_angle)
    result = feature_remesher.process(mesh, target_faces, base_remesher)
    
    # Boundary preservation
    if preserve_boundaries:
        boundary_remesher = BoundaryPreservingRemesher()
        result = boundary_remesher.preserve_boundary(mesh, result)
    
    return result
