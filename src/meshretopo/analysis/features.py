"""
Feature detection for meshes.

Detects sharp edges, corners, and other geometric features that should
be preserved during retopology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.core.fields import ScalarField, FieldLocation


@dataclass
class FeatureEdge:
    """Represents a detected feature edge."""
    v0: int  # Start vertex index
    v1: int  # End vertex index
    sharpness: float  # Edge sharpness (dihedral angle deviation from 180°)


@dataclass
class FeaturePoint:
    """Represents a detected feature point (corner)."""
    vertex: int  # Vertex index
    type: str  # "corner", "dart", "cusp", etc.
    importance: float


@dataclass
class FeatureSet:
    """Collection of detected features."""
    edges: list[FeatureEdge]
    points: list[FeaturePoint]
    edge_mask: np.ndarray  # Boolean mask for edges
    vertex_importance: ScalarField
    
    @property
    def num_feature_edges(self) -> int:
        return len(self.edges)
    
    @property
    def num_feature_points(self) -> int:
        return len(self.points)
    
    def get_feature_vertices(self) -> np.ndarray:
        """Get indices of all vertices on feature edges."""
        vertices = set()
        for edge in self.edges:
            vertices.add(edge.v0)
            vertices.add(edge.v1)
        return np.array(sorted(vertices))


class FeatureDetector:
    """
    Detect geometric features in meshes.
    
    Uses dihedral angles and curvature to identify edges and corners
    that need to be preserved during retopology.
    """
    
    def __init__(
        self,
        mesh: Mesh,
        angle_threshold: float = 30.0,  # degrees
        corner_threshold: float = 60.0,  # degrees
    ):
        self.mesh = mesh
        self.angle_threshold = np.radians(angle_threshold)
        self.corner_threshold = np.radians(corner_threshold)
        
        if mesh.face_normals is None:
            mesh.compute_normals()
    
    def detect(self) -> FeatureSet:
        """Run full feature detection."""
        edge_info = self._build_edge_info()
        feature_edges = self._detect_sharp_edges(edge_info)
        feature_points = self._detect_corners(feature_edges)
        vertex_importance = self._compute_vertex_importance(feature_edges)
        
        # Build edge mask
        edge_mask = np.zeros(len(edge_info), dtype=bool)
        for i, (edge_key, _) in enumerate(edge_info.items()):
            for fe in feature_edges:
                if (fe.v0, fe.v1) == edge_key or (fe.v1, fe.v0) == edge_key:
                    edge_mask[i] = True
                    break
        
        return FeatureSet(
            edges=feature_edges,
            points=feature_points,
            edge_mask=edge_mask,
            vertex_importance=vertex_importance
        )
    
    def _build_edge_info(self) -> dict:
        """Build edge-to-face adjacency information."""
        edge_faces = {}  # (v0, v1) -> [face_indices]
        
        for fi, face in enumerate(self.mesh.faces):
            n = len(face)
            for i in range(n):
                v0, v1 = face[i], face[(i + 1) % n]
                edge_key = (min(v0, v1), max(v0, v1))
                
                if edge_key not in edge_faces:
                    edge_faces[edge_key] = []
                edge_faces[edge_key].append(fi)
        
        return edge_faces
    
    def _detect_sharp_edges(self, edge_info: dict) -> list[FeatureEdge]:
        """Detect edges with high dihedral angle."""
        feature_edges = []
        
        for (v0, v1), face_indices in edge_info.items():
            if len(face_indices) != 2:
                # Boundary or non-manifold edge
                if len(face_indices) == 1:
                    feature_edges.append(FeatureEdge(v0, v1, sharpness=np.pi))
                continue
            
            # Compute dihedral angle
            n0 = self.mesh.face_normals[face_indices[0]]
            n1 = self.mesh.face_normals[face_indices[1]]
            
            cos_angle = np.clip(np.dot(n0, n1), -1, 1)
            dihedral = np.arccos(cos_angle)
            
            if dihedral > self.angle_threshold:
                feature_edges.append(FeatureEdge(v0, v1, sharpness=dihedral))
        
        return feature_edges
    
    def _detect_corners(self, feature_edges: list[FeatureEdge]) -> list[FeaturePoint]:
        """Detect corner vertices where multiple feature edges meet."""
        # Count feature edges per vertex
        vertex_edge_count = {}
        vertex_directions = {}  # vertex -> list of edge directions
        
        for edge in feature_edges:
            for v in [edge.v0, edge.v1]:
                vertex_edge_count[v] = vertex_edge_count.get(v, 0) + 1
            
            # Store edge direction at each endpoint
            direction = self.mesh.vertices[edge.v1] - self.mesh.vertices[edge.v0]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            if edge.v0 not in vertex_directions:
                vertex_directions[edge.v0] = []
            vertex_directions[edge.v0].append(direction)
            
            if edge.v1 not in vertex_directions:
                vertex_directions[edge.v1] = []
            vertex_directions[edge.v1].append(-direction)
        
        feature_points = []
        
        for v, count in vertex_edge_count.items():
            if count >= 3:
                # Vertex where 3+ feature edges meet
                feature_points.append(FeaturePoint(
                    vertex=v,
                    type="corner",
                    importance=1.0
                ))
            elif count == 2:
                # Check if edges form a sharp angle
                dirs = vertex_directions.get(v, [])
                if len(dirs) >= 2:
                    cos_angle = np.dot(dirs[0], dirs[1])
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    
                    if abs(np.pi - angle) > self.corner_threshold:
                        feature_points.append(FeaturePoint(
                            vertex=v,
                            type="corner",
                            importance=0.8
                        ))
        
        return feature_points
    
    def _compute_vertex_importance(self, feature_edges: list[FeatureEdge]) -> ScalarField:
        """Compute per-vertex importance based on features."""
        importance = np.zeros(self.mesh.num_vertices)
        
        for edge in feature_edges:
            # Sharpness contributes to importance
            weight = edge.sharpness / np.pi  # Normalize to [0, 1]
            importance[edge.v0] = max(importance[edge.v0], weight)
            importance[edge.v1] = max(importance[edge.v1], weight)
        
        return ScalarField(importance, FieldLocation.VERTEX, "feature_importance")


def detect_features(
    mesh: Mesh,
    angle_threshold: float = 30.0,
    corner_threshold: float = 60.0
) -> FeatureSet:
    """Convenience function for feature detection."""
    detector = FeatureDetector(mesh, angle_threshold, corner_threshold)
    return detector.detect()
