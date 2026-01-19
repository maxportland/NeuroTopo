"""
Feature detection for meshes.

Detects sharp edges, corners, and other geometric features that should
be preserved during retopology.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.core.fields import ScalarField, FieldLocation

logger = logging.getLogger("neurotopo.analysis.features")


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


# Threshold for using fast vectorized algorithm
FAST_THRESHOLD = 50000


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
        self._use_fast = mesh.num_faces > FAST_THRESHOLD
        
        if mesh.face_normals is None:
            mesh.compute_normals()
    
    def detect(self) -> FeatureSet:
        """Run full feature detection."""
        start_time = time.time()
        method = "fast" if self._use_fast else "standard"
        
        if self._use_fast:
            result = self._detect_fast()
        else:
            result = self._detect_standard()
        
        elapsed = time.time() - start_time
        logger.debug(f"Feature detection ({method}): {elapsed:.3f}s, "
                    f"{len(result.edges)} edges, {len(result.points)} points")
        
        return result
    
    def _detect_fast(self) -> FeatureSet:
        """Fast vectorized feature detection for large meshes."""
        faces = np.array(self.mesh.faces)
        normals = self.mesh.face_normals
        n_faces = len(faces)
        
        # Ensure we have triangles
        if faces.shape[1] != 3:
            # Fall back to standard for non-triangular
            return self._detect_standard()
        
        # Build edges as sorted pairs - vectorized
        all_edges = np.zeros((n_faces * 3, 2), dtype=np.int64)
        all_face_idx = np.zeros(n_faces * 3, dtype=np.int64)
        
        for i in range(3):
            start = i * n_faces
            end = (i + 1) * n_faces
            v0 = faces[:, i]
            v1 = faces[:, (i + 1) % 3]
            all_edges[start:end, 0] = np.minimum(v0, v1)
            all_edges[start:end, 1] = np.maximum(v0, v1)
            all_face_idx[start:end] = np.arange(n_faces)
        
        # Sort edges to group duplicates
        edge_keys = all_edges[:, 0] * (self.mesh.num_vertices + 1) + all_edges[:, 1]
        sort_idx = np.argsort(edge_keys)
        sorted_edges = all_edges[sort_idx]
        sorted_face_idx = all_face_idx[sort_idx]
        sorted_keys = edge_keys[sort_idx]
        
        # Find pairs of faces sharing each edge
        # Edges appear consecutively if shared by 2 faces
        is_shared = sorted_keys[:-1] == sorted_keys[1:]
        pair_starts = np.where(is_shared)[0]
        
        # Get face pairs
        face0 = sorted_face_idx[pair_starts]
        face1 = sorted_face_idx[pair_starts + 1]
        
        # Compute dihedral angles for all pairs at once
        n0 = normals[face0]
        n1 = normals[face1]
        cos_angle = np.clip(np.sum(n0 * n1, axis=1), -1, 1)
        dihedral = np.arccos(cos_angle)
        
        # Find sharp edges
        sharp_mask = dihedral > self.angle_threshold
        sharp_idx = pair_starts[sharp_mask]
        sharp_angles = dihedral[sharp_mask]
        
        # Extract feature edges
        feature_edges = []
        sharp_edge_v0v1 = sorted_edges[sharp_idx]
        for i, (v0, v1) in enumerate(sharp_edge_v0v1):
            feature_edges.append(FeatureEdge(int(v0), int(v1), float(sharp_angles[i])))
        
        # Also find boundary edges (appear only once)
        # Count occurrences of each edge key
        unique_keys, counts = np.unique(sorted_keys, return_counts=True)
        boundary_mask = counts == 1
        boundary_keys = unique_keys[boundary_mask]
        
        # Find boundary edges
        if len(boundary_keys) > 0:
            # Get first occurrence of each boundary edge
            boundary_idx = np.searchsorted(sorted_keys, boundary_keys)
            boundary_edges = sorted_edges[boundary_idx]
            for v0, v1 in boundary_edges:
                feature_edges.append(FeatureEdge(int(v0), int(v1), sharpness=np.pi))
        
        # Detect corners and compute importance
        feature_points = self._detect_corners(feature_edges)
        vertex_importance = self._compute_vertex_importance(feature_edges)
        
        # Build edge mask (simplified for fast version)
        edge_mask = np.zeros(len(feature_edges), dtype=bool)
        edge_mask[:] = True
        
        return FeatureSet(
            edges=feature_edges,
            points=feature_points,
            edge_mask=edge_mask,
            vertex_importance=vertex_importance
        )
    
    def _detect_standard(self) -> FeatureSet:
        """Standard feature detection for small meshes."""
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
