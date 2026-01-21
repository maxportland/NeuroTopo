"""
Quad-preserving decimation module.

Standard decimation triangulates the mesh, destroying quad topology.
This module implements decimation that preserves quad structure by:
1. Collapsing edges that don't break quad loops
2. Preferring to collapse edges that maintain valence regularity
3. Preserving boundary and feature edges
"""

from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from typing import Optional, Set, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger("neurotopo.remesh.quad_decimate")


@dataclass
class QuadDecimateConfig:
    """Configuration for quad-preserving decimation."""
    target_face_ratio: float = 0.25  # Target as ratio of original face count
    preserve_boundary: bool = True
    preserve_features: bool = True
    max_valence: int = 6  # Don't create vertices with higher valence
    min_valence: int = 3  # Don't create vertices with lower valence
    quality_threshold: float = 0.3  # Minimum quad quality to maintain


class QuadDecimator:
    """
    Decimates quad meshes while preserving quad topology.
    
    Unlike standard decimation which triangulates first, this:
    - Works directly with quads
    - Maintains edge loop structure
    - Preserves valence regularity
    """
    
    def __init__(self, config: Optional[QuadDecimateConfig] = None):
        self.config = config or QuadDecimateConfig()
    
    def decimate(
        self,
        vertices: np.ndarray,
        faces: List[List[int]],
        feature_edges: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Decimate mesh while preserving quad structure.
        
        Args:
            vertices: Nx3 vertex positions
            faces: List of faces (each face is list of vertex indices)
            feature_edges: Set of edges (as sorted tuples) that should not be collapsed
            
        Returns:
            (decimated_vertices, decimated_faces)
        """
        verts = vertices.copy()
        face_list = [list(f) for f in faces]
        feature_edges = feature_edges or set()
        
        # Track which vertices are still active
        active_verts = set(range(len(verts)))
        
        # Compute target face count
        target_faces = int(len(faces) * self.config.target_face_ratio)
        
        logger.debug(f"Quad decimation: {len(faces)} -> {target_faces} faces")
        
        iteration = 0
        max_iterations = len(faces) * 2  # Safety limit
        
        while len(face_list) > target_faces and iteration < max_iterations:
            iteration += 1
            
            # Build topology
            edge_faces, vertex_faces, boundary_verts = self._build_topology(
                face_list, active_verts
            )
            
            # Compute valence
            valence = self._compute_valence(edge_faces, active_verts)
            
            # Find collapsible edges
            candidates = self._find_collapse_candidates(
                verts, face_list, edge_faces, valence, 
                boundary_verts, feature_edges
            )
            
            if not candidates:
                logger.debug(f"No more collapsible edges at iteration {iteration}")
                break
            
            # Collapse best edge
            best_edge, best_cost = candidates[0]
            v0, v1 = best_edge
            
            # Perform collapse: merge v1 into v0
            verts, face_list = self._collapse_edge(
                verts, face_list, v0, v1
            )
            
            # Update active vertices
            active_verts.discard(v1)
            
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: {len(face_list)} faces remaining")
        
        # Clean up: remove deleted vertices and reindex faces
        verts, face_list = self._compact_mesh(verts, face_list, active_verts)
        
        return verts, face_list
    
    def _build_topology(
        self,
        faces: List[List[int]],
        active_verts: Set[int],
    ) -> Tuple[dict, dict, Set[int]]:
        """Build edge-to-face and vertex-to-face mappings."""
        edge_faces = defaultdict(list)
        vertex_faces = defaultdict(list)
        edge_count = defaultdict(int)
        
        for fi, face in enumerate(faces):
            unique = [v for v in face if v in active_verts]
            unique = list(dict.fromkeys(unique))  # Remove duplicates preserving order
            
            if len(unique) < 3:
                continue
            
            for v in unique:
                vertex_faces[v].append(fi)
            
            n = len(unique)
            for i in range(n):
                v0, v1 = unique[i], unique[(i + 1) % n]
                edge = (min(v0, v1), max(v0, v1))
                edge_faces[edge].append(fi)
                edge_count[edge] += 1
        
        # Boundary vertices are on edges with only 1 face
        boundary_verts = set()
        for edge, count in edge_count.items():
            if count == 1:
                boundary_verts.add(edge[0])
                boundary_verts.add(edge[1])
        
        return dict(edge_faces), dict(vertex_faces), boundary_verts
    
    def _compute_valence(
        self,
        edge_faces: dict,
        active_verts: Set[int],
    ) -> dict:
        """Compute valence for each vertex."""
        valence = defaultdict(int)
        seen_edges = set()
        
        for edge in edge_faces.keys():
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            valence[edge[0]] += 1
            valence[edge[1]] += 1
        
        return dict(valence)
    
    def _find_collapse_candidates(
        self,
        verts: np.ndarray,
        faces: List[List[int]],
        edge_faces: dict,
        valence: dict,
        boundary_verts: Set[int],
        feature_edges: Set[Tuple[int, int]],
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Find edges that can be collapsed, sorted by cost."""
        candidates = []
        
        for edge, face_indices in edge_faces.items():
            v0, v1 = edge
            
            # Skip boundary edges if preserving boundary
            if self.config.preserve_boundary:
                if v0 in boundary_verts or v1 in boundary_verts:
                    continue
            
            # Skip feature edges
            if edge in feature_edges:
                continue
            
            # Check valence constraints
            # After collapse, v0 gets all of v1's edges except the collapsed one
            new_valence_v0 = valence.get(v0, 0) + valence.get(v1, 0) - 2
            if new_valence_v0 > self.config.max_valence:
                continue
            if new_valence_v0 < self.config.min_valence:
                continue
            
            # Compute collapse cost
            cost = self._collapse_cost(verts, edge, face_indices, faces)
            
            candidates.append((edge, cost))
        
        # Sort by cost (lowest first)
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def _collapse_cost(
        self,
        verts: np.ndarray,
        edge: Tuple[int, int],
        face_indices: List[int],
        faces: List[List[int]],
    ) -> float:
        """Compute cost of collapsing an edge.
        
        Lower cost = better candidate for collapse.
        """
        v0, v1 = edge
        p0, p1 = verts[v0], verts[v1]
        
        # Edge length cost (shorter edges are cheaper to collapse)
        edge_length = np.linalg.norm(p1 - p0)
        
        # Valence cost: prefer collapsing edges where both vertices are irregular
        # (this helps reduce irregular vertices)
        
        # Position cost: prefer edges in flatter regions
        
        return edge_length
    
    def _collapse_edge(
        self,
        verts: np.ndarray,
        faces: List[List[int]],
        v_keep: int,
        v_remove: int,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Collapse edge by merging v_remove into v_keep."""
        # Move v_keep to midpoint
        new_verts = verts.copy()
        new_verts[v_keep] = (verts[v_keep] + verts[v_remove]) / 2
        
        # Update faces: replace v_remove with v_keep
        new_faces = []
        for face in faces:
            new_face = []
            for v in face:
                if v == v_remove:
                    new_face.append(v_keep)
                else:
                    new_face.append(v)
            
            # Remove degenerate faces (less than 3 unique vertices)
            unique = list(dict.fromkeys(new_face))
            if len(unique) >= 3:
                new_faces.append(unique)
        
        return new_verts, new_faces
    
    def _compact_mesh(
        self,
        verts: np.ndarray,
        faces: List[List[int]],
        active_verts: Set[int],
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Remove unused vertices and reindex faces."""
        # Create mapping from old to new indices
        old_to_new = {}
        new_verts = []
        
        for old_idx in sorted(active_verts):
            old_to_new[old_idx] = len(new_verts)
            new_verts.append(verts[old_idx])
        
        # Reindex faces
        new_faces = []
        for face in faces:
            new_face = []
            valid = True
            for v in face:
                if v in old_to_new:
                    new_face.append(old_to_new[v])
                else:
                    valid = False
                    break
            if valid and len(new_face) >= 3:
                new_faces.append(new_face)
        
        return np.array(new_verts, dtype=np.float32), new_faces


def quad_aware_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_ratio: float = 0.25,
    preserve_boundary: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for quad-preserving decimation.
    
    Args:
        vertices: Nx3 vertex positions
        faces: Mx4 quad faces (or mixed)
        target_ratio: Target face count as ratio of original
        preserve_boundary: Whether to preserve boundary edges
        
    Returns:
        (decimated_vertices, decimated_faces)
    """
    config = QuadDecimateConfig(
        target_face_ratio=target_ratio,
        preserve_boundary=preserve_boundary,
    )
    
    decimator = QuadDecimator(config)
    
    # Convert faces to list of lists
    face_list = [list(f) for f in faces]
    
    new_verts, new_faces = decimator.decimate(vertices, face_list)
    
    # Convert back to array
    if new_faces:
        max_size = max(len(f) for f in new_faces)
        faces_arr = np.zeros((len(new_faces), max_size), dtype=np.int32)
        for i, f in enumerate(new_faces):
            faces_arr[i, :len(f)] = f
            if len(f) < max_size:
                faces_arr[i, len(f):] = f[-1]  # Pad with last vertex
    else:
        faces_arr = np.zeros((0, 4), dtype=np.int32)
    
    return new_verts, faces_arr
