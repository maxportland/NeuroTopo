"""
Pole reduction utilities for topology optimization.

High-valence vertices (poles with 6+ edges) cause issues with:
- Subdivision surfaces (create artifacts)
- Deformation (unpredictable behavior)
- Edge flow (interrupt continuous loops)

This module provides tools to reduce pole count through:
- Edge collapse: Merge high-valence vertex with neighbor
- Face split: Split faces to redistribute valence
- Local remeshing: Reconstruct topology around problem areas
"""

from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, Set

logger = logging.getLogger("neurotopo.postprocess.pole_reduction")


class PoleReducer:
    """
    Reduces high-valence vertices in quad-dominant meshes.
    
    Target: Minimize vertices with valence != 4 (for quads) or != 6 (for tris)
    """
    
    def __init__(
        self,
        max_valence: int = 5,  # Vertices above this are candidates for reduction
        min_valence: int = 3,  # Don't create vertices below this
        iterations: int = 3,
        preserve_boundary: bool = True,
    ):
        self.max_valence = max_valence
        self.min_valence = min_valence
        self.iterations = iterations
        self.preserve_boundary = preserve_boundary
    
    def reduce(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce high-valence vertices.
        
        Returns:
            (new_vertices, new_faces)
        """
        verts = vertices.copy()
        face_list = [list(f) for f in faces]
        
        for iteration in range(self.iterations):
            # Compute current valence
            valence = self._compute_valence(face_list, len(verts))
            boundary = self._find_boundary_vertices(face_list) if self.preserve_boundary else set()
            
            # Find high-valence vertices
            high_valence_verts = []
            for vi, val in enumerate(valence):
                if val > self.max_valence and vi not in boundary:
                    high_valence_verts.append((val, vi))
            
            if not high_valence_verts:
                break
            
            # Sort by valence (highest first)
            high_valence_verts.sort(reverse=True)
            
            # Try to reduce each
            changes_made = 0
            for val, vi in high_valence_verts[:len(high_valence_verts) // 2 + 1]:
                result = self._try_reduce_vertex(vi, verts, face_list, valence, boundary)
                if result is not None:
                    verts, face_list = result
                    valence = self._compute_valence(face_list, len(verts))
                    changes_made += 1
            
            if changes_made == 0:
                break
            
            logger.debug(f"Pole reduction iteration {iteration+1}: reduced {changes_made} vertices")
        
        # Convert back to array
        if face_list:
            max_size = max(len(f) for f in face_list)
            faces_arr = np.zeros((len(face_list), max_size), dtype=np.int32)
            for i, f in enumerate(face_list):
                faces_arr[i, :len(f)] = f
                if len(f) < max_size:
                    faces_arr[i, len(f):] = f[-1]
        else:
            faces_arr = np.zeros((0, 4), dtype=np.int32)
        
        return verts, faces_arr
    
    def _compute_valence(self, faces: list, num_verts: int) -> np.ndarray:
        """Compute vertex valence from face list."""
        edges = set()
        for face in faces:
            n = len(face)
            for i in range(n):
                v0, v1 = int(face[i]), int(face[(i + 1) % n])
                if v0 != v1:  # Skip degenerate edges
                    edges.add((min(v0, v1), max(v0, v1)))
        
        valence = np.zeros(num_verts, dtype=np.int32)
        for e in edges:
            valence[e[0]] += 1
            valence[e[1]] += 1
        return valence
    
    def _find_boundary_vertices(self, faces: list) -> Set[int]:
        """Find vertices on mesh boundary."""
        edge_count = defaultdict(int)
        for face in faces:
            n = len(face)
            for i in range(n):
                v0, v1 = int(face[i]), int(face[(i + 1) % n])
                if v0 != v1:
                    e = (min(v0, v1), max(v0, v1))
                    edge_count[e] += 1
        
        boundary = set()
        for e, count in edge_count.items():
            if count == 1:
                boundary.add(e[0])
                boundary.add(e[1])
        return boundary
    
    def _try_reduce_vertex(
        self,
        vi: int,
        verts: np.ndarray,
        faces: list,
        valence: np.ndarray,
        boundary: Set[int],
    ) -> Optional[Tuple[np.ndarray, list]]:
        """
        Try to reduce valence of vertex vi.
        
        Strategy: Find a neighboring vertex with low valence and collapse
        the edge between them.
        """
        # Find faces containing this vertex
        vertex_faces = []
        for fi, face in enumerate(faces):
            if vi in face:
                vertex_faces.append(fi)
        
        if len(vertex_faces) < 2:
            return None
        
        # Find neighboring vertices
        neighbors = set()
        for fi in vertex_faces:
            face = faces[fi]
            for v in face:
                if v != vi:
                    neighbors.add(v)
        
        # Find best collapse candidate (neighbor with lowest valence that's not too low)
        best_candidate = None
        best_score = float('inf')
        
        for neighbor in neighbors:
            if neighbor in boundary:
                continue
            nval = valence[neighbor]
            if nval < self.min_valence + 1:
                continue  # Would create too-low valence
            
            # Score: prefer neighbors that will result in good combined valence
            # Combined valence after collapse = val[vi] + val[neighbor] - 2 (edges merged)
            combined = valence[vi] + nval - 2
            # We want combined valence close to target (4 for quads)
            score = abs(combined - 4)
            
            if score < best_score:
                best_score = score
                best_candidate = neighbor
        
        if best_candidate is None:
            return None
        
        # Perform edge collapse: merge vi into best_candidate
        return self._collapse_edge(vi, best_candidate, verts, faces)
    
    def _collapse_edge(
        self,
        v_remove: int,
        v_keep: int,
        verts: np.ndarray,
        faces: list,
    ) -> Tuple[np.ndarray, list]:
        """
        Collapse edge by merging v_remove into v_keep.
        """
        # Move v_keep to midpoint (optional, could also keep v_keep position)
        new_pos = (verts[v_remove] + verts[v_keep]) / 2
        new_verts = verts.copy()
        new_verts[v_keep] = new_pos
        
        # Update faces: replace v_remove with v_keep
        new_faces = []
        for face in faces:
            new_face = []
            for v in face:
                if v == v_remove:
                    new_face.append(v_keep)
                else:
                    new_face.append(v)
            
            # Remove degenerate faces (collapsed to line or point)
            unique_verts = list(dict.fromkeys(new_face))  # Preserve order, remove dups
            if len(unique_verts) >= 3:
                new_faces.append(unique_verts)
        
        return new_verts, new_faces
    
    def get_pole_statistics(
        self,
        faces: list,
        num_verts: int,
    ) -> dict:
        """Get statistics about poles in the mesh."""
        valence = self._compute_valence(faces, num_verts)
        
        stats = {
            'total_vertices': num_verts,
            'valence_distribution': {},
            'regular_vertices': 0,
            'poles_3': 0,
            'poles_5': 0,
            'poles_6plus': 0,
        }
        
        for val in valence:
            stats['valence_distribution'][int(val)] = stats['valence_distribution'].get(int(val), 0) + 1
            if val == 4:
                stats['regular_vertices'] += 1
            elif val == 3:
                stats['poles_3'] += 1
            elif val == 5:
                stats['poles_5'] += 1
            elif val >= 6:
                stats['poles_6plus'] += 1
        
        stats['regular_ratio'] = stats['regular_vertices'] / max(1, num_verts)
        stats['high_pole_ratio'] = stats['poles_6plus'] / max(1, num_verts)
        
        return stats


def reduce_poles(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_valence: int = 5,
    iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to reduce high-valence vertices.
    
    Args:
        vertices: Nx3 vertex positions
        faces: MxK face indices
        max_valence: Target maximum valence
        iterations: Number of reduction passes
        
    Returns:
        (new_vertices, new_faces)
    """
    reducer = PoleReducer(
        max_valence=max_valence,
        iterations=iterations,
    )
    return reducer.reduce(vertices, faces)
