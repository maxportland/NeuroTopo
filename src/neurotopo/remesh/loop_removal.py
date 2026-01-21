"""
Edge Loop Removal for Quad Mesh Decimation.

Reduces polygon count by removing entire edge loops while preserving
quad topology. Prioritizes removal of loops in flat/low-curvature areas
to maintain detail where it matters most.

Based on topology guidelines:
- Add more polygons where there's curvature
- Use fewer polygons in flat areas
- Focus geometry on areas that affect the model's outline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from collections import defaultdict

logger = logging.getLogger("neurotopo.remesh.loop_removal")


@dataclass
class EdgeLoop:
    """Represents an edge loop in the mesh."""
    vertices: list[int]  # Ordered list of vertex indices in the loop
    edges: list[tuple[int, int]]  # Edges in the loop
    is_closed: bool  # Whether the loop forms a closed ring
    importance_score: float = 0.0  # Higher = more important to keep


@dataclass 
class LoopRemovalConfig:
    """Configuration for edge loop removal."""
    curvature_weight: float = 0.6  # Weight for curvature in importance scoring
    silhouette_weight: float = 0.3  # Weight for silhouette contribution
    boundary_weight: float = 0.1  # Weight for boundary proximity
    min_loop_length: int = 4  # Minimum vertices in a removable loop
    preserve_boundary: bool = True  # Don't remove boundary loops


class EdgeLoopRemover:
    """
    Removes edge loops from quad meshes to reduce polygon count
    while preserving topology quality.
    
    Strategy:
    1. Detect all edge loops in the mesh
    2. Score each loop by importance (curvature, silhouette, etc.)
    3. Remove lowest-scoring loops until target count reached
    4. Each removal collapses a ring of quads into nothing
    """
    
    def __init__(self, config: Optional[LoopRemovalConfig] = None):
        self.config = config or LoopRemovalConfig()
    
    def reduce(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_faces: int,
        curvature: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reduce mesh by removing edge loops.
        
        Args:
            vertices: Nx3 vertex positions
            faces: Mx4 quad faces (or Mx4 with degenerate quads for tris)
            target_faces: Target number of faces
            curvature: Optional per-vertex curvature values
            
        Returns:
            Reduced (vertices, faces) tuple
        """
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int64)
        
        current_faces = len(faces)
        if current_faces <= target_faces:
            return vertices, faces
        
        # Compute curvature if not provided
        if curvature is None:
            curvature = self._compute_curvature(vertices, faces)
        
        # Iteratively remove loops until we reach target
        iteration = 0
        max_iterations = 100  # Safety limit
        
        while len(faces) > target_faces and iteration < max_iterations:
            iteration += 1
            
            # Find all edge loops
            loops = self._find_edge_loops(vertices, faces)
            
            if not loops:
                logger.debug(f"No more removable loops found at {len(faces)} faces")
                break
            
            # Score loops by importance
            self._score_loops(loops, vertices, faces, curvature)
            
            # Sort by importance (ascending - remove least important first)
            loops.sort(key=lambda l: l.importance_score)
            
            # Try to remove the least important loop
            removed = False
            for loop in loops:
                if loop.importance_score > 0.9:  # Don't remove very important loops
                    continue
                    
                new_verts, new_faces = self._remove_loop(vertices, faces, loop)
                if new_verts is not None and len(new_faces) < len(faces):
                    vertices = new_verts
                    faces = new_faces
                    # Update curvature for remaining vertices
                    curvature = self._compute_curvature(vertices, faces)
                    removed = True
                    logger.debug(f"Removed loop (score={loop.importance_score:.2f}), "
                               f"faces: {current_faces} -> {len(faces)}")
                    break
            
            if not removed:
                logger.debug(f"Could not remove any more loops at {len(faces)} faces")
                break
        
        logger.info(f"Loop removal: {current_faces} -> {len(faces)} faces "
                   f"in {iteration} iterations")
        
        return vertices, faces
    
    def _find_edge_loops(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> list[EdgeLoop]:
        """Find all edge loops in the mesh."""
        loops = []
        
        # Build edge-to-face mapping
        edge_faces = defaultdict(list)
        for fi, face in enumerate(faces):
            n = len(set(face))  # Handle degenerate quads
            unique = list(dict.fromkeys(face))
            for i in range(len(unique)):
                v0, v1 = unique[i], unique[(i + 1) % len(unique)]
                edge = (min(v0, v1), max(v0, v1))
                edge_faces[edge].append(fi)
        
        # Find interior edges (shared by exactly 2 faces)
        interior_edges = {e for e, fs in edge_faces.items() if len(fs) == 2}
        boundary_edges = {e for e, fs in edge_faces.items() if len(fs) == 1}
        
        # Build vertex-to-edge mapping for interior edges only
        vertex_edges = defaultdict(list)
        for e in interior_edges:
            vertex_edges[e[0]].append(e)
            vertex_edges[e[1]].append(e)
        
        # Trace edge loops
        used_edges = set()
        
        for start_edge in interior_edges:
            if start_edge in used_edges:
                continue
            
            # Try to trace a loop from this edge
            loop_edges = [start_edge]
            loop_verts = [start_edge[0], start_edge[1]]
            used_edges.add(start_edge)
            
            # Extend in one direction
            current_vert = start_edge[1]
            while True:
                # Find next edge continuing the loop
                next_edge = None
                for e in vertex_edges[current_vert]:
                    if e in used_edges:
                        continue
                    # Check if this edge continues in a "straight" direction
                    # (perpendicular to the quad diagonal)
                    if self._is_loop_continuation(faces, edge_faces, loop_edges[-1], e):
                        next_edge = e
                        break
                
                if next_edge is None:
                    break
                
                loop_edges.append(next_edge)
                used_edges.add(next_edge)
                next_vert = next_edge[0] if next_edge[1] == current_vert else next_edge[1]
                
                # Check if we've closed the loop
                if next_vert == loop_verts[0]:
                    loops.append(EdgeLoop(
                        vertices=loop_verts,
                        edges=loop_edges,
                        is_closed=True
                    ))
                    break
                
                loop_verts.append(next_vert)
                current_vert = next_vert
                
                # Safety limit
                if len(loop_verts) > len(vertices):
                    break
            
            # If not closed, also extend in the other direction
            if loop_edges and loop_verts[-1] != loop_verts[0]:
                current_vert = start_edge[0]
                while True:
                    next_edge = None
                    for e in vertex_edges[current_vert]:
                        if e in used_edges:
                            continue
                        if self._is_loop_continuation(faces, edge_faces, loop_edges[0], e):
                            next_edge = e
                            break
                    
                    if next_edge is None:
                        break
                    
                    loop_edges.insert(0, next_edge)
                    used_edges.add(next_edge)
                    next_vert = next_edge[0] if next_edge[1] == current_vert else next_edge[1]
                    loop_verts.insert(0, next_vert)
                    current_vert = next_vert
                    
                    if len(loop_verts) > len(vertices):
                        break
                
                # Add open loop if long enough
                if len(loop_verts) >= self.config.min_loop_length:
                    loops.append(EdgeLoop(
                        vertices=loop_verts,
                        edges=loop_edges,
                        is_closed=False
                    ))
        
        # Filter out boundary loops if configured
        if self.config.preserve_boundary:
            boundary_verts = set()
            for e in boundary_edges:
                boundary_verts.add(e[0])
                boundary_verts.add(e[1])
            
            loops = [l for l in loops 
                    if not any(v in boundary_verts for v in l.vertices)]
        
        return loops
    
    def _is_loop_continuation(
        self,
        faces: np.ndarray,
        edge_faces: dict,
        edge1: tuple[int, int],
        edge2: tuple[int, int]
    ) -> bool:
        """Check if edge2 continues the loop from edge1."""
        # Find shared face
        faces1 = set(edge_faces.get(edge1, []))
        faces2 = set(edge_faces.get(edge2, []))
        shared = faces1 & faces2
        
        if not shared:
            return False
        
        # In a quad, opposite edges are parallel
        # edge1 and edge2 should be on the same quad and be opposite edges
        for fi in shared:
            face = faces[fi]
            unique = list(dict.fromkeys(face))
            if len(unique) != 4:
                continue
            
            # Get edge indices in the face
            def edge_in_face(e, f):
                for i in range(4):
                    v0, v1 = f[i], f[(i + 1) % 4]
                    if (min(v0, v1), max(v0, v1)) == e:
                        return i
                return -1
            
            idx1 = edge_in_face(edge1, unique)
            idx2 = edge_in_face(edge2, unique)
            
            if idx1 >= 0 and idx2 >= 0:
                # Opposite edges in a quad are at indices 0,2 or 1,3
                if abs(idx1 - idx2) == 2:
                    return True
        
        return False
    
    def _score_loops(
        self,
        loops: list[EdgeLoop],
        vertices: np.ndarray,
        faces: np.ndarray,
        curvature: np.ndarray
    ) -> None:
        """Score each loop by importance."""
        # Compute mesh bounds for silhouette scoring
        bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        center = vertices.mean(axis=0)
        
        for loop in loops:
            # Curvature score: average curvature along the loop
            loop_curvature = np.mean([curvature[v] for v in loop.vertices])
            curvature_score = np.clip(loop_curvature / (np.pi / 4), 0, 1)
            
            # Silhouette score: how much the loop contributes to the outline
            # Vertices far from center and with normals perpendicular to view
            # contribute more to silhouette
            loop_verts = vertices[loop.vertices]
            distances = np.linalg.norm(loop_verts - center, axis=1)
            silhouette_score = np.mean(distances) / (bbox_diag / 2)
            silhouette_score = np.clip(silhouette_score, 0, 1)
            
            # Boundary proximity score (higher if near boundary)
            boundary_score = 0.0
            if not loop.is_closed:
                boundary_score = 0.5  # Open loops are somewhat important
            
            # Combined importance score
            loop.importance_score = (
                self.config.curvature_weight * curvature_score +
                self.config.silhouette_weight * silhouette_score +
                self.config.boundary_weight * boundary_score
            )
    
    def _remove_loop(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        loop: EdgeLoop
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Remove an edge loop by collapsing it.
        
        For a closed loop, this removes a ring of quads.
        For an open loop, this removes a strip of quads.
        """
        if not loop.is_closed:
            # For open loops, use edge collapse along the loop
            return self._collapse_open_loop(vertices, faces, loop)
        
        # For closed loops, collapse all loop vertices to their neighbors
        return self._collapse_closed_loop(vertices, faces, loop)
    
    def _collapse_closed_loop(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        loop: EdgeLoop
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Collapse a closed edge loop."""
        loop_verts = set(loop.vertices)
        
        # Build adjacency
        vertex_neighbors = defaultdict(set)
        for face in faces:
            unique = list(dict.fromkeys(face))
            for i, v in enumerate(unique):
                for j, u in enumerate(unique):
                    if i != j:
                        vertex_neighbors[v].add(u)
        
        # Find the "other side" vertices for each loop vertex
        # These are neighbors that are not in the loop
        collapse_map = {}
        for v in loop.vertices:
            non_loop_neighbors = [n for n in vertex_neighbors[v] if n not in loop_verts]
            if non_loop_neighbors:
                # Collapse to the first non-loop neighbor
                collapse_map[v] = non_loop_neighbors[0]
            else:
                # Can't collapse this vertex
                return None, None
        
        # Apply collapse: remap vertices
        vert_remap = {i: i for i in range(len(vertices))}
        for old_v, new_v in collapse_map.items():
            vert_remap[old_v] = new_v
        
        # Remap faces and remove degenerate ones
        new_faces = []
        for face in faces:
            new_face = [vert_remap[v] for v in face]
            # Remove duplicate consecutive vertices
            unique_face = []
            for i, v in enumerate(new_face):
                if v != new_face[(i - 1) % len(new_face)]:
                    unique_face.append(v)
            
            # Keep face if it still has at least 3 vertices
            if len(set(unique_face)) >= 3:
                # Pad to 4 if needed
                while len(unique_face) < 4:
                    unique_face.append(unique_face[-1])
                new_faces.append(unique_face[:4])
        
        if not new_faces:
            return None, None
        
        new_faces = np.array(new_faces, dtype=np.int64)
        
        # Remove unused vertices and compact
        used_verts = set(new_faces.flatten())
        vert_compact = {old: new for new, old in enumerate(sorted(used_verts))}
        
        new_vertices = vertices[sorted(used_verts)]
        new_faces = np.array([[vert_compact[v] for v in f] for f in new_faces], dtype=np.int64)
        
        return new_vertices, new_faces
    
    def _collapse_open_loop(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        loop: EdgeLoop
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Collapse an open edge loop (edge strip)."""
        # For open loops, we collapse pairs of adjacent vertices
        # This is more complex, so for now we skip open loops
        # and only handle closed loops
        return None, None
    
    def _compute_curvature(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """Compute per-vertex curvature using angle deficit method."""
        n_verts = len(vertices)
        angle_sum = np.zeros(n_verts)
        
        for face in faces:
            unique = list(dict.fromkeys(face))
            n = len(unique)
            if n < 3:
                continue
            
            verts = vertices[unique]
            
            for i in range(n):
                p0 = verts[(i - 1) % n]
                p1 = verts[i]
                p2 = verts[(i + 1) % n]
                
                e1 = p0 - p1
                e2 = p2 - p1
                n1 = np.linalg.norm(e1)
                n2 = np.linalg.norm(e2)
                
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                    angle = np.arccos(cos_angle)
                    angle_sum[unique[i]] += angle
        
        # Curvature is absolute angle deficit from 2π
        curvature = np.abs(2 * np.pi - angle_sum)
        
        return curvature
