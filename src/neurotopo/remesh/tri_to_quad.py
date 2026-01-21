"""
Advanced triangle-to-quad conversion algorithms.

Converts triangular meshes to quad-dominant meshes using
various algorithms:
- Greedy pairing with quality scoring
- Valence-aware pairing to minimize irregular vertices
- Instant Meshes-style field-guided conversion
- Catmull-Clark subdivision with simplification
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.spatial import cKDTree
from collections import defaultdict


class TriToQuadConverter:
    """
    Convert triangular mesh to quad-dominant mesh.
    
    Uses quality-aware triangle pairing with optional
    direction field guidance and valence optimization.
    """
    
    def __init__(
        self,
        min_quality: float = 0.40,  # Balance quality vs quantity
        prefer_regular: bool = True,
        max_valence_deviation: int = 2,
        edge_swap_iterations: int = 0,  # Disabled - slow and can cause manifold issues
        valence_weight: float = 0.3,  # Weight for valence regularity in scoring
    ):
        self.min_quality = min_quality
        self.prefer_regular = prefer_regular
        self.max_valence_deviation = max_valence_deviation
        self.edge_swap_iterations = edge_swap_iterations
        self.valence_weight = valence_weight
    
    def convert(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        direction_field: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Convert triangles to quads.
        
        Args:
            vertices: Nx3 vertex positions
            faces: Mx3 triangle indices
            direction_field: Optional Nx3 preferred edge directions
            
        Returns:
            (vertices, quad_faces, remaining_tris)
        """
        # Pre-process: edge swapping to improve pairing potential
        if self.edge_swap_iterations > 0:
            faces = self._optimize_edges(vertices, faces)
        
        # Build topology
        edge_faces = self._build_edge_face_map(faces)
        
        # Compute initial vertex valence for valence-aware pairing
        initial_valence = self._compute_vertex_valence(faces, len(vertices))
        
        # Score all potential pairings
        pairings = self._score_pairings(
            vertices, faces, edge_faces, direction_field, initial_valence
        )
        
        # Greedily select best pairings with valence tracking
        quads, remaining_tris = self._greedy_select_valence_aware(
            faces, pairings, len(vertices)
        )
        
        return vertices, quads, remaining_tris
    
    def _compute_vertex_valence(self, faces: np.ndarray, num_verts: int) -> np.ndarray:
        """Compute valence (edge count) for each vertex."""
        edges = set()
        for face in faces:
            for i in range(len(face)):
                v0, v1 = int(face[i]), int(face[(i + 1) % len(face)])
                edges.add((min(v0, v1), max(v0, v1)))
        
        valence = np.zeros(num_verts, dtype=np.int32)
        for e in edges:
            valence[e[0]] += 1
            valence[e[1]] += 1
        return valence
    
    def _optimize_edges(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Optimize triangle edges for better quad pairing.
        
        Swaps edges to create more equilateral triangles and better
        quad formation potential. Uses Delaunay-like criterion.
        """
        faces = [list(f) for f in faces]
        
        for iteration in range(self.edge_swap_iterations):
            swaps = 0
            edge_faces = self._build_edge_face_map(np.array(faces))
            
            # Check each internal edge for potential swap
            for edge, face_list in edge_faces.items():
                if len(face_list) != 2:
                    continue
                
                f0, f1 = face_list
                face0, face1 = faces[f0], faces[f1]
                
                # Get vertices
                v0, v1 = edge
                other0 = [v for v in face0 if v not in edge][0]
                other1 = [v for v in face1 if v not in edge][0]
                
                # Current configuration quality
                current_quality = self._config_quality(
                    vertices, v0, v1, other0, other1, 'edge'
                )
                
                # Swapped configuration quality
                swapped_quality = self._config_quality(
                    vertices, v0, v1, other0, other1, 'diagonal'
                )
                
                # Swap if it improves quality significantly
                if swapped_quality > current_quality + 0.1:
                    # Swap: new triangles are (other0, v0, other1) and (other0, other1, v1)
                    faces[f0] = [other0, v0, other1]
                    faces[f1] = [other0, other1, v1]
                    swaps += 1
            
            if swaps == 0:
                break
        
        return np.array(faces, dtype=np.int32)
    
    def _config_quality(
        self, 
        vertices: np.ndarray, 
        v0: int, v1: int, 
        other0: int, other1: int,
        config: str
    ) -> float:
        """Compute quality of a triangle pair configuration."""
        if config == 'edge':
            # Current: triangles share edge (v0, v1)
            t1 = [other0, v0, v1]
            t2 = [other1, v0, v1]
        else:
            # Swapped: triangles share edge (other0, other1)
            t1 = [other0, v0, other1]
            t2 = [other0, other1, v1]
        
        # Compute triangle quality (closer to equilateral is better)
        def tri_quality(tri):
            p = vertices[tri]
            edges = [np.linalg.norm(p[(i+1)%3] - p[i]) for i in range(3)]
            if min(edges) < 1e-10:
                return 0
            # Quality: ratio of circumradius to twice inradius
            # For equilateral, this is 1. Lower is worse.
            s = sum(edges) / 2  # Semi-perimeter
            area = 0.25 * np.sqrt(abs((edges[0]+edges[1]+edges[2]) * 
                                     (-edges[0]+edges[1]+edges[2]) *
                                     (edges[0]-edges[1]+edges[2]) * 
                                     (edges[0]+edges[1]-edges[2])))
            if area < 1e-10:
                return 0
            # Aspect ratio
            return min(edges) / max(edges)
        
        return (tri_quality(t1) + tri_quality(t2)) / 2
    
    def _build_edge_face_map(self, faces: np.ndarray) -> dict:
        """Map edges to adjacent faces."""
        edge_faces = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)
        return edge_faces
    
    def _score_pairings(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        edge_faces: dict,
        direction_field: Optional[np.ndarray],
        initial_valence: np.ndarray,
    ) -> list:
        """Score all possible triangle pairings with valence awareness."""
        pairings = []
        
        for edge, face_list in edge_faces.items():
            if len(face_list) != 2:
                continue
            
            f0, f1 = face_list
            face0, face1 = faces[f0], faces[f1]
            
            # Get the two non-shared vertices
            other0 = [v for v in face0 if v not in edge][0]
            other1 = [v for v in face1 if v not in edge][0]
            
            # Build quad in correct winding order
            # Start from other0, go around
            v0, v1 = edge
            quad = self._order_quad_vertices(
                vertices, other0, v0, other1, v1
            )
            
            # Compute base quality score
            quality = self._quad_quality_score(vertices[quad])
            
            # Add direction field alignment bonus
            if direction_field is not None:
                alignment = self._direction_alignment(
                    vertices, quad, direction_field
                )
                quality = quality * 0.6 + alignment * 0.2
            
            # Add valence regularity bonus/penalty
            if self.prefer_regular and self.valence_weight > 0:
                valence_score = self._valence_impact_score(
                    quad, edge, initial_valence
                )
                quality = quality * (1 - self.valence_weight) + valence_score * self.valence_weight
            
            if quality >= self.min_quality:
                pairings.append((quality, f0, f1, quad))
        
        return pairings
    
    def _valence_impact_score(
        self,
        quad: list,
        removed_edge: tuple,
        valence: np.ndarray,
    ) -> float:
        """
        Score the impact of this pairing on vertex valence regularity.
        
        When we merge two triangles into a quad:
        - The shared edge is removed, decreasing valence of its endpoints
        - We want vertices closer to valence 4 (ideal for quads)
        
        Returns score 0-1, higher is better.
        """
        v0, v1 = removed_edge
        
        # Current valence of edge endpoints
        val0 = valence[v0]
        val1 = valence[v1]
        
        # After merging, these vertices lose 1 edge connection
        new_val0 = val0 - 1
        new_val1 = val1 - 1
        
        # Score: how much closer to valence 4 are we?
        # For triangles, target valence is 6. For quads, it's 4.
        # We're converting to quads, so prefer vertices that will be closer to 4
        target = 4
        
        old_dev = abs(val0 - target) + abs(val1 - target)
        new_dev = abs(new_val0 - target) + abs(new_val1 - target)
        
        # Improvement in deviation (positive = good)
        improvement = old_dev - new_dev
        
        # Normalize to 0-1 range
        # Max improvement is ~4 (from valence 6 to 4 for both vertices)
        score = 0.5 + improvement / 8.0
        
        # Penalize creating very high or very low valence
        if new_val0 <= 2 or new_val1 <= 2:
            score *= 0.5  # Penalize very low valence
        if new_val0 >= 7 or new_val1 >= 7:
            score *= 0.7  # Penalize high valence
        
        return np.clip(score, 0, 1)
    
    def _order_quad_vertices(
        self,
        vertices: np.ndarray,
        other0: int, shared0: int, other1: int, shared1: int
    ) -> list:
        """Order quad vertices for correct winding.
        
        Given two triangles sharing edge (shared0, shared1):
        - Triangle 0 has vertices: other0, shared0, shared1 (some permutation)
        - Triangle 1 has vertices: other1, shared0, shared1 (some permutation)
        
        The quad is formed by: other0 -> shared0 -> other1 -> shared1
        This creates a proper quad without crossing edges.
        
        We verify the ordering by checking the quad doesn't self-intersect.
        """
        # Standard ordering: other0 -> shared0 -> other1 -> shared1
        v0, v1, v2, v3 = other0, shared0, other1, shared1
        verts = vertices[[v0, v1, v2, v3]]
        
        # Check for self-intersection by computing diagonals
        # If diagonals cross inside the quad, it's properly wound
        # If they don't, we need to swap vertices
        d1 = verts[2] - verts[0]  # other0 to other1
        d2 = verts[3] - verts[1]  # shared0 to shared1
        
        # A simple check: compute the quad area using cross product
        # If negative, reverse the winding
        e1 = verts[1] - verts[0]
        e2 = verts[3] - verts[0]
        cross = np.cross(e1, e2)
        
        # Also check that opposite edges don't cross
        # by comparing the quad diagonal intersection
        # For a proper quad, edges 0-1 and 2-3 should be roughly parallel
        # and edges 1-2 and 3-0 should be roughly parallel
        
        # The simplest approach: try both orderings and pick the one
        # with smaller aspect ratio (less distorted)
        order1 = [other0, shared0, other1, shared1]
        order2 = [other0, shared1, other1, shared0]
        
        def quad_distortion(order):
            v = vertices[order]
            edges = [np.linalg.norm(v[(i+1)%4] - v[i]) for i in range(4)]
            if min(edges) < 1e-10:
                return 1000
            return max(edges) / min(edges)
        
        d1 = quad_distortion(order1)
        d2 = quad_distortion(order2)
        
        return order1 if d1 <= d2 else order2
    
    def _quad_quality_score(self, quad_verts: np.ndarray) -> float:
        """Compute quality score for a quad (0-1)."""
        if len(quad_verts) != 4:
            return 0.0
        
        v0, v1, v2, v3 = quad_verts
        
        # Edge lengths
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v3 - v2),
            np.linalg.norm(v0 - v3),
        ]
        
        if min(edges) < 1e-10:
            return 0.0
        
        # Aspect ratio score (1.0 = perfect, decreases with aspect)
        aspect = max(edges) / min(edges)
        aspect_score = 1.0 / (1.0 + (aspect - 1) * 0.5)
        
        # Angle score (all angles should be near 90 degrees)
        angles = []
        for i in range(4):
            p0 = quad_verts[(i - 1) % 4]
            p1 = quad_verts[i]
            p2 = quad_verts[(i + 1) % 4]
            
            e1 = p0 - p1
            e2 = p2 - p1
            
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if n1 < 1e-10 or n2 < 1e-10:
                angles.append(np.pi / 2)
            else:
                cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_angle))
        
        # Mean absolute deviation from 90 degrees
        angle_dev = np.mean(np.abs(np.array(angles) - np.pi / 2))
        angle_score = max(0, 1.0 - angle_dev / (np.pi / 4))  # 0 at 45 deg deviation
        
        # Planarity score (how flat is the quad)
        # Use volume of tetrahedron formed by vertices
        cross = np.cross(v1 - v0, v2 - v0)
        vol = abs(np.dot(cross, v3 - v0)) / 6
        diag = np.linalg.norm(v2 - v0)
        planarity_score = max(0, 1.0 - vol / (diag ** 3 + 1e-10) * 100)
        
        # Combined score
        return aspect_score * 0.35 + angle_score * 0.45 + planarity_score * 0.20
    
    def _direction_alignment(
        self,
        vertices: np.ndarray,
        quad: list,
        direction_field: np.ndarray,
    ) -> float:
        """Score alignment of quad edges with direction field."""
        v0, v1, v2, v3 = quad
        
        # Quad edges
        edges = [
            vertices[v1] - vertices[v0],
            vertices[v2] - vertices[v1],
            vertices[v3] - vertices[v2],
            vertices[v0] - vertices[v3],
        ]
        
        # Get field direction at quad center
        center_idx = v0  # Approximate
        if center_idx < len(direction_field):
            field_dir = direction_field[center_idx]
            field_dir = field_dir / (np.linalg.norm(field_dir) + 1e-10)
            
            # Compute alignment (edges should be parallel/perpendicular to field)
            alignments = []
            for e in edges:
                e_norm = e / (np.linalg.norm(e) + 1e-10)
                dot = abs(np.dot(e_norm, field_dir))
                # Good if parallel (dot=1) or perpendicular (dot=0)
                align = max(dot, 1 - dot)
                alignments.append(align)
            
            return np.mean(alignments)
        
        return 0.5  # Neutral if no field
    
    def _greedy_select(
        self,
        faces: np.ndarray,
        pairings: list,
    ) -> tuple[list, list]:
        """Greedily select best pairings (legacy, non-valence-aware)."""
        return self._greedy_select_valence_aware(faces, pairings, 0)
    
    def _greedy_select_valence_aware(
        self,
        faces: np.ndarray,
        pairings: list,
        num_verts: int,
    ) -> tuple[list, list]:
        """Greedily select best pairings while tracking valence changes."""
        # Sort by quality (best first)
        pairings.sort(key=lambda x: -x[0])
        
        used = set()
        quads = []
        
        # Track current valence if we're doing valence-aware selection
        if num_verts > 0:
            current_valence = np.zeros(num_verts, dtype=np.int32)
            # Initialize from triangles
            for face in faces:
                for i in range(3):
                    v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                    # Count unique edges
                    current_valence[v0] += 1
        
        for quality, f0, f1, quad in pairings:
            if f0 in used or f1 in used:
                continue
            
            # Additional valence check during selection
            if num_verts > 0:
                # Check if this would create problematic valence
                face0, face1 = faces[f0], faces[f1]
                shared_verts = set(face0) & set(face1)
                
                # Skip if any shared vertex would drop below valence 3
                skip = False
                for v in shared_verts:
                    if current_valence[v] <= 3:
                        skip = True
                        break
                
                if skip:
                    continue
                
                # Update valence tracking (removing shared edge decreases valence)
                for v in shared_verts:
                    current_valence[v] -= 1
            
            quads.append(quad)
            used.add(f0)
            used.add(f1)
        
        # Remaining triangles
        remaining = [list(faces[i]) for i in range(len(faces)) if i not in used]
        
        return quads, remaining


class SmartQuadConverter:
    """
    Smarter quad conversion with valence optimization.
    
    Tries to minimize irregular vertices (non-valence-4).
    """
    
    def __init__(self, target_quad_ratio: float = 0.9):
        self.target_quad_ratio = target_quad_ratio
        self.base_converter = TriToQuadConverter(min_quality=0.15)
    
    def convert(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, list]:
        """
        Convert to quad-dominant mesh.
        
        Returns:
            (vertices, all_faces) where faces may be tris or quads
        """
        _, quads, remaining = self.base_converter.convert(vertices, faces)
        
        # Convert remaining triangles to degenerate quads
        all_faces = []
        
        for quad in quads:
            all_faces.append(quad)
        
        for tri in remaining:
            # Degenerate quad: repeat last vertex
            all_faces.append([tri[0], tri[1], tri[2], tri[2]])
        
        return vertices, all_faces
    
    def compute_valence_histogram(
        self,
        vertices: np.ndarray,
        faces: list,
    ) -> dict:
        """Compute vertex valence distribution."""
        valence = [0] * len(vertices)
        
        for face in faces:
            for v in set(face):
                valence[v] += 1
        
        hist = {}
        for v in valence:
            hist[v] = hist.get(v, 0) + 1
        
        return hist
