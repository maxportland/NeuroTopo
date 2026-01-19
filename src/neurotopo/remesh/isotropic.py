"""
Isotropic remeshing backend.

Implements incremental isotropic remeshing with:
- Edge splits (subdivide long edges)
- Edge collapses (merge short edges)
- Edge flips (improve triangle quality)
- Tangential smoothing (regularize vertex positions)

This produces high-quality triangular meshes that can then
be converted to quads.
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree

from neurotopo.core.mesh import Mesh
from neurotopo.guidance.composer import GuidanceFields
from neurotopo.remesh.base import Remesher, RemeshResult


class IsotropicRemesher(Remesher):
    """
    Isotropic remeshing via incremental mesh operations.
    
    Based on the algorithm from:
    "A Remeshing Approach to Multiresolution Modeling" (Botsch & Kobbelt)
    """
    
    def __init__(
        self,
        iterations: int = 5,
        target_edge_length: Optional[float] = None,
        adaptive: bool = True,
        smoothing_weight: float = 0.5,
        preserve_boundary: bool = True,
        **kwargs
    ):
        self.iterations = iterations
        self.target_edge_length = target_edge_length
        self.adaptive = adaptive
        self.smoothing_weight = smoothing_weight
        self.preserve_boundary = preserve_boundary
    
    @property
    def name(self) -> str:
        return "isotropic"
    
    @property
    def supports_quads(self) -> bool:
        return False
    
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """Perform isotropic remeshing."""
        start_time = time.time()
        
        try:
            # Ensure triangular mesh
            if not mesh.is_triangular:
                mesh = mesh.triangulate()
            
            # Compute target edge length from face count
            if self.target_edge_length is not None:
                target_len = self.target_edge_length
            elif guidance.target_face_count is not None:
                # Estimate edge length from target face count
                # For triangles: F ≈ 2V, E ≈ 3V, so avg_edge² ≈ 2*area/F
                total_area = self._compute_total_area(mesh)
                target_len = np.sqrt(2 * total_area / guidance.target_face_count) * 1.1
            else:
                target_len = guidance.size_field.mean
            
            # Build mutable mesh structure
            verts, faces = mesh.vertices.copy(), mesh.faces.copy()
            
            # Build original surface for projection
            import trimesh
            original_tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False
            )
            
            # Iterative remeshing
            for iteration in range(self.iterations):
                # Skip if too few faces
                if len(faces) < 10:
                    break
                    
                # Get local target lengths if adaptive
                if self.adaptive and guidance.size_field is not None:
                    local_targets = self._get_local_targets(verts, guidance, target_len)
                else:
                    local_targets = np.full(len(verts), target_len)
                
                # Step 1: Split long edges
                verts, faces, local_targets = self._split_long_edges(
                    verts, faces, local_targets, factor=4/3
                )
                
                if len(faces) < 10:
                    break
                
                # Step 2: Collapse short edges (limit aggressiveness)
                verts, faces, local_targets = self._collapse_short_edges(
                    verts, faces, local_targets, factor=4/5
                )
                
                if len(faces) < 10:
                    break
                
                # Step 3: Flip edges to improve valence
                faces = self._flip_edges_for_valence(verts, faces)
                
                # Step 4: Tangential smoothing
                verts = self._tangential_smoothing(
                    verts, faces, original_tm, self.smoothing_weight
                )
            
            # Clean up mesh
            verts, faces = self._remove_degenerate(verts, faces)
            
            output = Mesh(
                vertices=verts,
                faces=faces,
                name=f"{mesh.name}_isotropic"
            )
            
            elapsed = time.time() - start_time
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                iterations=self.iterations,
                time_seconds=elapsed,
                metadata={"backend": "isotropic", "target_edge": target_len}
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return RemeshResult(
                mesh=mesh,
                success=False,
                actual_face_count=mesh.num_faces,
                time_seconds=elapsed,
                metadata={"error": str(e)}
            )
    
    def _compute_total_area(self, mesh: Mesh) -> float:
        """Compute total surface area."""
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        return areas.sum()
    
    def _get_local_targets(
        self,
        verts: np.ndarray,
        guidance: GuidanceFields,
        base_target: float
    ) -> np.ndarray:
        """Get per-vertex target edge lengths from guidance."""
        if guidance.size_field is None:
            return np.full(len(verts), base_target)
        
        # Interpolate size field to vertices
        # For now, use KD-tree to find nearest guidance vertex
        if hasattr(guidance.size_field, 'values'):
            # Scale size field values
            sf_values = guidance.size_field.values
            sf_mean = sf_values.mean()
            if sf_mean > 0:
                targets = (sf_values / sf_mean) * base_target
                return np.clip(targets, base_target * 0.3, base_target * 3.0)
        
        return np.full(len(verts), base_target)
    
    def _build_edge_map(self, faces: np.ndarray) -> dict:
        """Build edge to face mapping."""
        edge_faces = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)
        return edge_faces
    
    def _build_vertex_faces(self, verts: np.ndarray, faces: np.ndarray) -> list:
        """Build vertex to face adjacency."""
        vert_faces = [[] for _ in range(len(verts))]
        for fi, face in enumerate(faces):
            for vi in face:
                vert_faces[vi].append(fi)
        return vert_faces
    
    def _split_long_edges(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        targets: np.ndarray,
        factor: float = 4/3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split edges longer than factor * target_length."""
        verts = list(verts)
        faces = list(faces)
        targets = list(targets)
        
        # Find edges to split
        edge_faces = self._build_edge_map(np.array(faces))
        
        edges_to_split = []
        for (v0, v1), face_list in edge_faces.items():
            edge_len = np.linalg.norm(np.array(verts[v1]) - np.array(verts[v0]))
            target = (targets[v0] + targets[v1]) / 2
            
            if edge_len > factor * target:
                edges_to_split.append((v0, v1, face_list))
        
        # Split edges (limit per iteration to avoid explosion)
        max_splits = len(faces) // 2
        for v0, v1, face_list in edges_to_split[:max_splits]:
            if len(face_list) not in [1, 2]:
                continue
            
            # Create new vertex at midpoint
            new_v = (np.array(verts[v0]) + np.array(verts[v1])) / 2
            new_vi = len(verts)
            verts.append(new_v)
            targets.append((targets[v0] + targets[v1]) / 2)
            
            # Split faces containing this edge
            new_faces = []
            faces_to_remove = set(face_list)
            
            for fi in face_list:
                face = faces[fi]
                # Find the third vertex
                other = [v for v in face if v != v0 and v != v1][0]
                
                # Create two new triangles
                new_faces.append([v0, new_vi, other])
                new_faces.append([new_vi, v1, other])
            
            # Update faces list
            faces = [f for i, f in enumerate(faces) if i not in faces_to_remove]
            faces.extend(new_faces)
        
        return np.array(verts), np.array(faces), np.array(targets)
    
    def _collapse_short_edges(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        targets: np.ndarray,
        factor: float = 4/5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collapse edges shorter than factor * target_length."""
        verts = verts.copy()
        faces = list(faces)
        targets = targets.copy()
        
        # Find edges to collapse
        edge_faces = self._build_edge_map(np.array(faces))
        
        edges_to_collapse = []
        for (v0, v1), face_list in edge_faces.items():
            edge_len = np.linalg.norm(verts[v1] - verts[v0])
            target = (targets[v0] + targets[v1]) / 2
            
            if edge_len < factor * target and len(face_list) == 2:
                edges_to_collapse.append((edge_len, v0, v1, face_list))
        
        # Sort by length (shortest first)
        edges_to_collapse.sort(key=lambda x: x[0])
        
        # Track collapsed vertices
        vertex_map = list(range(len(verts)))
        collapsed = set()
        
        max_collapses = len(faces) // 4
        collapse_count = 0
        
        for _, v0, v1, face_list in edges_to_collapse:
            if collapse_count >= max_collapses:
                break
            
            # Skip if either vertex already collapsed
            if v0 in collapsed or v1 in collapsed:
                continue
            
            # Move v0 to midpoint and collapse v1 into v0
            verts[v0] = (verts[v0] + verts[v1]) / 2
            targets[v0] = (targets[v0] + targets[v1]) / 2
            vertex_map[v1] = v0
            collapsed.add(v1)
            collapse_count += 1
        
        # Remap faces
        new_faces = []
        for face in faces:
            remapped = [vertex_map[v] for v in face]
            # Skip degenerate faces
            if len(set(remapped)) == 3:
                new_faces.append(remapped)
        
        # Compact vertex array
        used_verts = set()
        for face in new_faces:
            used_verts.update(face)
        
        old_to_new = {old: new for new, old in enumerate(sorted(used_verts))}
        new_verts = verts[sorted(used_verts)]
        new_targets = targets[sorted(used_verts)]
        new_faces = [[old_to_new[v] for v in face] for face in new_faces]
        
        return new_verts, np.array(new_faces), new_targets
    
    def _flip_edges_for_valence(
        self,
        verts: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """Flip edges to improve vertex valence towards 6."""
        faces = list(faces)
        
        # Compute current valences
        valence = np.zeros(len(verts), dtype=int)
        for face in faces:
            for v in face:
                valence[v] += 1
        
        # Build edge map
        edge_faces = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append((fi, i))
        
        # Try flipping interior edges
        for (v0, v1), face_info in edge_faces.items():
            if len(face_info) != 2:
                continue
            
            (fi0, ei0), (fi1, ei1) = face_info
            
            # Get opposite vertices
            face0 = faces[fi0]
            face1 = faces[fi1]
            opp0 = face0[(ei0 + 2) % 3]
            opp1 = face1[(ei1 + 2) % 3]
            
            # Current valence deviation
            current_dev = (
                abs(valence[v0] - 6) + abs(valence[v1] - 6) +
                abs(valence[opp0] - 6) + abs(valence[opp1] - 6)
            )
            
            # After flip: v0, v1 lose 1, opp0, opp1 gain 1
            new_dev = (
                abs(valence[v0] - 1 - 6) + abs(valence[v1] - 1 - 6) +
                abs(valence[opp0] + 1 - 6) + abs(valence[opp1] + 1 - 6)
            )
            
            # Flip if it improves valence
            if new_dev < current_dev:
                # Check geometric validity (avoid flipping to non-convex)
                # Simple check: new triangles should have same orientation
                n0 = np.cross(
                    verts[opp1] - verts[v0],
                    verts[opp0] - verts[v0]
                )
                n1 = np.cross(
                    verts[opp0] - verts[v1],
                    verts[opp1] - verts[v1]
                )
                
                if np.dot(n0, n1) > 0:
                    # Perform flip
                    faces[fi0] = [v0, opp1, opp0]
                    faces[fi1] = [v1, opp0, opp1]
                    
                    # Update valences
                    valence[v0] -= 1
                    valence[v1] -= 1
                    valence[opp0] += 1
                    valence[opp1] += 1
        
        return np.array(faces)
    
    def _tangential_smoothing(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        original_mesh,
        weight: float
    ) -> np.ndarray:
        """Smooth vertices tangentially while staying on surface."""
        verts = verts.copy()
        
        # Build adjacency
        neighbors = [set() for _ in range(len(verts))]
        for face in faces:
            for i in range(3):
                neighbors[face[i]].add(face[(i + 1) % 3])
                neighbors[face[i]].add(face[(i + 2) % 3])
        
        # Smooth
        new_verts = verts.copy()
        for vi in range(len(verts)):
            if not neighbors[vi]:
                continue
            
            # Compute centroid of neighbors
            neighbor_list = list(neighbors[vi])
            centroid = verts[neighbor_list].mean(axis=0)
            
            # Move towards centroid
            smoothed = verts[vi] * (1 - weight) + centroid * weight
            
            # Project to original surface
            try:
                closest, _, _ = original_mesh.nearest.on_surface([smoothed])
                new_verts[vi] = closest[0]
            except Exception:
                new_verts[vi] = smoothed
        
        return new_verts
    
    def _remove_degenerate(
        self,
        verts: np.ndarray,
        faces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove degenerate faces and unused vertices."""
        # Remove degenerate faces
        valid_faces = []
        for face in faces:
            if len(set(face)) == 3:
                v0, v1, v2 = verts[face]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                if area > 1e-10:
                    valid_faces.append(face)
        
        if not valid_faces:
            return verts, faces
        
        faces = np.array(valid_faces)
        
        # Remove unused vertices
        used = np.unique(faces.flatten())
        old_to_new = {old: new for new, old in enumerate(used)}
        
        new_verts = verts[used]
        new_faces = np.array([[old_to_new[v] for v in face] for face in faces])
        
        return new_verts, new_faces
