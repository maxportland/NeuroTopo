"""
Custom guided quad remesher.

A simplified quad remesher that uses guidance fields directly.
This serves as a baseline and can be improved incrementally.
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree

from neurotopo.core.mesh import Mesh
from neurotopo.guidance.composer import GuidanceFields
from neurotopo.remesh.base import Remesher, RemeshResult


class GuidedQuadRemesher(Remesher):
    """
    Custom quad remesher that directly uses guidance fields.
    
    Algorithm:
    1. Sample points on surface using blue noise with size field
    2. Build connectivity using Delaunay + guidance directions
    3. Convert to quads by pairing triangles
    4. Optimize vertex positions
    
    This is a simplified implementation - a full production version
    would use more sophisticated field-aligned methods.
    """
    
    def __init__(
        self,
        optimization_iterations: int = 10,
        use_direction_field: bool = True,
        quad_regularization: float = 0.5,
        preserve_boundary: bool = True,
        **kwargs  # Accept additional kwargs for compatibility
    ):
        self.optimization_iterations = optimization_iterations
        self.use_direction_field = use_direction_field
        self.quad_regularization = quad_regularization
        self.preserve_boundary = preserve_boundary
    
    @property
    def name(self) -> str:
        return "guided_quad"
    
    @property
    def supports_quads(self) -> bool:
        return True
    
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """Perform guided quad remeshing."""
        start_time = time.time()
        
        try:
            # Step 1: Sample points on surface
            sample_positions, sample_normals = self._sample_surface(mesh, guidance)
            
            if len(sample_positions) < 4:
                raise ValueError("Not enough sample points generated")
            
            # Step 2: Build initial triangulation
            tri_faces = self._triangulate_samples(sample_positions, sample_normals, mesh)
            
            # Step 3: Convert to quads
            quad_faces = self._triangles_to_quads(
                sample_positions, tri_faces, guidance, mesh
            )
            
            # Step 4: Optimize positions
            optimized_verts = self._optimize_positions(
                sample_positions, quad_faces, mesh, guidance
            )
            
            output = Mesh(
                vertices=optimized_verts,
                faces=quad_faces,
                name=f"{mesh.name}_guided_quad"
            )
            
            elapsed = time.time() - start_time
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                iterations=self.optimization_iterations,
                time_seconds=elapsed,
                metadata={"backend": "guided_quad", "samples": len(sample_positions)}
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
    
    def _sample_surface(
        self, 
        mesh: Mesh, 
        guidance: GuidanceFields
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample points on mesh surface using size field."""
        if not mesh.is_triangular:
            mesh = mesh.triangulate()
        
        # Compute face areas
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        
        face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = face_areas.sum()
        
        # Determine number of samples from target face count
        if guidance.target_face_count is not None:
            n_samples = int(guidance.target_face_count * 1.2)  # Slightly oversample
        else:
            avg_size = guidance.size_field.mean
            avg_quad_area = avg_size ** 2
            n_samples = int(total_area / avg_quad_area)
        
        n_samples = max(100, min(n_samples, 100000))  # Clamp
        
        # Sample faces proportional to area
        face_probs = face_areas / total_area
        sampled_faces = np.random.choice(
            len(mesh.faces), size=n_samples, p=face_probs
        )
        
        # Generate random barycentric coordinates
        r1 = np.random.random(n_samples)
        r2 = np.random.random(n_samples)
        sqrt_r1 = np.sqrt(r1)
        
        # Barycentric coordinates
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2
        
        # Compute positions
        positions = (
            u[:, np.newaxis] * mesh.vertices[mesh.faces[sampled_faces, 0]] +
            v[:, np.newaxis] * mesh.vertices[mesh.faces[sampled_faces, 1]] +
            w[:, np.newaxis] * mesh.vertices[mesh.faces[sampled_faces, 2]]
        )
        
        # Compute normals
        if mesh.face_normals is None:
            mesh.compute_normals()
        normals = mesh.face_normals[sampled_faces]
        
        # Poisson disk filtering for better distribution
        positions, normals = self._poisson_disk_filter(
            positions, normals, guidance.size_field.mean * 0.8
        )
        
        return positions, normals
    
    def _poisson_disk_filter(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        min_dist: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter points to ensure minimum spacing."""
        if len(positions) == 0:
            return positions, normals
        
        # Build KD-tree
        tree = cKDTree(positions)
        
        # Mark points to keep
        keep = np.ones(len(positions), dtype=bool)
        
        for i in range(len(positions)):
            if not keep[i]:
                continue
            
            # Find neighbors within min_dist
            neighbors = tree.query_ball_point(positions[i], min_dist)
            
            # Remove neighbors (except self)
            for j in neighbors:
                if j != i and keep[j]:
                    keep[j] = False
        
        return positions[keep], normals[keep]
    
    def _triangulate_samples(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        original_mesh: Mesh
    ) -> np.ndarray:
        """Create triangulation of sampled points using proximity-based connectivity."""
        from scipy.spatial import cKDTree, ConvexHull
        
        # Build KD-tree for nearest neighbor queries
        tree = cKDTree(positions)
        
        # Estimate average edge length from size field
        avg_edge = np.mean([np.linalg.norm(positions[i] - positions[j]) 
                          for i in range(min(100, len(positions))) 
                          for j in tree.query(positions[i], k=2)[1][1:2]])
        
        # Try 3D convex hull first (works well for convex shapes like spheres)
        try:
            hull = ConvexHull(positions)
            faces = hull.simplices
            
            # For closed convex shapes, hull gives good initial triangulation
            if len(faces) > len(positions) * 0.8:
                # Filter by edge length to remove long edges
                valid_faces = []
                max_edge = avg_edge * 3.0
                for face in faces:
                    v0, v1, v2 = positions[face]
                    edges = [np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)]
                    if max(edges) < max_edge:
                        valid_faces.append(face)
                
                if len(valid_faces) > len(positions) * 0.3:
                    return np.array(valid_faces)
        except Exception:
            pass
        
        # Fallback: Build triangulation from nearest neighbors
        # This works better for non-convex shapes
        k_neighbors = min(12, len(positions) - 1)
        faces = set()
        
        for i in range(len(positions)):
            # Get k nearest neighbors
            dists, neighbors = tree.query(positions[i], k=k_neighbors + 1)
            neighbors = neighbors[1:]  # Exclude self
            
            # Create triangles with pairs of neighbors
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    n1, n2 = neighbors[j], neighbors[k]
                    
                    # Check if n1 and n2 are also close to each other
                    edge_dist = np.linalg.norm(positions[n1] - positions[n2])
                    if edge_dist < avg_edge * 2.5:
                        # Create triangle (sorted for deduplication)
                        tri = tuple(sorted([i, n1, n2]))
                        faces.add(tri)
        
        faces = [list(f) for f in faces]
        
        # Filter degenerate triangles and check normals
        valid_faces = []
        for face in faces:
            v0, v1, v2 = positions[face[0]], positions[face[1]], positions[face[2]]
            
            # Check area
            normal = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(normal)
            if area < 1e-10:
                continue
            
            # Check that triangle normal aligns with average vertex normal
            face_normal = normal / (2 * area)
            avg_vert_normal = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3
            if np.dot(face_normal, avg_vert_normal) < -0.3:
                # Flip winding
                face = [face[0], face[2], face[1]]
            
            valid_faces.append(face)
        
        return np.array(valid_faces) if valid_faces else np.array([[0, 1, 2]])
    
    def _triangles_to_quads(
        self,
        positions: np.ndarray,
        tri_faces: np.ndarray,
        guidance: GuidanceFields,
        original_mesh: Mesh
    ) -> np.ndarray:
        """Convert triangles to quads by pairing adjacent triangles optimally."""
        # Build edge-to-face mapping
        edge_faces = {}  # (v0, v1) -> [face_idx, ...]
        
        for fi, face in enumerate(tri_faces):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)
        
        # Score all possible pairings and sort by quality
        pairings = []
        for edge, face_list in edge_faces.items():
            if len(face_list) != 2:
                continue
            
            f0, f1 = face_list
            
            # Compute quad and score
            quad = self._merge_triangles(tri_faces[f0], tri_faces[f1], edge)
            if quad is not None:
                score = self._quad_quality(positions[quad])
                pairings.append((score, f0, f1, quad))
        
        # Sort by quality (highest first)
        pairings.sort(key=lambda x: -x[0])
        
        # Greedily select best pairings
        used = set()
        quads = []
        
        for score, f0, f1, quad in pairings:
            if f0 in used or f1 in used:
                continue
            
            if score > 0.2:  # Lower threshold to allow more quads
                quads.append(quad)
                used.add(f0)
                used.add(f1)
        
        # Handle remaining triangles
        remaining_tris = []
        for fi, face in enumerate(tri_faces):
            if fi not in used:
                # Keep as triangle (3 vertices + repeat last)
                remaining_tris.append([face[0], face[1], face[2], face[2]])
        
        all_faces = quads + remaining_tris
        
        if not all_faces:
            return np.array([[0, 1, 2, 2]])
        
        return np.array(all_faces)
    
    def _merge_triangles(
        self,
        tri1: np.ndarray,
        tri2: np.ndarray,
        shared_edge: tuple[int, int]
    ) -> Optional[list[int]]:
        """Merge two triangles sharing an edge into a quad."""
        # Find vertices not on shared edge
        v0, v1 = shared_edge
        
        other1 = [v for v in tri1 if v not in shared_edge][0]
        other2 = [v for v in tri2 if v not in shared_edge][0]
        
        # Order quad vertices: other1 -> v0 -> other2 -> v1
        return [other1, v0, other2, v1]
    
    def _quad_quality(self, vertices: np.ndarray) -> float:
        """
        Compute quad quality metric.
        
        Returns value in [0, 1] where 1 = perfect square.
        """
        if len(vertices) != 4:
            return 0.0
        
        v0, v1, v2, v3 = vertices
        
        # Edge lengths
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v3 - v2),
            np.linalg.norm(v0 - v3),
        ]
        
        if min(edges) < 1e-10:
            return 0.0
        
        # Aspect ratio
        avg_edge = np.mean(edges)
        edge_variance = np.std(edges) / avg_edge
        
        # Corner angles
        angles = []
        for i in range(4):
            p0 = vertices[(i - 1) % 4]
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 4]
            
            e1 = p0 - p1
            e2 = p2 - p1
            
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if n1 < 1e-10 or n2 < 1e-10:
                angles.append(0)
            else:
                cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_angle))
        
        # Ideal angle is 90 degrees
        angle_deviation = np.mean(np.abs(np.array(angles) - np.pi / 2)) / (np.pi / 2)
        
        # Combined quality
        quality = (1 - edge_variance) * 0.5 + (1 - angle_deviation) * 0.5
        
        return max(0.0, min(1.0, quality))
    
    def _optimize_positions(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        original_mesh: Mesh,
        guidance: GuidanceFields
    ) -> np.ndarray:
        """Optimize vertex positions to improve quad quality and surface fidelity."""
        import trimesh
        
        vertices = vertices.copy()
        
        # Build adjacency
        adjacency = [[] for _ in range(len(vertices))]
        for face in faces:
            unique_verts = list(set(face))  # Handle degenerate quads
            for i, vi in enumerate(unique_verts):
                for j, vj in enumerate(unique_verts):
                    if i != j and vj not in adjacency[vi]:
                        adjacency[vi].append(vj)
        
        # Create trimesh for surface projection
        tri_mesh = original_mesh.triangulate() if not original_mesh.is_triangular else original_mesh
        try:
            tm = trimesh.Trimesh(
                vertices=tri_mesh.vertices,
                faces=tri_mesh.faces,
                process=False
            )
            use_trimesh = True
        except Exception:
            use_trimesh = False
            tree = cKDTree(original_mesh.vertices)
        
        for iteration in range(self.optimization_iterations):
            new_vertices = vertices.copy()
            
            # Adaptive smoothing weight - less smoothing as we iterate
            smooth_weight = self.quad_regularization * (1 - iteration / self.optimization_iterations * 0.5)
            
            for vi in range(len(vertices)):
                if not adjacency[vi]:
                    continue
                
                # Laplacian smoothing towards neighbor centroid
                neighbor_center = np.mean(vertices[adjacency[vi]], axis=0)
                smoothed = vertices[vi] * (1 - smooth_weight) + neighbor_center * smooth_weight
                
                # Project back to original surface
                if use_trimesh:
                    # Use trimesh's closest point for accurate projection
                    closest, _, _ = tm.nearest.on_surface([smoothed])
                    projected = closest[0]
                else:
                    # Fallback to nearest vertex
                    _, idx = tree.query(smoothed)
                    projected = original_mesh.vertices[idx]
                
                # Blend smoothed and projected positions
                # More weight to projection as iterations increase
                proj_weight = 0.5 + 0.3 * (iteration / self.optimization_iterations)
                new_vertices[vi] = smoothed * (1 - proj_weight) + projected * proj_weight
            
            vertices = new_vertices
        
        # Final pass: ensure all vertices are on surface
        if use_trimesh:
            closest, _, _ = tm.nearest.on_surface(vertices)
            vertices = closest
        
        return vertices
