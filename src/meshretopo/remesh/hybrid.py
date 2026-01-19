"""
Hybrid remeshing backend.

Combines high-quality triangle generation with intelligent
quad conversion for production-quality output.
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.guidance.composer import GuidanceFields
from meshretopo.remesh.base import Remesher, RemeshResult
from meshretopo.remesh.tri_to_quad import TriToQuadConverter, SmartQuadConverter


class HybridRemesher(Remesher):
    """
    Hybrid remesher that produces high-quality quads.
    
    Strategy:
    1. Use trimesh for initial decimation (good fidelity)
    2. Convert triangles to quads using optimal pairing
    3. Optimize quad shapes through smoothing
    """
    
    def __init__(
        self,
        quad_ratio: float = 0.8,  # Target ratio of quads vs tris
        optimization_passes: int = 5,
        preserve_boundary: bool = True,
        **kwargs
    ):
        self.quad_ratio = quad_ratio
        self.optimization_passes = optimization_passes
        self.preserve_boundary = preserve_boundary
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    @property
    def supports_quads(self) -> bool:
        return True
    
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """Perform hybrid remeshing."""
        start_time = time.time()
        
        try:
            import trimesh
            
            # Step 1: Initial decimation with trimesh
            target_faces = guidance.target_face_count or mesh.num_faces // 4
            # Request 2x triangles since we'll pair them into quads
            target_tris = int(target_faces * 2)
            
            tri_mesh = mesh.triangulate() if not mesh.is_triangular else mesh
            
            tm = trimesh.Trimesh(
                vertices=tri_mesh.vertices,
                faces=tri_mesh.faces,
                process=False
            )
            
            # Decimate using available method
            try:
                simplified = tm.simplify_quadric_decimation(target_tris)
            except Exception:
                # Fallback to scipy-based decimation
                from scipy.spatial import cKDTree
                
                # Simple vertex clustering decimation
                verts = tri_mesh.vertices
                cluster_size = np.sqrt(tm.area / target_tris) * 1.5
                
                # Cluster vertices
                tree = cKDTree(verts)
                clusters = tree.query_ball_tree(tree, cluster_size)
                
                # Find cluster representatives
                used = set()
                new_verts = []
                vert_map = {}
                
                for i, cluster in enumerate(clusters):
                    if i in used:
                        continue
                    # Use centroid of cluster
                    cluster_verts = verts[cluster]
                    centroid = cluster_verts.mean(axis=0)
                    new_idx = len(new_verts)
                    new_verts.append(centroid)
                    for j in cluster:
                        vert_map[j] = new_idx
                        used.add(j)
                
                # Remap faces
                new_faces = []
                for face in tri_mesh.faces:
                    new_face = [vert_map[v] for v in face]
                    if len(set(new_face)) == 3:
                        new_faces.append(new_face)
                
                simplified = trimesh.Trimesh(
                    vertices=np.array(new_verts),
                    faces=np.array(new_faces) if new_faces else np.array([[0, 1, 2]]),
                    process=True
                )
            
            verts = np.array(simplified.vertices)
            faces = np.array(simplified.faces)
            
            # Step 2: Convert triangles to quads using improved converter
            converter = TriToQuadConverter(min_quality=0.2, prefer_regular=True)
            verts, quad_faces, remaining_tris = converter.convert(verts, faces)
            
            # Build output - quads and remaining triangles as degenerate quads
            all_faces = []
            
            for qf in quad_faces:
                all_faces.append(qf)  # Quad as 4 vertices
            
            for tf in remaining_tris:
                # Degenerate quad: repeat last vertex
                all_faces.append([tf[0], tf[1], tf[2], tf[2]])
            
            faces = np.array(all_faces)
            
            # Step 3: Optimize quad shapes using enhanced optimizer
            from meshretopo.postprocess import QuadOptimizer
            optimizer = QuadOptimizer(
                iterations=self.optimization_passes * 2,
                smoothing_weight=0.4,
                surface_weight=0.8,
            )
            verts = optimizer.optimize(verts, faces, tm)
            
            output = Mesh(
                vertices=verts,
                faces=faces,
                name=f"{mesh.name}_hybrid"
            )
            
            elapsed = time.time() - start_time
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                iterations=self.optimization_passes,
                time_seconds=elapsed,
                metadata={
                    "backend": "hybrid",
                    "quad_count": len(quad_faces),
                    "tri_count": len(remaining_tris)
                }
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
    
    def _triangles_to_quads(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        target_ratio: float
    ) -> tuple[list, list]:
        """Convert triangles to quads by optimal pairing."""
        # Build edge-to-face mapping
        edge_faces = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)
        
        # Score all possible triangle pairings
        pairings = []
        for edge, face_list in edge_faces.items():
            if len(face_list) != 2:
                continue
            
            f0, f1 = face_list
            
            # Get quad vertices
            face0, face1 = faces[f0], faces[f1]
            other0 = [v for v in face0 if v not in edge][0]
            other1 = [v for v in face1 if v not in edge][0]
            
            # Order: other0 -> edge[0] -> other1 -> edge[1]
            quad = [other0, edge[0], other1, edge[1]]
            
            # Score based on quad quality
            score = self._quad_quality(verts[quad])
            pairings.append((score, f0, f1, quad))
        
        # Sort by quality (best first)
        pairings.sort(key=lambda x: -x[0])
        
        # Greedily select pairings up to target ratio
        max_quads = int(len(faces) * target_ratio / 2)
        used = set()
        quads = []
        
        for score, f0, f1, quad in pairings:
            if len(quads) >= max_quads:
                break
            if f0 in used or f1 in used:
                continue
            if score < 0.15:  # Minimum quality
                continue
            
            quads.append(quad)
            used.add(f0)
            used.add(f1)
        
        # Remaining triangles
        remaining = [list(faces[i]) for i in range(len(faces)) if i not in used]
        
        return quads, remaining
    
    def _quad_quality(self, vertices: np.ndarray) -> float:
        """Compute quad quality (0-1, 1=perfect square)."""
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
        
        # Aspect ratio penalty
        avg_edge = np.mean(edges)
        edge_variance = np.std(edges) / avg_edge
        
        # Corner angle penalty
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
        
        angle_deviation = np.mean(np.abs(np.array(angles) - np.pi / 2)) / (np.pi / 2)
        
        # Combined quality
        quality = (1 - edge_variance) * 0.5 + (1 - angle_deviation) * 0.5
        return max(0.0, min(1.0, quality))
    
    def _optimize_quads(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        original_tm
    ) -> np.ndarray:
        """Optimize vertex positions for better quad quality."""
        verts = verts.copy()
        
        # Build adjacency (handle degenerate quads)
        adjacency = [set() for _ in range(len(verts))]
        for face in faces:
            # Get unique vertices in face
            unique = list(set(face))
            for i, vi in enumerate(unique):
                for j, vj in enumerate(unique):
                    if i != j:
                        adjacency[vi].add(vj)
        
        for iteration in range(self.optimization_passes):
            new_verts = verts.copy()
            weight = 0.3 * (1 - iteration / self.optimization_passes)
            
            for vi in range(len(verts)):
                if not adjacency[vi]:
                    continue
                
                # Laplacian smoothing
                neighbors = list(adjacency[vi])
                centroid = verts[neighbors].mean(axis=0)
                smoothed = verts[vi] * (1 - weight) + centroid * weight
                
                # Project to surface
                try:
                    closest, _, _ = original_tm.nearest.on_surface([smoothed])
                    new_verts[vi] = closest[0]
                except Exception:
                    new_verts[vi] = smoothed
            
            verts = new_verts
        
        return verts
