"""
Hybrid remeshing backend.

Combines high-quality triangle generation with intelligent
quad conversion for production-quality output.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.guidance.composer import GuidanceFields
from neurotopo.remesh.base import Remesher, RemeshResult
from neurotopo.remesh.tri_to_quad import TriToQuadConverter, SmartQuadConverter

logger = logging.getLogger("neurotopo.remesh.hybrid")


class HybridRemesher(Remesher):
    """
    Hybrid remesher that produces high-quality quads.
    
    Strategy:
    1. Use fast decimation (fast_simplification or trimesh)
    2. Convert triangles to quads using optimal pairing
    3. Light optimization for quad shapes
    """
    
    def __init__(
        self,
        quad_ratio: float = 0.8,  # Target ratio of quads vs tris
        optimization_passes: int = 3,  # Reduced for speed
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
            # Step 1: Fast decimation
            target_faces = guidance.target_face_count or mesh.num_faces // 4
            # Request 2x triangles since we'll pair them into quads
            target_tris = int(target_faces * 2)
            
            tri_mesh = mesh.triangulate() if not mesh.is_triangular else mesh
            verts, faces = self._fast_decimate(tri_mesh, target_tris)
            
            # Step 1.5: Improve triangle quality with edge relaxation
            try:
                verts, faces = self._improve_triangles(verts, faces, iterations=20)
            except Exception as e:
                logger.debug(f"Triangle improvement failed: {e}")
            
            # Step 2: Convert triangles to quads
            converter = TriToQuadConverter(min_quality=0.35, prefer_regular=True)
            verts, quad_faces, remaining_tris = converter.convert(verts, faces)
            
            # Build output - store quads and triangles separately
            # Use object array to allow mixed face sizes
            all_faces = []
            
            for qf in quad_faces:
                all_faces.append(np.array(qf, dtype=np.int32))  # Quad as 4 vertices
            
            for tf in remaining_tris:
                all_faces.append(np.array(tf, dtype=np.int32))  # Triangle as 3 vertices
            
            # For now, still need to convert to homogeneous array for Mesh class
            # Convert triangles to degenerate quads for compatibility
            homo_faces = []
            for face in all_faces:
                if len(face) == 4:
                    homo_faces.append(face)
                elif len(face) == 3:
                    # Store as degenerate quad for array compatibility
                    homo_faces.append([face[0], face[1], face[2], face[2]])
            
            faces = np.array(homo_faces, dtype=np.int32)
            
            # Step 3: Light optimization (only if small enough)
            # Skip for large meshes - the optimizer is too slow
            if len(faces) < 3000 and self.optimization_passes > 0:
                try:
                    import trimesh
                    original_tm = trimesh.Trimesh(
                        vertices=tri_mesh.vertices,
                        faces=tri_mesh.faces,
                        process=False
                    )
                    
                    from neurotopo.postprocess import QuadOptimizer
                    optimizer = QuadOptimizer(
                        iterations=min(self.optimization_passes, 2),
                        smoothing_weight=0.3,
                        surface_weight=0.7,
                    )
                    verts = optimizer.optimize(verts, faces, original_tm)
                except Exception as opt_err:
                    logger.debug(f"Optimization skipped: {opt_err}")
            
            # Final manifold repair after all processing
            from neurotopo.postprocess.manifold import ManifoldRepair
            repair = ManifoldRepair(verbose=False)
            verts, faces = repair.repair(verts, faces)
            
            output = Mesh(
                vertices=verts,
                faces=faces,
                name=f"{mesh.name}_hybrid"
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Hybrid remesh: {mesh.num_faces} -> {output.num_faces} faces "
                       f"({len(quad_faces)} quads, {len(remaining_tris)} tris) in {elapsed:.2f}s")
            
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
            logger.error(f"Hybrid remesh failed after {elapsed:.2f}s: {e}")
            return RemeshResult(
                mesh=mesh,
                success=False,
                actual_face_count=mesh.num_faces,
                time_seconds=elapsed,
                metadata={"error": str(e)}
            )
    
    def _fast_decimate(
        self,
        mesh: Mesh,
        target_faces: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fast mesh decimation using best available method."""
        # Try fast_simplification first (much faster)
        try:
            import fast_simplification
            
            target_ratio = min(0.99, max(0.01, target_faces / mesh.num_faces))
            verts, faces = fast_simplification.simplify(
                mesh.vertices.astype(np.float32),
                mesh.faces.astype(np.int32),
                target_reduction=1 - target_ratio
            )
            return verts, faces
        except ImportError:
            pass
        
        # Fallback to trimesh
        try:
            import trimesh
            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False
            )
            
            # Try quadric decimation
            try:
                simplified = tm.simplify_quadric_decimation(target_faces)
                return np.array(simplified.vertices), np.array(simplified.faces)
            except Exception:
                pass
            
            # Fallback: vertex clustering
            from scipy.spatial import cKDTree
            
            cluster_size = np.sqrt(tm.area / target_faces) * 1.5
            tree = cKDTree(mesh.vertices)
            clusters = tree.query_ball_tree(tree, cluster_size)
            
            used = set()
            new_verts = []
            vert_map = {}
            
            for i, cluster in enumerate(clusters):
                if i in used:
                    continue
                cluster_verts = mesh.vertices[cluster]
                centroid = cluster_verts.mean(axis=0)
                new_idx = len(new_verts)
                new_verts.append(centroid)
                for j in cluster:
                    vert_map[j] = new_idx
                    used.add(j)
            
            new_faces = []
            for face in mesh.faces:
                new_face = [vert_map.get(v, 0) for v in face]
                if len(set(new_face)) == 3:
                    new_faces.append(new_face)
            
            return np.array(new_verts), np.array(new_faces)
            
        except Exception as e:
            logger.warning(f"Decimation failed: {e}, returning original")
            return mesh.vertices.copy(), mesh.faces.copy()
    
    def _improve_triangles(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        iterations: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Improve triangle quality with Laplacian smoothing.
        
        Makes triangles more equilateral, which leads to better quads when paired.
        """
        verts = vertices.copy()
        faces = np.array(faces)
        
        # Build adjacency
        from collections import defaultdict
        
        neighbors = defaultdict(set)
        edge_count = defaultdict(int)
        
        for face in faces:
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i+1) % 3])
                neighbors[v0].add(v1)
                neighbors[v1].add(v0)
                e = tuple(sorted([v0, v1]))
                edge_count[e] += 1
        
        # Identify boundary vertices
        boundary_verts = set()
        for e, count in edge_count.items():
            if count == 1:
                boundary_verts.add(e[0])
                boundary_verts.add(e[1])
        
        # Laplacian smoothing
        for _ in range(iterations):
            new_verts = verts.copy()
            for vi in range(len(verts)):
                if vi in boundary_verts:
                    continue
                if len(neighbors[vi]) == 0:
                    continue
                    
                # Uniform Laplacian: average of neighbors
                neighbor_pos = verts[list(neighbors[vi])]
                centroid = neighbor_pos.mean(axis=0)
                # Blend: move 30% toward centroid
                new_verts[vi] = verts[vi] * 0.7 + centroid * 0.3
            
            verts = new_verts
        
        return verts, faces
