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

from meshretopo.core.mesh import Mesh
from meshretopo.guidance.composer import GuidanceFields
from meshretopo.remesh.base import Remesher, RemeshResult
from meshretopo.remesh.tri_to_quad import TriToQuadConverter, SmartQuadConverter

logger = logging.getLogger("meshretopo.remesh.hybrid")


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
            
            # Step 2: Convert triangles to quads
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
            
            # Step 3: Light optimization (only if small enough)
            if len(faces) < 20000 and self.optimization_passes > 0:
                try:
                    import trimesh
                    original_tm = trimesh.Trimesh(
                        vertices=tri_mesh.vertices,
                        faces=tri_mesh.faces,
                        process=False
                    )
                    
                    from meshretopo.postprocess import QuadOptimizer
                    optimizer = QuadOptimizer(
                        iterations=self.optimization_passes,
                        smoothing_weight=0.3,
                        surface_weight=0.7,
                    )
                    verts = optimizer.optimize(verts, faces, original_tm)
                except Exception as opt_err:
                    logger.debug(f"Optimization skipped: {opt_err}")
            
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
