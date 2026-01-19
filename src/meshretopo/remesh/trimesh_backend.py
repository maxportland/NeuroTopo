"""
Trimesh-based remeshing backend.

A fallback remesher that uses trimesh when PyMeshLab is not available.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.guidance.composer import GuidanceFields
from meshretopo.remesh.base import Remesher, RemeshResult

logger = logging.getLogger("meshretopo.remesh.trimesh")


class TrimeshRemesher(Remesher):
    """
    Remeshing using trimesh's simplification capabilities.
    
    This is a fallback remesher for when PyMeshLab is not available.
    Uses vertex clustering and edge collapse for simplification.
    """
    
    def __init__(
        self,
        iterations: int = 1,
        preserve_boundary: bool = True,
        **kwargs
    ):
        self.iterations = iterations
        self.preserve_boundary = preserve_boundary
    
    @property
    def name(self) -> str:
        return "trimesh"
    
    @property
    def supports_quads(self) -> bool:
        return False
    
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """Remesh using trimesh simplification."""
        import trimesh
        
        start_time = time.time()
        
        try:
            # Convert to trimesh
            if mesh.is_quad:
                mesh = mesh.triangulate()
            
            tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            
            # Determine target face count
            if guidance.target_face_count is not None:
                target_faces = guidance.target_face_count
            else:
                # Default to 10% of original
                target_faces = max(100, mesh.num_faces // 10)
            
            # Use trimesh's simplify_quadric_decimation if available
            try:
                # This requires the optional dependency 'trimesh[easy]' with fast_simplification
                simplified = tm.simplify_quadric_decimation(target_faces)
            except Exception:
                # Fallback to vertex clustering
                # Calculate appropriate cell size based on target faces
                # Roughly: faces ≈ vertices/2, vertices ≈ (bbox_volume/cell_size^3)
                bbox_volume = np.prod(tm.bounding_box.extents)
                estimated_cell_size = (bbox_volume / (target_faces * 2)) ** (1/3)
                
                # Use subdivide/merge approach
                simplified = self._simplify_by_clustering(tm, target_faces)
            
            output = Mesh(
                vertices=np.asarray(simplified.vertices),
                faces=np.asarray(simplified.faces),
                name=f"{mesh.name}_remeshed"
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Trimesh remesh: {mesh.num_faces} -> {output.num_faces} faces in {elapsed:.2f}s")
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                iterations=self.iterations,
                time_seconds=elapsed,
                metadata={"backend": "trimesh"}
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
    
    def _simplify_by_clustering(self, tm, target_faces: int):
        """
        Simplify mesh by vertex clustering.
        
        Groups nearby vertices together and recreates faces.
        """
        import trimesh
        
        # Current face count
        current_faces = len(tm.faces)
        
        if current_faces <= target_faces:
            return tm
        
        # Calculate reduction ratio
        ratio = target_faces / current_faces
        
        # Estimate cluster size from ratio
        # More faces = smaller clusters needed
        bbox_diagonal = np.linalg.norm(tm.bounding_box.extents)
        
        # Start with a cell size estimate
        cell_size = bbox_diagonal * (1 - ratio) * 0.1
        cell_size = max(cell_size, bbox_diagonal * 0.01)  # Minimum cell size
        
        # Iteratively adjust cell size to hit target
        for _ in range(10):
            # Create vertex grid
            grid_indices = np.floor(tm.vertices / cell_size).astype(int)
            
            # Find unique grid cells
            unique_cells, inverse_indices = np.unique(
                grid_indices, axis=0, return_inverse=True
            )
            
            # Average vertices in each cell
            new_vertices = np.zeros((len(unique_cells), 3))
            cell_counts = np.zeros(len(unique_cells))
            
            for vi, cell_idx in enumerate(inverse_indices):
                new_vertices[cell_idx] += tm.vertices[vi]
                cell_counts[cell_idx] += 1
            
            new_vertices /= cell_counts[:, np.newaxis]
            
            # Remap faces
            new_faces = inverse_indices[tm.faces]
            
            # Remove degenerate faces (where all vertices map to same cell)
            valid_faces = []
            for face in new_faces:
                if len(set(face)) == 3:  # All three vertices are different
                    valid_faces.append(face)
            
            if len(valid_faces) == 0:
                # Too aggressive, increase cell size and try again
                cell_size *= 0.5
                continue
            
            estimated_faces = len(valid_faces)
            
            if abs(estimated_faces - target_faces) < target_faces * 0.2:
                # Close enough
                break
            elif estimated_faces > target_faces:
                # Need bigger cells (more reduction)
                cell_size *= 1.2
            else:
                # Need smaller cells (less reduction)
                cell_size *= 0.8
        
        if not valid_faces:
            # Couldn't simplify, return original
            return tm
        
        # Create new mesh
        new_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=np.array(valid_faces)
        )
        
        # Clean up (remove unreferenced vertices, merge close vertices)
        new_mesh.remove_unreferenced_vertices()
        new_mesh.merge_vertices()
        
        return new_mesh


def create_fallback_remesher(**kwargs) -> Remesher:
    """Create the best available remesher."""
    try:
        # Try to import PyMeshLab
        import pymeshlab
        from meshretopo.remesh.pymeshlab_backend import PyMeshLabRemesher
        return PyMeshLabRemesher(**kwargs)
    except ImportError:
        # Fall back to trimesh
        return TrimeshRemesher(**kwargs)
