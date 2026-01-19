"""
PyMeshLab remeshing backend.

Uses PyMeshLab's remeshing filters with guidance field support.
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.guidance.composer import GuidanceFields
from neurotopo.remesh.base import Remesher, RemeshResult


class PyMeshLabRemesher(Remesher):
    """
    Remeshing using PyMeshLab.
    
    Supports isotropic remeshing with adaptive sizing.
    Primary output is triangles, but can post-process to quads.
    """
    
    def __init__(
        self,
        iterations: int = 3,
        adaptive: bool = True,
        preserve_boundary: bool = True,
        smooth_iterations: int = 2,
    ):
        self.iterations = iterations
        self.adaptive = adaptive
        self.preserve_boundary = preserve_boundary
        self.smooth_iterations = smooth_iterations
    
    @property
    def name(self) -> str:
        return "pymeshlab"
    
    @property
    def supports_quads(self) -> bool:
        return False  # PyMeshLab primarily does triangle remeshing
    
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """Remesh using PyMeshLab."""
        import pymeshlab
        
        start_time = time.time()
        
        # Create MeshSet
        ms = pymeshlab.MeshSet()
        
        # Convert our mesh to PyMeshLab format
        if mesh.is_quad:
            mesh = mesh.triangulate()
        
        m = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces
        )
        ms.add_mesh(m)
        
        # Compute target edge length from guidance
        avg_size = guidance.size_field.mean
        target_length = avg_size
        
        # Estimate target percentage based on face count
        if guidance.target_face_count is not None:
            # Approximate: target percentage = target_faces / current_faces
            target_perc = guidance.target_face_count / mesh.num_faces
            target_perc = max(0.1, min(10.0, target_perc))  # Clamp to reasonable range
        else:
            target_perc = 1.0
        
        try:
            # Run isotropic remeshing
            ms.meshing_isotropic_explicit_remeshing(
                iterations=self.iterations,
                targetlen=pymeshlab.PercentageValue(target_perc * 100 / mesh.num_faces * 5000),
                adaptive=self.adaptive,
                selectedonly=False,
                checksurfdist=True,
                maxsurfdist=pymeshlab.PercentageValue(0.5),
                splitflag=True,
                collapseflag=True,
                swapflag=True,
                smoothflag=True,
                reprojectflag=True
            )
            
            # Optional smoothing pass
            if self.smooth_iterations > 0:
                ms.apply_coord_laplacian_smoothing(
                    stepsmoothnum=self.smooth_iterations,
                    boundary=not self.preserve_boundary
                )
            
            # Extract result
            result_mesh = ms.current_mesh()
            
            output = Mesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                name=f"{mesh.name}_remeshed"
            )
            
            elapsed = time.time() - start_time
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                iterations=self.iterations,
                time_seconds=elapsed,
                metadata={"backend": "pymeshlab", "method": "isotropic_explicit"}
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return RemeshResult(
                mesh=mesh,  # Return original on failure
                success=False,
                actual_face_count=mesh.num_faces,
                time_seconds=elapsed,
                metadata={"error": str(e)}
            )


class PyMeshLabQuadRemesher(Remesher):
    """
    Quad remeshing via PyMeshLab.
    
    Uses triangle remeshing followed by quad conversion.
    """
    
    def __init__(
        self,
        triangle_iterations: int = 3,
        quad_dominant: bool = True,
    ):
        self.triangle_iterations = triangle_iterations
        self.quad_dominant = quad_dominant
        self.tri_remesher = PyMeshLabRemesher(iterations=triangle_iterations)
    
    @property
    def name(self) -> str:
        return "pymeshlab_quad"
    
    @property
    def supports_quads(self) -> bool:
        return True
    
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """Remesh to quads via PyMeshLab."""
        import pymeshlab
        
        start_time = time.time()
        
        # First do triangle remeshing
        tri_result = self.tri_remesher.remesh(mesh, guidance, **kwargs)
        
        if not tri_result.success:
            return tri_result
        
        # Convert to quads
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(
            vertex_matrix=tri_result.mesh.vertices,
            face_matrix=tri_result.mesh.faces
        )
        ms.add_mesh(m)
        
        try:
            # Use turn into quad-dominant mesh
            ms.meshing_tri_to_quad_by_smart_triangle_pairing()
            
            result_mesh = ms.current_mesh()
            
            # PyMeshLab returns mixed tri/quad - extract as faces
            faces = result_mesh.face_matrix()
            
            # Check if we got quads (PyMeshLab may still return triangles)
            output = Mesh(
                vertices=result_mesh.vertex_matrix(),
                faces=faces,
                name=f"{mesh.name}_quad"
            )
            
            elapsed = time.time() - start_time
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                time_seconds=elapsed,
                metadata={"backend": "pymeshlab", "method": "tri_to_quad"}
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            # Return triangulated result on quad conversion failure
            return RemeshResult(
                mesh=tri_result.mesh,
                success=True,  # Triangle remesh succeeded
                actual_face_count=tri_result.mesh.num_faces,
                time_seconds=elapsed,
                metadata={"error": str(e), "fallback": "triangles"}
            )
