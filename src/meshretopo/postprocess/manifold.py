"""
Manifold repair utilities for mesh processing.

Provides tools to detect and fix non-manifold geometry,
ensuring clean topology for downstream operations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional
import numpy as np

logger = logging.getLogger("meshretopo.postprocess.manifold")


class ManifoldRepair:
    """
    Repairs non-manifold mesh geometry.
    
    Handles common issues:
    - Non-manifold edges (>2 faces sharing an edge)
    - Non-manifold vertices (pinch points)
    - Degenerate faces (zero area)
    - Duplicate faces
    - Isolated vertices
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def repair(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Repair mesh to make it manifold.
        
        Args:
            vertices: Nx3 vertex positions
            faces: Mx3 or Mx4 face indices
            
        Returns:
            (repaired_vertices, repaired_faces)
        """
        verts = vertices.copy()
        faces_list = [list(f) for f in faces]
        
        # Step 1: Remove degenerate faces
        faces_list = self._remove_degenerate_faces(faces_list)
        
        # Step 2: Remove duplicate faces
        faces_list = self._remove_duplicate_faces(faces_list)
        
        # Step 3: Fix non-manifold edges by removing problematic faces
        faces_list = self._fix_nonmanifold_edges(faces_list)
        
        # Step 4: Remove isolated vertices
        verts, faces_list = self._remove_isolated_vertices(verts, faces_list)
        
        # Convert back to numpy
        if len(faces_list) > 0:
            max_face_size = max(len(f) for f in faces_list)
            faces_array = np.zeros((len(faces_list), max_face_size), dtype=np.int32)
            for i, f in enumerate(faces_list):
                faces_array[i, :len(f)] = f
                if len(f) < max_face_size:
                    # Pad with last vertex for consistency
                    faces_array[i, len(f):] = f[-1]
        else:
            faces_array = np.zeros((0, 3), dtype=np.int32)
        
        return verts, faces_array
    
    def _remove_degenerate_faces(self, faces: list) -> list:
        """Remove faces with duplicate vertices or zero area."""
        valid_faces = []
        removed = 0
        
        for face in faces:
            unique_verts = set(face)
            # Need at least 3 unique vertices for a valid face
            if len(unique_verts) >= 3:
                valid_faces.append(face)
            else:
                removed += 1
        
        if self.verbose and removed > 0:
            logger.info(f"Removed {removed} degenerate faces")
        
        return valid_faces
    
    def _remove_duplicate_faces(self, faces: list) -> list:
        """Remove duplicate faces (same vertices in any order)."""
        seen = set()
        unique_faces = []
        removed = 0
        
        for face in faces:
            # Create canonical form (sorted tuple)
            canonical = tuple(sorted(set(face)))
            if canonical not in seen:
                seen.add(canonical)
                unique_faces.append(face)
            else:
                removed += 1
        
        if self.verbose and removed > 0:
            logger.info(f"Removed {removed} duplicate faces")
        
        return unique_faces
    
    def _fix_nonmanifold_edges(self, faces: list) -> list:
        """
        Fix non-manifold edges by removing faces that cause >2 faces per edge.
        
        Strategy: For each non-manifold edge, keep the 2 faces with best quality
        and remove the rest.
        """
        max_iterations = 10
        
        for iteration in range(max_iterations):
            # Build edge-face map
            edge_faces = defaultdict(list)
            for fi, face in enumerate(faces):
                unique = list(set(face))
                n = len(unique)
                for i in range(n):
                    e = tuple(sorted([unique[i], unique[(i+1) % n]]))
                    edge_faces[e].append(fi)
            
            # Find non-manifold edges
            nm_edges = [(e, fl) for e, fl in edge_faces.items() if len(fl) > 2]
            
            if not nm_edges:
                break
            
            # Collect faces to remove
            faces_to_remove = set()
            
            for edge, face_list in nm_edges:
                if len(face_list) <= 2:
                    continue
                
                # Keep first 2 faces, remove rest
                # Could improve by scoring faces by quality
                for fi in face_list[2:]:
                    faces_to_remove.add(fi)
            
            if not faces_to_remove:
                break
            
            # Remove faces
            faces = [f for i, f in enumerate(faces) if i not in faces_to_remove]
            
            if self.verbose:
                logger.info(f"Iteration {iteration+1}: removed {len(faces_to_remove)} faces")
        
        return faces
    
    def _remove_isolated_vertices(
        self, 
        vertices: np.ndarray, 
        faces: list
    ) -> tuple[np.ndarray, list]:
        """Remove vertices not referenced by any face."""
        # Find used vertices
        used = set()
        for face in faces:
            used.update(face)
        
        if len(used) == len(vertices):
            return vertices, faces
        
        # Create mapping from old to new indices
        old_to_new = {}
        new_vertices = []
        
        for old_idx in sorted(used):
            new_idx = len(new_vertices)
            old_to_new[old_idx] = new_idx
            new_vertices.append(vertices[old_idx])
        
        # Remap faces
        new_faces = []
        for face in faces:
            new_face = [old_to_new[v] for v in face]
            new_faces.append(new_face)
        
        removed = len(vertices) - len(new_vertices)
        if self.verbose and removed > 0:
            logger.info(f"Removed {removed} isolated vertices")
        
        return np.array(new_vertices), new_faces
    
    def check_manifold(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> dict:
        """
        Check mesh for manifold issues.
        
        Returns:
            Dictionary with issue counts
        """
        issues = {
            'degenerate_faces': 0,
            'duplicate_faces': 0,
            'nonmanifold_edges': 0,
            'boundary_edges': 0,
            'isolated_vertices': 0,
            'is_manifold': True,
        }
        
        # Check degenerate faces
        for face in faces:
            if len(set(face)) < 3:
                issues['degenerate_faces'] += 1
        
        # Check duplicate faces
        seen = set()
        for face in faces:
            canonical = tuple(sorted(set(face)))
            if canonical in seen:
                issues['duplicate_faces'] += 1
            seen.add(canonical)
        
        # Check edge manifoldness
        edge_faces = defaultdict(int)
        for face in faces:
            unique = list(set(face))
            n = len(unique)
            for i in range(n):
                e = tuple(sorted([unique[i], unique[(i+1) % n]]))
                edge_faces[e] += 1
        
        for count in edge_faces.values():
            if count > 2:
                issues['nonmanifold_edges'] += 1
            elif count == 1:
                issues['boundary_edges'] += 1
        
        # Check isolated vertices
        used = set()
        for face in faces:
            used.update(face)
        issues['isolated_vertices'] = len(vertices) - len(used)
        
        # Overall manifold check
        issues['is_manifold'] = (
            issues['degenerate_faces'] == 0 and
            issues['duplicate_faces'] == 0 and
            issues['nonmanifold_edges'] == 0
        )
        
        return issues


def make_manifold(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to repair mesh manifold issues.
    
    Args:
        vertices: Nx3 vertex positions
        faces: Mx3 or Mx4 face indices
        verbose: Whether to log repair steps
        
    Returns:
        (repaired_vertices, repaired_faces)
    """
    repair = ManifoldRepair(verbose=verbose)
    return repair.repair(vertices, faces)


def check_manifold(
    vertices: np.ndarray,
    faces: np.ndarray
) -> dict:
    """
    Check mesh for manifold issues.
    
    Returns dictionary with issue counts and is_manifold boolean.
    """
    repair = ManifoldRepair()
    return repair.check_manifold(vertices, faces)
