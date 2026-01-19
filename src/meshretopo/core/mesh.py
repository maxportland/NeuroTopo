"""
Core mesh data structure with unified representation.

Designed to work seamlessly with multiple backends (trimesh, open3d, pymeshlab)
while providing a consistent interface for the retopology pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np


@dataclass
class Mesh:
    """
    Unified mesh representation for the retopology pipeline.
    
    Attributes:
        vertices: Nx3 array of vertex positions
        faces: Mx3 (triangles) or Mx4 (quads) array of face indices
        normals: Optional Nx3 array of vertex normals
        face_normals: Optional Mx3 array of face normals
        name: Optional mesh identifier
        metadata: Additional mesh properties
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    face_normals: Optional[np.ndarray] = None
    name: str = "unnamed"
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize mesh data."""
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be Nx3, got shape {self.vertices.shape}")
        
        if self.faces.ndim != 2 or self.faces.shape[1] not in (3, 4):
            raise ValueError(f"Faces must be Mx3 or Mx4, got shape {self.faces.shape}")
    
    @property
    def num_vertices(self) -> int:
        return len(self.vertices)
    
    @property
    def num_faces(self) -> int:
        return len(self.faces)
    
    @property
    def is_quad(self) -> bool:
        """Check if mesh is quad-dominant."""
        return self.faces.shape[1] == 4
    
    @property
    def is_triangular(self) -> bool:
        """Check if mesh is triangular."""
        return self.faces.shape[1] == 3
    
    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box (min, max)."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)
    
    @property
    def center(self) -> np.ndarray:
        """Get bounding box center."""
        min_b, max_b = self.bounds
        return (min_b + max_b) / 2
    
    @property
    def diagonal(self) -> float:
        """Get bounding box diagonal length."""
        min_b, max_b = self.bounds
        return np.linalg.norm(max_b - min_b)
    
    def compute_normals(self) -> None:
        """Compute vertex and face normals."""
        if self.is_triangular:
            self._compute_triangle_normals()
        else:
            self._compute_quad_normals()
    
    def _compute_triangle_normals(self) -> None:
        """Compute normals for triangular mesh."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        # Face normals
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self.face_normals = face_normals / norms
        
        # Vertex normals (area-weighted average)
        self.normals = np.zeros_like(self.vertices)
        for i, face in enumerate(self.faces):
            for vi in face:
                self.normals[vi] += self.face_normals[i]
        
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self.normals = self.normals / norms
    
    def _compute_quad_normals(self) -> None:
        """Compute normals for quad mesh (use triangle fan)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        v3 = self.vertices[self.faces[:, 3]]
        
        # Average of two triangle normals
        n1 = np.cross(v1 - v0, v2 - v0)
        n2 = np.cross(v2 - v0, v3 - v0)
        face_normals = n1 + n2
        
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self.face_normals = face_normals / norms
        
        # Vertex normals
        self.normals = np.zeros_like(self.vertices)
        for i, face in enumerate(self.faces):
            for vi in face:
                self.normals[vi] += self.face_normals[i]
        
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self.normals = self.normals / norms
    
    def triangulate(self) -> Mesh:
        """Convert quad mesh to triangles."""
        if self.is_triangular:
            return self.copy()
        
        # Split each quad into two triangles
        tri_faces = []
        for face in self.faces:
            tri_faces.append([face[0], face[1], face[2]])
            tri_faces.append([face[0], face[2], face[3]])
        
        return Mesh(
            vertices=self.vertices.copy(),
            faces=np.array(tri_faces),
            name=f"{self.name}_triangulated",
            metadata=self.metadata.copy()
        )
    
    def copy(self) -> Mesh:
        """Create a deep copy."""
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            face_normals=self.face_normals.copy() if self.face_normals is not None else None,
            name=self.name,
            metadata=self.metadata.copy()
        )
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Mesh:
        """Load mesh from file (OBJ, PLY, STL, etc.)."""
        import trimesh
        
        path = Path(path)
        tm = trimesh.load(path, force='mesh')
        
        return cls(
            vertices=np.asarray(tm.vertices),
            faces=np.asarray(tm.faces),
            name=path.stem,
            metadata={"source_file": str(path)}
        )
    
    def to_file(self, path: Union[str, Path]) -> None:
        """Save mesh to file."""
        import trimesh
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.is_quad:
            # trimesh doesn't handle quads well, use manual OBJ export
            self._export_obj(path)
        else:
            tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
            tm.export(path)
    
    def _export_obj(self, path: Path) -> None:
        """Export to OBJ format (supports quads)."""
        with open(path, 'w') as f:
            f.write(f"# MeshRetopo export: {self.name}\n")
            f.write(f"# Vertices: {self.num_vertices}, Faces: {self.num_faces}\n\n")
            
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            for face in self.faces:
                indices = " ".join(str(i + 1) for i in face)
                f.write(f"f {indices}\n")
    
    def to_trimesh(self):
        """Convert to trimesh object."""
        import trimesh
        if self.is_quad:
            return self.triangulate().to_trimesh()
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
    
    def to_open3d(self):
        """Convert to Open3D mesh."""
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        
        if self.is_quad:
            tri = self.triangulate()
            mesh.vertices = o3d.utility.Vector3dVector(tri.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(tri.faces)
        else:
            mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        
        mesh.compute_vertex_normals()
        return mesh
    
    def __repr__(self) -> str:
        face_type = "quads" if self.is_quad else "triangles"
        return f"Mesh('{self.name}', {self.num_vertices} verts, {self.num_faces} {face_type})"
