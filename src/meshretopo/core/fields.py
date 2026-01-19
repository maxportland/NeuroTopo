"""
Scalar and vector fields on meshes.

Used to represent guidance information (size fields, direction fields, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import numpy as np

from meshretopo.core.mesh import Mesh


class FieldLocation(Enum):
    """Where field values are defined."""
    VERTEX = "vertex"
    FACE = "face"
    EDGE = "edge"


@dataclass
class ScalarField:
    """
    Scalar field on a mesh surface.
    
    Used for: curvature, sizing field, importance weights, etc.
    """
    values: np.ndarray
    location: FieldLocation
    name: str = "scalar_field"
    
    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
    
    @property
    def min(self) -> float:
        return float(self.values.min())
    
    @property
    def max(self) -> float:
        return float(self.values.max())
    
    @property
    def mean(self) -> float:
        return float(self.values.mean())
    
    def normalize(self, vmin: float = 0.0, vmax: float = 1.0) -> ScalarField:
        """Normalize values to [vmin, vmax] range."""
        field_min, field_max = self.min, self.max
        if field_max - field_min < 1e-10:
            normalized = np.full_like(self.values, (vmin + vmax) / 2)
        else:
            normalized = (self.values - field_min) / (field_max - field_min)
            normalized = normalized * (vmax - vmin) + vmin
        
        return ScalarField(normalized, self.location, f"{self.name}_normalized")
    
    def clamp(self, vmin: float, vmax: float) -> ScalarField:
        """Clamp values to [vmin, vmax]."""
        clamped = np.clip(self.values, vmin, vmax)
        return ScalarField(clamped, self.location, f"{self.name}_clamped")
    
    def smooth(self, mesh: Mesh, iterations: int = 1, factor: float = 0.5) -> ScalarField:
        """Laplacian smoothing of scalar field (vectorized)."""
        if self.location != FieldLocation.VERTEX:
            raise ValueError("Smoothing only supported for vertex fields")
        
        values = self.values.copy()
        n_verts = mesh.num_vertices
        faces = np.array(mesh.faces)
        
        # Build sparse adjacency using scipy for large meshes
        if n_verts > 10000:
            try:
                from scipy import sparse
                
                # Build sparse adjacency matrix
                rows = []
                cols = []
                for i in range(len(faces[0])):
                    for j in range(len(faces[0])):
                        if i != j:
                            rows.extend(faces[:, i])
                            cols.extend(faces[:, j])
                
                data = np.ones(len(rows))
                adj_matrix = sparse.csr_matrix(
                    (data, (rows, cols)), 
                    shape=(n_verts, n_verts)
                )
                # Make binary (some edges may be duplicated)
                adj_matrix.data[:] = 1
                
                # Compute degree matrix
                degree = np.array(adj_matrix.sum(axis=1)).flatten()
                degree = np.maximum(degree, 1)
                
                # Smooth using matrix operations
                for _ in range(iterations):
                    neighbor_sum = adj_matrix.dot(values)
                    neighbor_avg = neighbor_sum / degree
                    values = values * (1 - factor) + neighbor_avg * factor
                
                return ScalarField(values, self.location, f"{self.name}_smoothed")
            except ImportError:
                pass  # Fall back to basic method
        
        # Basic method for small meshes
        adjacency = [set() for _ in range(n_verts)]
        for face in mesh.faces:
            for i, vi in enumerate(face):
                for j, vj in enumerate(face):
                    if i != j:
                        adjacency[vi].add(vj)
        
        adjacency = [list(adj) for adj in adjacency]
        
        # Smooth
        for _ in range(iterations):
            new_values = values.copy()
            for i, neighbors in enumerate(adjacency):
                if neighbors:
                    neighbor_avg = np.mean(values[neighbors])
                    new_values[i] = values[i] * (1 - factor) + neighbor_avg * factor
            values = new_values
        
        return ScalarField(values, self.location, f"{self.name}_smoothed")
    
    def apply(self, func: Callable[[np.ndarray], np.ndarray]) -> ScalarField:
        """Apply a function to the values."""
        return ScalarField(func(self.values), self.location, self.name)
    
    def __add__(self, other: ScalarField) -> ScalarField:
        if self.location != other.location:
            raise ValueError("Cannot add fields with different locations")
        return ScalarField(self.values + other.values, self.location, f"{self.name}+{other.name}")
    
    def __mul__(self, scalar: float) -> ScalarField:
        return ScalarField(self.values * scalar, self.location, self.name)
    
    def __repr__(self) -> str:
        return f"ScalarField('{self.name}', {self.location.value}, range=[{self.min:.3f}, {self.max:.3f}])"


@dataclass
class VectorField:
    """
    Vector field on a mesh surface.
    
    Used for: edge flow directions, gradient fields, etc.
    """
    vectors: np.ndarray
    location: FieldLocation
    name: str = "vector_field"
    
    def __post_init__(self):
        self.vectors = np.asarray(self.vectors, dtype=np.float64)
    
    @property
    def magnitudes(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=1)
    
    def normalize(self) -> VectorField:
        """Normalize all vectors to unit length."""
        mags = self.magnitudes[:, np.newaxis]
        mags = np.where(mags > 1e-10, mags, 1.0)
        normalized = self.vectors / mags
        return VectorField(normalized, self.location, f"{self.name}_normalized")
    
    def project_to_tangent(self, mesh: Mesh) -> VectorField:
        """Project vectors to mesh tangent plane."""
        if self.location != FieldLocation.VERTEX:
            raise NotImplementedError("Only vertex fields supported")
        
        if mesh.normals is None:
            mesh.compute_normals()
        
        # v_tangent = v - (v · n) * n
        dots = np.sum(self.vectors * mesh.normals, axis=1, keepdims=True)
        projected = self.vectors - dots * mesh.normals
        
        return VectorField(projected, self.location, f"{self.name}_tangent")
    
    def smooth(self, mesh: Mesh, iterations: int = 1, factor: float = 0.5) -> VectorField:
        """Laplacian smoothing of vector field."""
        if self.location != FieldLocation.VERTEX:
            raise ValueError("Smoothing only supported for vertex fields")
        
        vectors = self.vectors.copy()
        
        # Build adjacency
        adjacency = [[] for _ in range(mesh.num_vertices)]
        for face in mesh.faces:
            for i, vi in enumerate(face):
                for j, vj in enumerate(face):
                    if i != j:
                        adjacency[vi].append(vj)
        adjacency = [list(set(adj)) for adj in adjacency]
        
        for _ in range(iterations):
            new_vectors = vectors.copy()
            for i, neighbors in enumerate(adjacency):
                if neighbors:
                    neighbor_avg = np.mean(vectors[neighbors], axis=0)
                    new_vectors[i] = vectors[i] * (1 - factor) + neighbor_avg * factor
            vectors = new_vectors
        
        return VectorField(vectors, self.location, f"{self.name}_smoothed")
    
    def to_scalar(self) -> ScalarField:
        """Convert to scalar field using magnitudes."""
        return ScalarField(self.magnitudes, self.location, f"{self.name}_magnitude")
    
    def __repr__(self) -> str:
        avg_mag = float(self.magnitudes.mean())
        return f"VectorField('{self.name}', {self.location.value}, avg_mag={avg_mag:.3f})"


@dataclass 
class DirectionField:
    """
    Direction field (cross field) on mesh surface.
    
    Represents 4-RoSy (4-way rotationally symmetric) directions for quad meshing.
    Each direction can be rotated by 90° and still be valid.
    """
    directions: np.ndarray  # Primary direction at each location
    location: FieldLocation
    singularities: Optional[np.ndarray] = None  # Indices of singular points
    name: str = "direction_field"
    
    def __post_init__(self):
        self.directions = np.asarray(self.directions, dtype=np.float64)
    
    def get_orthogonal(self, mesh: Mesh) -> np.ndarray:
        """Get the orthogonal direction (rotated 90° in tangent plane)."""
        if mesh.normals is None:
            mesh.compute_normals()
        
        if self.location == FieldLocation.VERTEX:
            normals = mesh.normals
        else:
            normals = mesh.face_normals
        
        return np.cross(normals, self.directions)
    
    def __repr__(self) -> str:
        n_sing = len(self.singularities) if self.singularities is not None else 0
        return f"DirectionField('{self.name}', {self.location.value}, {n_sing} singularities)"
