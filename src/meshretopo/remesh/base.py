"""
Abstract base class for remeshing backends.

All remeshers implement this interface, allowing easy swapping
between different algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from meshretopo.core.mesh import Mesh
from meshretopo.guidance.composer import GuidanceFields


@dataclass
class RemeshResult:
    """Result from a remeshing operation."""
    mesh: Mesh
    success: bool
    actual_face_count: int
    iterations: int = 0
    time_seconds: float = 0.0
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Remesher(ABC):
    """Abstract base class for remeshing algorithms."""
    
    @abstractmethod
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """
        Perform remeshing operation.
        
        Args:
            mesh: Input high-poly mesh
            guidance: Guidance fields for the remeshing
            **kwargs: Backend-specific options
            
        Returns:
            RemeshResult with the output mesh and metadata
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        pass
    
    @property
    @abstractmethod
    def supports_quads(self) -> bool:
        """Whether this backend can produce quad meshes."""
        pass
    
    @property
    def supports_guidance(self) -> bool:
        """Whether this backend uses guidance fields."""
        return True
