"""Remeshing backends."""

from meshretopo.remesh.base import Remesher, RemeshResult
from meshretopo.remesh.guided import GuidedQuadRemesher
from meshretopo.remesh.trimesh_backend import TrimeshRemesher, create_fallback_remesher
from meshretopo.remesh.isotropic import IsotropicRemesher
from meshretopo.remesh.hybrid import HybridRemesher
from meshretopo.remesh.feature_aware import (
    FeaturePreservingRemesher,
    AdaptiveDensityRemesher,
    BoundaryPreservingRemesher,
    feature_aware_remesh,
)

# Conditionally import PyMeshLab backends
try:
    from meshretopo.remesh.pymeshlab_backend import PyMeshLabRemesher, PyMeshLabQuadRemesher
    _HAS_PYMESHLAB = True
except ImportError:
    _HAS_PYMESHLAB = False
    PyMeshLabRemesher = None
    PyMeshLabQuadRemesher = None

__all__ = [
    "Remesher",
    "RemeshResult",
    "GuidedQuadRemesher",
    "TrimeshRemesher",
    "IsotropicRemesher",
    "HybridRemesher",
    "FeaturePreservingRemesher",
    "AdaptiveDensityRemesher",
    "BoundaryPreservingRemesher",
    "feature_aware_remesh",
    "create_fallback_remesher",
]

if _HAS_PYMESHLAB:
    __all__.extend(["PyMeshLabRemesher", "PyMeshLabQuadRemesher"])


def get_remesher(name: str, **kwargs) -> Remesher:
    """Factory function to get a remesher by name."""
    remeshers = {
        "trimesh": TrimeshRemesher,
        "guided_quad": GuidedQuadRemesher,
        "isotropic": IsotropicRemesher,
        "hybrid": HybridRemesher,
    }
    
    if _HAS_PYMESHLAB:
        remeshers["pymeshlab"] = PyMeshLabRemesher
        remeshers["pymeshlab_quad"] = PyMeshLabQuadRemesher
    
    # Default to trimesh if pymeshlab requested but not available
    if name == "pymeshlab" and not _HAS_PYMESHLAB:
        print("Warning: PyMeshLab not available, using trimesh backend")
        name = "trimesh"
    
    if name not in remeshers:
        available = ", ".join(remeshers.keys())
        raise ValueError(f"Unknown remesher: {name}. Available: {available}")
    
    return remeshers[name](**kwargs)
