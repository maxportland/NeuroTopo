"""
MeshRetopo: AI-Assisted Retopology

Neural-guided, deterministically-controlled mesh retopology.
"""

__version__ = "0.3.0"

from meshretopo.core.mesh import Mesh
from meshretopo.pipeline import RetopoPipeline
from meshretopo.tuning import auto_retopo, AutoTuner

# Utilities
try:
    from meshretopo.core.io import load_mesh, save_mesh
except ImportError:
    load_mesh = None
    save_mesh = None

__all__ = [
    "Mesh",
    "RetopoPipeline", 
    "auto_retopo",
    "AutoTuner",
    "load_mesh",
    "save_mesh",
    "__version__",
]
