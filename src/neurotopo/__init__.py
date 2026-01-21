"""
NeuroTopo: AI-Assisted Retopology

Neural-guided, deterministically-controlled mesh retopology.
"""

__version__ = "0.3.0"

# Suppress trimesh's verbose timing logs (on_surface, etc.) by default
import logging
logging.getLogger("trimesh").setLevel(logging.WARNING)

from neurotopo.core.mesh import Mesh
from neurotopo.pipeline import RetopoPipeline
from neurotopo.tuning import auto_retopo, AutoTuner

# Utilities
try:
    from neurotopo.core.io import load_mesh, save_mesh
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
