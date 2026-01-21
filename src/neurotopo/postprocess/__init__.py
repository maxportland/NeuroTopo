"""Post-processing modules for mesh optimization."""

from neurotopo.postprocess.optimizer import QuadOptimizer, EdgeFlowOptimizer
from neurotopo.postprocess.manifold import ManifoldRepair, make_manifold, check_manifold
from neurotopo.postprocess.shrinkwrap import (
    Shrinkwrap, 
    ShrinkwrapConfig, 
    ProjectedVertexTracker,
    create_shrinkwrap,
)
from neurotopo.postprocess.enhanced_optimizer import (
    EnhancedQuadOptimizer,
    EnhancedOptimizerConfig,
    optimize_with_features,
)
from neurotopo.postprocess.relaxation import (
    ProjectedRelaxation,
    RelaxationConfig,
    LocalRelaxBrush,
    relax_mesh,
    relax_with_features,
)
from neurotopo.postprocess.pole_reduction import (
    PoleReducer,
    reduce_poles,
)

__all__ = [
    # Original optimizer
    "QuadOptimizer", 
    "EdgeFlowOptimizer",
    # Manifold repair
    "ManifoldRepair",
    "make_manifold",
    "check_manifold",
    # Shrinkwrap (surface snapping)
    "Shrinkwrap",
    "ShrinkwrapConfig",
    "ProjectedVertexTracker",
    "create_shrinkwrap",
    # Enhanced optimizer with feature locking
    "EnhancedQuadOptimizer",
    "EnhancedOptimizerConfig",
    "optimize_with_features",
    # Relaxation with projection
    "ProjectedRelaxation",
    "RelaxationConfig",
    "LocalRelaxBrush",
    "relax_mesh",
    "relax_with_features",
    # Pole reduction
    "PoleReducer",
    "reduce_poles",
]
