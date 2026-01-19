"""
Main pipeline orchestration.

Provides the high-level API for running the full retopology pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from meshretopo.core.mesh import Mesh
from meshretopo.analysis import detect_features, create_default_analyzer
from meshretopo.guidance import GuidanceComposer, GuidanceFields
from meshretopo.remesh import get_remesher, RemeshResult
from meshretopo.evaluation import evaluate_retopology, RetopologyScore


class RetopoPipeline:
    """
    High-level retopology pipeline.
    
    Combines all stages (analysis, guidance, remeshing, evaluation)
    into a simple interface.
    """
    
    def __init__(
        self,
        backend: str = "trimesh",
        target_faces: Optional[int] = None,
        neural_weight: float = 0.7,
        feature_weight: float = 0.3,
        preserve_boundary: bool = True,
    ):
        """
        Initialize pipeline.
        
        Args:
            backend: Remeshing backend (trimesh, pymeshlab, pymeshlab_quad, guided_quad)
            target_faces: Target number of faces (None = auto)
            neural_weight: Weight for neural guidance (0-1)
            feature_weight: Weight for feature preservation (0-1)
            preserve_boundary: Preserve mesh boundaries
        """
        self.backend = backend
        self.target_faces = target_faces
        self.neural_weight = neural_weight
        self.feature_weight = feature_weight
        self.preserve_boundary = preserve_boundary
        
        # Create components
        self.analyzer = create_default_analyzer()
        self.composer = GuidanceComposer(
            neural_weight=neural_weight,
            feature_weight=feature_weight
        )
        self.remesher = get_remesher(
            backend,
            preserve_boundary=preserve_boundary
        )
    
    def process(
        self,
        input_mesh: Union[str, Path, Mesh],
        target_faces: Optional[int] = None,
        evaluate: bool = True,
        edge_flow_optimization: bool = False,
    ) -> tuple[Mesh, Optional[RetopologyScore]]:
        """
        Run full retopology pipeline.
        
        Args:
            input_mesh: Input mesh (path or Mesh object)
            target_faces: Override target face count
            evaluate: Whether to run evaluation
            edge_flow_optimization: Apply edge flow alignment post-processing
            
        Returns:
            Tuple of (output_mesh, score)
        """
        # Load mesh if needed
        if isinstance(input_mesh, (str, Path)):
            mesh = Mesh.from_file(input_mesh)
        else:
            mesh = input_mesh
        
        # Analysis
        prediction = self.analyzer.predict(mesh)
        features = detect_features(mesh)
        
        # Compose guidance
        target = target_faces or self.target_faces
        guidance = self.composer.compose(
            mesh, prediction, features,
            target_faces=target
        )
        
        # Remesh
        result = self.remesher.remesh(mesh, guidance)
        
        if not result.success:
            raise RuntimeError(f"Remeshing failed: {result.metadata.get('error', 'unknown')}")
        
        output_mesh = result.mesh
        
        # Optional edge flow optimization
        if edge_flow_optimization and guidance.direction_field is not None:
            try:
                import trimesh
                from meshretopo.postprocess import EdgeFlowOptimizer
                
                # Create trimesh for projection
                tri_mesh = mesh.triangulate() if not mesh.is_triangular else mesh
                tm = trimesh.Trimesh(
                    vertices=tri_mesh.vertices,
                    faces=tri_mesh.faces,
                    process=False
                )
                
                # Get direction field values
                direction_values = guidance.direction_field.values
                
                # Optimize
                optimizer = EdgeFlowOptimizer(alignment_strength=0.3, iterations=5)
                optimized_verts = optimizer.optimize(
                    output_mesh.vertices,
                    output_mesh.faces,
                    direction_field=direction_values[:len(output_mesh.vertices)] if len(direction_values) >= len(output_mesh.vertices) else None,
                    original_mesh=tm,
                )
                
                output_mesh = Mesh(
                    vertices=optimized_verts,
                    faces=output_mesh.faces,
                    name=output_mesh.name
                )
            except Exception:
                pass  # Skip if optimization fails
        
        # Evaluate
        score = None
        if evaluate:
            score = evaluate_retopology(output_mesh, mesh)
        
        return output_mesh, score
    
    def iterate(
        self,
        input_mesh: Union[str, Path, Mesh],
        target_score: float = 80.0,
        max_iterations: int = 5,
        target_faces: Optional[int] = None
    ) -> tuple[Mesh, RetopologyScore]:
        """
        Iteratively improve retopology until target score is reached.
        
        Args:
            input_mesh: Input mesh
            target_score: Stop when this score is reached
            max_iterations: Maximum number of iterations
            target_faces: Target face count
            
        Returns:
            Best result achieved
        """
        # Load mesh if needed
        if isinstance(input_mesh, (str, Path)):
            current_mesh = Mesh.from_file(input_mesh)
            original_mesh = current_mesh
        else:
            current_mesh = input_mesh
            original_mesh = input_mesh
        
        best_mesh = None
        best_score = None
        
        for i in range(max_iterations):
            output_mesh, score = self.process(current_mesh, target_faces, evaluate=True)
            
            # Always evaluate against original
            score = evaluate_retopology(output_mesh, original_mesh)
            
            print(f"Iteration {i+1}: Score = {score.overall_score:.1f}")
            
            if best_score is None or score.overall_score > best_score.overall_score:
                best_mesh = output_mesh
                best_score = score
            
            if score.overall_score >= target_score:
                print(f"Target score {target_score} reached!")
                break
            
            # Use output as next input (refinement)
            current_mesh = output_mesh
        
        return best_mesh, best_score


def retopo(
    input_mesh: Union[str, Path, Mesh],
    output_path: Optional[Union[str, Path]] = None,
    target_faces: Optional[int] = None,
    backend: str = "trimesh"
) -> tuple[Mesh, RetopologyScore]:
    """
    Simple retopology function.
    
    Args:
        input_mesh: Input mesh path or object
        output_path: Optional output path (saves if provided)
        target_faces: Target number of faces
        backend: Remeshing backend
        
    Returns:
        Tuple of (output_mesh, score)
    """
    pipeline = RetopoPipeline(backend=backend, target_faces=target_faces)
    output_mesh, score = pipeline.process(input_mesh)
    
    if output_path:
        output_mesh.to_file(output_path)
    
    return output_mesh, score
