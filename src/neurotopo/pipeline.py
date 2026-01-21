"""
Main pipeline orchestration.

Provides the high-level API for running the full retopology pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from neurotopo.core.mesh import Mesh
from neurotopo.analysis import detect_features, create_default_analyzer
from neurotopo.guidance import GuidanceComposer, GuidanceFields
from neurotopo.remesh import get_remesher, RemeshResult
from neurotopo.evaluation import evaluate_retopology, RetopologyScore
from neurotopo.utils.timing import (
    timed_operation, reset_timing_log, get_timing_log,
    run_with_timeout,
    TIMEOUT_CURVATURE, TIMEOUT_FEATURES, TIMEOUT_REMESH, TIMEOUT_EVALUATION,
    TimeoutError,
)

logger = logging.getLogger("neurotopo.pipeline")


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
        semantic_analysis: bool = False,
        semantic_api_provider: str = "openai",
        semantic_model: Optional[str] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            backend: Remeshing backend (trimesh, pymeshlab, pymeshlab_quad, guided_quad)
            target_faces: Target number of faces (None = auto)
            neural_weight: Weight for neural guidance (0-1)
            feature_weight: Weight for feature preservation (0-1)
            preserve_boundary: Preserve mesh boundaries
            semantic_analysis: Enable AI-powered semantic segmentation
            semantic_api_provider: Vision API provider ("openai" or "anthropic")
            semantic_model: Specific model to use (default: auto-select)
        """
        self.backend = backend
        self.target_faces = target_faces
        self.neural_weight = neural_weight
        self.feature_weight = feature_weight
        self.preserve_boundary = preserve_boundary
        self.semantic_analysis = semantic_analysis
        self.semantic_api_provider = semantic_api_provider
        self.semantic_model = semantic_model
        
        # Timeout configuration (can be overridden per-instance)
        self.timeout_analysis = TIMEOUT_CURVATURE
        self.timeout_features = TIMEOUT_FEATURES
        self.timeout_remesh = TIMEOUT_REMESH
        self.timeout_evaluation = TIMEOUT_EVALUATION
        self.timeout_postprocess = 120.0  # Post-processing timeout
        self.enforce_timeouts = False  # Set True to kill long operations
        
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
        enable_timing: bool = True,
        enforce_timeouts: Optional[bool] = None,
    ) -> tuple[Mesh, Optional[RetopologyScore]]:
        """
        Run full retopology pipeline.
        
        Args:
            input_mesh: Input mesh (path or Mesh object)
            target_faces: Override target face count
            evaluate: Whether to run evaluation
            edge_flow_optimization: Apply edge flow alignment post-processing
            enable_timing: Log timing information
            enforce_timeouts: Force timeout enforcement (overrides instance setting)
            
        Returns:
            Tuple of (output_mesh, score)
            
        Raises:
            TimeoutError: If any stage exceeds its timeout (when enforce_timeouts=True)
        """
        if enable_timing:
            reset_timing_log()
        
        # Determine if we should enforce timeouts
        do_enforce = enforce_timeouts if enforce_timeouts is not None else self.enforce_timeouts
        
        # Load mesh if needed
        if isinstance(input_mesh, (str, Path)):
            mesh = Mesh.from_file(input_mesh)
        else:
            mesh = input_mesh
        
        logger.info(f"Processing mesh: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Optional semantic analysis for topology-aware remeshing
        semantic_guidance = None
        if self.semantic_analysis:
            with timed_operation("semantic_analysis", log=enable_timing):
                try:
                    from neurotopo.analysis.semantic import SemanticAnalyzer
                    from neurotopo.guidance.semantic import SemanticGuidanceGenerator
                    
                    logger.info("Running AI semantic analysis...")
                    analyzer = SemanticAnalyzer(
                        api_provider=self.semantic_api_provider,
                        model=self.semantic_model,
                    )
                    segmentation = analyzer.analyze(mesh)
                    
                    if segmentation.segments:
                        logger.info(f"Detected {len(segmentation.segments)} semantic regions")
                        generator = SemanticGuidanceGenerator()
                        semantic_guidance = generator.generate(mesh, segmentation)
                    else:
                        logger.info("No semantic regions detected")
                except Exception as e:
                    logger.warning(f"Semantic analysis failed: {e}")
        
        # Analysis with timing and optional timeout enforcement
        with timed_operation("analysis", log=enable_timing):
            if do_enforce:
                prediction = run_with_timeout(
                    self.analyzer.predict, (mesh,),
                    timeout=self.timeout_analysis,
                    operation_name="analysis"
                )
            else:
                prediction = self.analyzer.predict(mesh)
        
        # Feature detection with timing
        with timed_operation("feature_detection", log=enable_timing):
            if do_enforce:
                features = run_with_timeout(
                    detect_features, (mesh,),
                    timeout=self.timeout_features,
                    operation_name="feature_detection"
                )
            else:
                features = detect_features(mesh)
        
        # Compose guidance (enhanced with semantic info if available)
        with timed_operation("guidance_composition", log=enable_timing):
            target = target_faces or self.target_faces
            guidance = self.composer.compose(
                mesh, prediction, features,
                target_faces=target
            )
            
            # Blend semantic guidance if available
            if semantic_guidance is not None:
                guidance = self._blend_semantic_guidance(guidance, semantic_guidance)
        
        # Remesh with timing
        with timed_operation("remeshing", log=enable_timing):
            if do_enforce:
                result = run_with_timeout(
                    self.remesher.remesh, (mesh, guidance),
                    timeout=self.timeout_remesh,
                    operation_name="remeshing"
                )
            else:
                result = self.remesher.remesh(mesh, guidance)
        
        if not result.success:
            raise RuntimeError(f"Remeshing failed: {result.metadata.get('error', 'unknown')}")
        
        output_mesh = result.mesh
        
        # Ensure manifold mesh
        with timed_operation("manifold_repair", log=enable_timing):
            try:
                from neurotopo.postprocess.manifold import ManifoldRepair
                repair = ManifoldRepair(verbose=False)
                repaired_verts, repaired_faces = repair.repair(
                    output_mesh.vertices.copy(),
                    output_mesh.faces.copy()
                )
                output_mesh = Mesh(
                    vertices=repaired_verts,
                    faces=repaired_faces,
                    name=output_mesh.name
                )
                logger.debug(f"Manifold repair: {len(repaired_faces)} faces")
            except Exception as e:
                logger.warning(f"Manifold repair failed: {e}")
        
        # Enhanced post-processing (RetopoFlow-inspired)
        with timed_operation("enhanced_postprocess", log=enable_timing):
            try:
                if do_enforce:
                    output_mesh = run_with_timeout(
                        self._enhanced_postprocess,
                        kwargs={
                            'output_mesh': output_mesh,
                            'original_mesh': mesh,
                            'features': features,
                            'edge_flow_optimization': edge_flow_optimization,
                            'guidance': guidance,
                        },
                        timeout=self.timeout_postprocess,
                        operation_name="enhanced_postprocess"
                    )
                else:
                    output_mesh = self._enhanced_postprocess(
                        output_mesh=output_mesh,
                        original_mesh=mesh,
                        features=features,
                        edge_flow_optimization=edge_flow_optimization,
                        guidance=guidance,
                    )
            except TimeoutError:
                logger.warning("Enhanced postprocess timed out, returning mesh without postprocessing")
                # Continue with unprocessed mesh rather than failing
        
        # Evaluate with timing
        score = None
        if evaluate:
            with timed_operation("evaluation", log=enable_timing):
                if do_enforce:
                    score = run_with_timeout(
                        evaluate_retopology, (output_mesh, mesh),
                        timeout=self.timeout_evaluation,
                        operation_name="evaluation"
                    )
                else:
                    score = evaluate_retopology(output_mesh, mesh)
        
        # Log timing summary
        if enable_timing:
            timing_log = get_timing_log()
            logger.info(timing_log.summary())
        
        return output_mesh, score
    
    def _enhanced_postprocess(
        self,
        output_mesh: Mesh,
        original_mesh: Mesh,
        features,
        edge_flow_optimization: bool,
        guidance: GuidanceFields,
    ) -> Mesh:
        """
        Enhanced post-processing with RetopoFlow-inspired techniques.
        
        Includes:
        - Feature edge locking
        - Surface projection (shrinkwrap)
        - Principal curvature alignment
        - Iterative relaxation with projection
        - Final shrinkwrap pass
        """
        try:
            import trimesh
            from neurotopo.postprocess import (
                EnhancedQuadOptimizer,
                EnhancedOptimizerConfig,
                relax_with_features,
            )
            
            # Create trimesh for projection
            tri_mesh = original_mesh.triangulate() if not original_mesh.is_triangular else original_mesh
            tm = trimesh.Trimesh(
                vertices=tri_mesh.vertices,
                faces=tri_mesh.faces,
                process=False
            )
            
            # Decimate large meshes for faster projection queries
            # Keep enough detail for accurate surface projection
            MAX_PROJECTION_FACES = 50000
            if len(tm.faces) > MAX_PROJECTION_FACES:
                logger.debug(f"Decimating projection mesh: {len(tm.faces)} -> {MAX_PROJECTION_FACES}")
                try:
                    tm = tm.simplify_quadric_decimation(MAX_PROJECTION_FACES)
                except Exception:
                    # Fall back if quadric decimation not available
                    pass
            
            # Get feature vertices for locking
            feature_vertices = None
            if features is not None:
                try:
                    feature_vertices = set(features.get_feature_vertices())
                    logger.debug(f"Locking {len(feature_vertices)} feature vertices")
                except Exception:
                    pass
            
            # Compute principal directions for curvature alignment
            principal_directions = None
            if edge_flow_optimization:
                try:
                    from neurotopo.analysis.curvature import CurvatureAnalyzer
                    analyzer = CurvatureAnalyzer(original_mesh)
                    principal_directions = analyzer.compute_principal_directions()
                except Exception as e:
                    logger.debug(f"Principal direction computation failed: {e}")
            
            # Enhanced optimization with feature locking and shrinkwrap
            # Use fewer iterations for speed while maintaining quality
            config = EnhancedOptimizerConfig(
                iterations=6,  # Reduced from 12 for speed
                lock_feature_edges=True,
                lock_boundary=True,
                final_shrinkwrap=True,
                project_every_step=False,  # Only project at end for speed
                align_to_curvature=edge_flow_optimization and principal_directions is not None,
            )
            
            optimizer = EnhancedQuadOptimizer(config)
            optimized_verts = optimizer.optimize(
                vertices=output_mesh.vertices,
                faces=output_mesh.faces,
                original_mesh=tm,
                feature_vertices=feature_vertices,
                principal_directions=principal_directions,
            )
            
            # Light relaxation pass with projection (reduced iterations)
            relaxed_verts = relax_with_features(
                vertices=optimized_verts,
                faces=output_mesh.faces,
                original_mesh=tm,
                feature_set=features,
                iterations=3,  # Reduced from 5
            )
            
            output_mesh = Mesh(
                vertices=relaxed_verts,
                faces=output_mesh.faces,
                name=output_mesh.name
            )
            
        except Exception as e:
            logger.debug(f"Enhanced post-processing failed: {e}")
            # Fall back to legacy edge flow optimization if new code fails
            if edge_flow_optimization and guidance.direction_field is not None:
                output_mesh = self._legacy_edge_flow_optimization(
                    output_mesh, original_mesh, guidance
                )
        
        return output_mesh
    
    def _legacy_edge_flow_optimization(
        self,
        output_mesh: Mesh,
        original_mesh: Mesh,
        guidance: GuidanceFields,
    ) -> Mesh:
        """Legacy edge flow optimization (fallback)."""
        try:
            import trimesh
            from neurotopo.postprocess import EdgeFlowOptimizer
            
            tri_mesh = original_mesh.triangulate() if not original_mesh.is_triangular else original_mesh
            tm = trimesh.Trimesh(
                vertices=tri_mesh.vertices,
                faces=tri_mesh.faces,
                process=False
            )
            
            direction_values = guidance.direction_field.values
            
            optimizer = EdgeFlowOptimizer(alignment_strength=0.3, iterations=5)
            optimized_verts = optimizer.optimize(
                output_mesh.vertices,
                output_mesh.faces,
                direction_field=direction_values[:len(output_mesh.vertices)] if len(direction_values) >= len(output_mesh.vertices) else None,
                original_mesh=tm,
            )
            
            return Mesh(
                vertices=optimized_verts,
                faces=output_mesh.faces,
                name=output_mesh.name
            )
        except Exception:
            return output_mesh
    
    def _blend_semantic_guidance(
        self,
        guidance: GuidanceFields,
        semantic: 'SemanticGuidanceFields'
    ) -> GuidanceFields:
        """
        Blend semantic guidance into standard guidance fields.
        
        Adjusts the size field based on semantic density requirements
        and stores semantic info for the remesher to use.
        """
        import numpy as np
        from neurotopo.core.fields import ScalarField, FieldLocation
        
        # Blend size field with semantic density
        # Higher density regions -> smaller quads
        density = semantic.density_field.values
        density_factor = 1.0 / np.clip(density, 0.5, 2.0)  # Invert density
        
        # Ensure arrays are same length (interpolate if needed)
        if len(density_factor) != len(guidance.size_field.values):
            # Use nearest neighbor interpolation for now
            if len(density_factor) > len(guidance.size_field.values):
                density_factor = density_factor[:len(guidance.size_field.values)]
            else:
                # Extend with 1.0 (neutral)
                density_factor = np.concatenate([
                    density_factor,
                    np.ones(len(guidance.size_field.values) - len(density_factor))
                ])
        
        blended_sizes = guidance.size_field.values * density_factor
        blended_size_field = ScalarField(
            blended_sizes,
            FieldLocation.VERTEX,
            "semantic_blended_size"
        )
        
        # Create new guidance with blended size
        return GuidanceFields(
            size_field=blended_size_field,
            direction_field=guidance.direction_field,
            importance_field=guidance.importance_field,
            target_face_count=guidance.target_face_count,
            symmetry_plane=guidance.symmetry_plane,
        )
    
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
