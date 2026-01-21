"""
Quality metrics for retopology evaluation.

Provides comprehensive metrics to objectively measure retopology quality,
enabling data-driven iteration and improvement.

Performance optimizations:
- MeshTopologyCache: Single-pass computation of all mesh topology data
- Vectorized quad quality metrics using NumPy broadcasting
- Efficient edge/valence computation with pre-allocated arrays
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np
from scipy.spatial import cKDTree

from neurotopo.core.mesh import Mesh
from neurotopo.evaluation.visual import VisualEvaluator, VisualQualityMetrics

logger = logging.getLogger("neurotopo.evaluation")


@dataclass
class MeshTopologyCache:
    """
    Pre-computed mesh topology data for efficient metric computation.
    
    Computes all topology information in a single pass through the mesh,
    avoiding redundant traversals in individual metric functions.
    """
    # Basic counts
    num_vertices: int
    num_faces: int
    num_edges: int
    
    # Valence data
    valence: np.ndarray  # Per-vertex valence
    valence_histogram: Dict[int, int]
    
    # Edge data
    edge_count: Dict[Tuple[int, int], int]  # Edge -> face count
    boundary_edges: int
    non_manifold_edges: int
    
    # Face angles (for quad quality)
    face_angles: np.ndarray  # F x 4 angles per face (for quads)
    face_aspect_ratios: np.ndarray  # F aspect ratios
    face_types: np.ndarray  # 3=tri, 4=quad, 0=degenerate
    
    # Curvature (for pole analysis)
    vertex_curvature: np.ndarray  # Per-vertex curvature estimate
    
    @classmethod
    def from_mesh(cls, mesh: Mesh) -> 'MeshTopologyCache':
        """Build topology cache from mesh in a single pass."""
        n_verts = mesh.num_vertices
        n_faces = mesh.num_faces
        is_quad_mesh = mesh.is_quad
        
        # Pre-allocate arrays
        # NOTE: We compute EDGE valence (number of edges per vertex), not face valence
        # This is the correct metric for topology guidelines (valence 4 = regular for quads)
        edge_valence = np.zeros(n_verts, dtype=np.int32)
        vertex_angle_sum = np.zeros(n_verts, dtype=np.float64)
        
        # Edge tracking - use set for unique edges
        edge_set = set()
        edge_count = {}
        
        # Face data
        face_angles = np.zeros((n_faces, 4), dtype=np.float64)
        face_aspect_ratios = np.zeros(n_faces, dtype=np.float64)
        face_types = np.zeros(n_faces, dtype=np.int32)
        
        # Single pass through all faces
        for fi, face in enumerate(mesh.faces):
            unique_verts = list(dict.fromkeys(face))  # Preserve order, remove duplicates
            n_unique = len(unique_verts)
            
            # Determine face type
            if n_unique < 3:
                face_types[fi] = 0  # Degenerate
                continue
            elif n_unique == 3:
                face_types[fi] = 3  # Triangle
            else:
                face_types[fi] = 4  # Quad
            
            # Build edges and track valence
            for i in range(n_unique):
                v0, v1 = unique_verts[i], unique_verts[(i + 1) % n_unique]
                if v0 != v1:
                    edge = (min(v0, v1), max(v0, v1))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
                    
                    # Only count edge for valence if we haven't seen it
                    if edge not in edge_set:
                        edge_set.add(edge)
                        edge_valence[v0] += 1
                        edge_valence[v1] += 1
            
            # Compute face geometry
            verts = mesh.vertices[unique_verts]
            
            if n_unique == 4:
                # Quad - compute all 4 angles and aspect ratio
                edges = np.array([
                    np.linalg.norm(verts[(i+1)%4] - verts[i])
                    for i in range(4)
                ])
                
                if edges.min() > 1e-10:
                    face_aspect_ratios[fi] = edges.max() / edges.min()
                else:
                    face_aspect_ratios[fi] = 10.0  # Degenerate
                
                # Compute angles
                for i in range(4):
                    p0 = verts[(i - 1) % 4]
                    p1 = verts[i]
                    p2 = verts[(i + 1) % 4]
                    
                    e1 = p0 - p1
                    e2 = p2 - p1
                    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                    
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                        angle = np.arccos(cos_angle)
                        face_angles[fi, i] = angle
                        vertex_angle_sum[unique_verts[i]] += angle
                    else:
                        face_angles[fi, i] = np.pi / 2
                        
            elif n_unique == 3:
                # Triangle
                edges = np.array([
                    np.linalg.norm(verts[(i+1)%3] - verts[i])
                    for i in range(3)
                ])
                
                if edges.min() > 1e-10:
                    face_aspect_ratios[fi] = edges.max() / edges.min()
                else:
                    face_aspect_ratios[fi] = 10.0
                
                for i in range(3):
                    p0 = verts[(i - 1) % 3]
                    p1 = verts[i]
                    p2 = verts[(i + 1) % 3]
                    
                    e1 = p0 - p1
                    e2 = p2 - p1
                    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                    
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                        angle = np.arccos(cos_angle)
                        face_angles[fi, i] = angle
                        vertex_angle_sum[unique_verts[i]] += angle
                    else:
                        face_angles[fi, i] = np.pi / 3
        
        # Compute derived values
        boundary_edges = sum(1 for c in edge_count.values() if c == 1)
        non_manifold_edges = sum(1 for c in edge_count.values() if c > 2)
        
        # Valence histogram (using edge valence, not face count)
        valence_hist = {}
        for v in edge_valence:
            valence_hist[int(v)] = valence_hist.get(int(v), 0) + 1
        
        # Vertex curvature (angle deficit method)
        vertex_curvature = np.abs(2 * np.pi - vertex_angle_sum)
        
        return cls(
            num_vertices=n_verts,
            num_faces=n_faces,
            num_edges=len(edge_count),
            valence=edge_valence,  # Now correctly using edge valence
            valence_histogram=valence_hist,
            edge_count=edge_count,
            boundary_edges=boundary_edges,
            non_manifold_edges=non_manifold_edges,
            face_angles=face_angles,
            face_aspect_ratios=face_aspect_ratios,
            face_types=face_types,
            vertex_curvature=vertex_curvature,
        )


@dataclass
class PoleAnalysis:
    """Analysis of poles (non-valence-4 vertices) in the mesh.
    
    Based on topology best practices:
    - Poles should be placed in flat areas, not at deformation points
    - 3-poles and 5-poles are acceptable; 6+ poles are problematic
    - Fewer poles = cleaner subdivision
    """
    total_poles: int  # Count of all non-valence-4 vertices
    n3_poles: int  # 3-valence poles (E-poles)
    n5_poles: int  # 5-valence poles (N-poles)
    n6_plus_poles: int  # Problematic high-valence poles
    pole_ratio: float  # Ratio of poles to total vertices
    poles_in_high_curvature: int  # Poles in high-curvature (deformation) areas
    pole_placement_score: float  # 0-1, higher = better placement (away from curves)
    
    def to_dict(self) -> dict:
        return {
            "total_poles": self.total_poles,
            "n3_poles": self.n3_poles,
            "n5_poles": self.n5_poles,
            "n6_plus_poles": self.n6_plus_poles,
            "pole_ratio": self.pole_ratio,
            "poles_in_high_curvature": self.poles_in_high_curvature,
            "pole_placement_score": self.pole_placement_score,
        }


@dataclass
class EdgeLoopMetrics:
    """Metrics for edge loop quality.
    
    Good edge loops are continuous paths of edges that:
    - Follow natural contours of the form
    - Support areas that will deform
    - Enable easy selection and modification
    """
    avg_loop_length: float  # Average number of edges in detected loops
    loop_continuity_score: float  # 0-1, how many edges are part of clean loops
    loop_alignment_score: float  # 0-1, how well loops align with curvature direction
    
    def to_dict(self) -> dict:
        return {
            "avg_loop_length": self.avg_loop_length,
            "loop_continuity_score": self.loop_continuity_score,
            "loop_alignment_score": self.loop_alignment_score,
        }


@dataclass
class QuadQualityMetrics:
    """Metrics specific to quad quality."""
    aspect_ratio_mean: float  # 1.0 = perfect squares
    aspect_ratio_std: float
    angle_deviation_mean: float  # Deviation from 90 degrees (radians)
    angle_deviation_std: float
    valence_histogram: dict[int, int]  # Vertex valence distribution
    irregular_vertex_ratio: float  # Ratio of non-valence-4 vertices
    # New topology-informed metrics
    pole_analysis: Optional[PoleAnalysis] = None
    edge_loop_metrics: Optional[EdgeLoopMetrics] = None
    
    def to_dict(self) -> dict:
        result = {
            "aspect_ratio_mean": self.aspect_ratio_mean,
            "aspect_ratio_std": self.aspect_ratio_std,
            "angle_deviation_mean": self.angle_deviation_mean,
            "angle_deviation_std": self.angle_deviation_std,
            "valence_histogram": self.valence_histogram,
            "irregular_vertex_ratio": self.irregular_vertex_ratio,
        }
        if self.pole_analysis:
            result["pole_analysis"] = self.pole_analysis.to_dict()
        if self.edge_loop_metrics:
            result["edge_loop_metrics"] = self.edge_loop_metrics.to_dict()
        return result


@dataclass
class GeometricFidelityMetrics:
    """Metrics measuring how well retopo matches original."""
    hausdorff_distance: float  # Maximum surface deviation
    mean_distance: float  # Average surface deviation
    rms_distance: float  # Root mean square distance
    normal_deviation_mean: float  # Average normal difference (radians)
    coverage: float  # What fraction of original is covered
    
    def to_dict(self) -> dict:
        return {
            "hausdorff_distance": self.hausdorff_distance,
            "mean_distance": self.mean_distance,
            "rms_distance": self.rms_distance,
            "normal_deviation_mean": self.normal_deviation_mean,
            "coverage": self.coverage,
        }


@dataclass
class TopologyMetrics:
    """Metrics about mesh topology."""
    num_vertices: int
    num_faces: int
    num_edges: int
    euler_characteristic: int
    num_boundaries: int
    is_manifold: bool
    genus: int
    # New topology-informed scores
    pole_quality_score: float = 100.0  # 0-100, penalizes bad pole placement
    edge_flow_score: float = 100.0  # 0-100, rewards good edge loop continuity
    
    def to_dict(self) -> dict:
        return {
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "num_edges": self.num_edges,
            "euler_characteristic": self.euler_characteristic,
            "num_boundaries": self.num_boundaries,
            "is_manifold": self.is_manifold,
            "genus": self.genus,
            "pole_quality_score": self.pole_quality_score,
            "edge_flow_score": self.edge_flow_score,
        }


@dataclass
class RetopologyScore:
    """Combined score for retopology quality."""
    quad_quality: QuadQualityMetrics
    geometric_fidelity: GeometricFidelityMetrics
    topology: TopologyMetrics
    visual_quality: Optional[VisualQualityMetrics] = None
    
    # Composite scores (0-100)
    overall_score: float = 0.0
    quad_score: float = 0.0
    fidelity_score: float = 0.0
    topology_score: float = 0.0
    visual_score: float = 0.0
    
    # Weights used for scoring
    weights: dict = field(default_factory=lambda: {
        "quad": 0.30,
        "fidelity": 0.30,
        "topology": 0.15,
        "visual": 0.25
    })
    
    def compute_scores(self, reference_diagonal: float = 1.0):
        """Compute composite scores from individual metrics."""
        # Quad quality score (0-100)
        # Aspect ratio: 1.0 is ideal, up to 2.0 is acceptable for artistic meshes
        # Score = 100 when AR=1, ~75 when AR=1.5, ~50 when AR=2
        ar_score = max(0, 100 - (self.quad_quality.aspect_ratio_mean - 1) * 80)
        ar_score = min(100, ar_score)
        
        # Angle deviation: convert to degrees for intuitive scaling
        # 0 deg is ideal, up to 15 deg is good, 30 deg is poor
        angle_deg = np.degrees(self.quad_quality.angle_deviation_mean)
        angle_score = max(0, 100 - angle_deg * 3.33)  # 0 at 30 deg deviation
        
        # Valence: irregular vertices are acceptable at feature lines
        # Use gentler penalty - 50% irregular is still score of 50
        valence_score = max(0, 100 - self.quad_quality.irregular_vertex_ratio * 100)
        
        # Weight angle and aspect more than valence for practical meshes
        self.quad_score = (ar_score * 0.35 + angle_score * 0.45 + valence_score * 0.20)
        
        # Geometric fidelity score (0-100)
        # Normalize distances by reference diagonal
        # Hausdorff: 0% = 100 score, 3% of diagonal = 50, 6% = 0
        hausdorff_pct = (self.geometric_fidelity.hausdorff_distance / reference_diagonal) * 100
        hausdorff_score = max(0, min(100, 100 - hausdorff_pct * 16.67))
        
        # Mean distance: 0% = 100, 2% of diagonal = 0
        mean_pct = (self.geometric_fidelity.mean_distance / reference_diagonal) * 100
        mean_score = max(0, min(100, 100 - mean_pct * 50))
        
        # Normal deviation in degrees - decimation often changes normals significantly
        # 0 deg = 100, 30 deg = 50, 60 deg = 0 (more lenient for low-poly)
        normal_deg = np.degrees(self.geometric_fidelity.normal_deviation_mean)
        normal_score = max(0, min(100, 100 - normal_deg * 1.67))
        
        # Coverage is important - 80%+ is good for simplified meshes
        # 100% = 100, 80% = 80, 50% = 50
        coverage_score = self.geometric_fidelity.coverage * 100
        
        self.fidelity_score = (hausdorff_score * 0.30 + mean_score * 0.30 + 
                               normal_score * 0.15 + coverage_score * 0.25)
        
        # Topology score (0-100) - now incorporates pole and edge flow quality
        manifold_score = 100 if self.topology.is_manifold else 0
        boundary_score = 100 if self.topology.num_boundaries <= 1 else max(0, 100 - self.topology.num_boundaries * 10)
        
        # Include new topology-informed metrics (pole placement & edge flow)
        pole_score = self.topology.pole_quality_score
        edge_flow_score = self.topology.edge_flow_score
        
        # Weight: manifold is critical, then pole placement, then boundaries, then edge flow
        self.topology_score = (
            manifold_score * 0.40 +  # Manifold is most critical
            pole_score * 0.25 +       # Good pole placement matters for subdivision
            boundary_score * 0.20 +   # Boundaries affect usability
            edge_flow_score * 0.15    # Edge loops help deformation
        )
        
        # Visual score (0-100)
        if self.visual_quality is not None:
            # Weights for visual components
            self.visual_score = (
                self.visual_quality.shading_smoothness * 35 +
                self.visual_quality.edge_visibility * 30 +
                self.visual_quality.silhouette_quality * 20 +
                self.visual_quality.render_consistency * 15
            )
        else:
            # Default to neutral if no visual eval
            self.visual_score = 50.0
        
        # Overall weighted score
        self.overall_score = (
            self.weights["quad"] * self.quad_score +
            self.weights["fidelity"] * self.fidelity_score +
            self.weights["topology"] * self.topology_score +
            self.weights["visual"] * self.visual_score
        )
    
    def to_dict(self) -> dict:
        result = {
            "overall_score": self.overall_score,
            "quad_score": self.quad_score,
            "fidelity_score": self.fidelity_score,
            "topology_score": self.topology_score,
            "visual_score": self.visual_score,
            "quad_quality": self.quad_quality.to_dict(),
            "geometric_fidelity": self.geometric_fidelity.to_dict(),
            "topology": self.topology.to_dict(),
        }
        if self.visual_quality is not None:
            result["visual_quality"] = self.visual_quality.to_dict()
        return result
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"=== Retopology Quality Score: {self.overall_score:.1f}/100 ===",
            f"  Quad Quality:      {self.quad_score:.1f}/100",
            f"    - Aspect ratio:  {self.quad_quality.aspect_ratio_mean:.3f} (ideal: 1.0)",
            f"    - Angle dev:     {np.degrees(self.quad_quality.angle_deviation_mean):.1f}° (ideal: 0°)",
            f"    - Irregular:     {self.quad_quality.irregular_vertex_ratio*100:.1f}%",
        ]
        # Add pole analysis if available
        if self.quad_quality.pole_analysis:
            pa = self.quad_quality.pole_analysis
            lines.extend([
                f"    - Poles:         {pa.total_poles} ({pa.n3_poles} 3-poles, {pa.n5_poles} 5-poles)",
                f"    - Pole placement: {pa.pole_placement_score*100:.1f}% (away from curves)",
            ])
        lines.extend([
            f"  Geometric Fidelity: {self.fidelity_score:.1f}/100",
            f"    - Hausdorff:     {self.geometric_fidelity.hausdorff_distance:.6f}",
            f"    - Mean dist:     {self.geometric_fidelity.mean_distance:.6f}",
            f"    - Coverage:      {self.geometric_fidelity.coverage*100:.1f}%",
            f"  Topology:          {self.topology_score:.1f}/100",
            f"    - Vertices:      {self.topology.num_vertices}",
            f"    - Faces:         {self.topology.num_faces}",
            f"    - Manifold:      {self.topology.is_manifold}",
            f"    - Pole quality:  {self.topology.pole_quality_score:.1f}/100",
            f"    - Edge flow:     {self.topology.edge_flow_score:.1f}/100",
            f"  Visual Quality:    {self.visual_score:.1f}/100",
        ])
        if self.visual_quality is not None:
            lines.extend([
                f"    - Shading:       {self.visual_quality.shading_smoothness*100:.1f}%",
                f"    - Edge qual:     {self.visual_quality.edge_visibility*100:.1f}%",
                f"    - Silhouette:    {self.visual_quality.silhouette_quality*100:.1f}%",
            ])
        return "\n".join(lines)


class MeshEvaluator:
    """
    Comprehensive mesh quality evaluator.
    
    Computes all metrics needed to assess retopology quality.
    
    Performance optimized:
    - Uses MeshTopologyCache for single-pass topology computation
    - Vectorized metric calculations with NumPy
    - Shared cache across all metric functions
    """
    
    def __init__(self, sample_count: int = 10000, enable_visual: bool = True):
        self.sample_count = sample_count
        self.enable_visual = enable_visual
        self._visual_evaluator = None
        self._topo_cache: Optional[MeshTopologyCache] = None
    
    @property
    def visual_evaluator(self) -> VisualEvaluator:
        """Lazy-load visual evaluator."""
        if self._visual_evaluator is None:
            self._visual_evaluator = VisualEvaluator()
        return self._visual_evaluator
    
    def evaluate(
        self,
        retopo_mesh: Mesh,
        original_mesh: Optional[Mesh] = None
    ) -> RetopologyScore:
        """
        Evaluate retopology quality.
        
        Args:
            retopo_mesh: The retopologized mesh
            original_mesh: Original high-poly mesh (for fidelity metrics)
            
        Returns:
            RetopologyScore with all metrics
        """
        start_time = time.time()
        
        # Build topology cache ONCE for all metrics
        t0 = time.time()
        self._topo_cache = MeshTopologyCache.from_mesh(retopo_mesh)
        logger.debug(f"Topology cache built: {time.time() - t0:.3f}s")
        
        # Compute quad quality using cache
        t0 = time.time()
        quad_quality = self._compute_quad_quality_cached(retopo_mesh)
        logger.debug(f"Quad quality computation: {time.time() - t0:.3f}s")
        
        # Compute geometric fidelity (if original provided)
        t0 = time.time()
        if original_mesh is not None:
            geometric_fidelity = self._compute_fidelity(retopo_mesh, original_mesh)
            reference_diagonal = original_mesh.diagonal
            logger.debug(f"Fidelity computation: {time.time() - t0:.3f}s")
        else:
            geometric_fidelity = GeometricFidelityMetrics(
                hausdorff_distance=0,
                mean_distance=0,
                rms_distance=0,
                normal_deviation_mean=0,
                coverage=1.0
            )
            reference_diagonal = retopo_mesh.diagonal
        
        # Compute topology metrics using cache
        t0 = time.time()
        topology = self._compute_topology_cached(retopo_mesh)
        logger.debug(f"Topology computation: {time.time() - t0:.3f}s")
        
        # Compute visual quality metrics
        visual_quality = None
        if self.enable_visual:
            t0 = time.time()
            visual_quality = self.visual_evaluator.evaluate(retopo_mesh, original_mesh)
            logger.debug(f"Visual evaluation: {time.time() - t0:.3f}s")
        
        # Create score and compute composites
        score = RetopologyScore(
            quad_quality=quad_quality,
            geometric_fidelity=geometric_fidelity,
            topology=topology,
            visual_quality=visual_quality,
        )
        score.compute_scores(reference_diagonal)
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation complete in {total_time:.2f}s, score: {score.overall_score:.1f}")
        
        # Clear cache
        self._topo_cache = None
        
        return score
    
    def _compute_quad_quality_cached(self, mesh: Mesh) -> QuadQualityMetrics:
        """Compute quad quality metrics using pre-computed cache."""
        cache = self._topo_cache
        
        # Filter to valid faces
        quad_mask = cache.face_types == 4
        tri_mask = cache.face_types == 3
        
        # Get aspect ratios for valid faces
        valid_mask = (cache.face_types == 4) | (cache.face_types == 3)
        aspect_ratios = cache.face_aspect_ratios[valid_mask]
        
        # Compute angle deviations
        # For quads: deviation from 90 degrees (π/2)
        # For tris: deviation from 60 degrees (π/3)
        angle_deviations = []
        
        if np.any(quad_mask):
            quad_angles = cache.face_angles[quad_mask, :4]
            quad_angle_devs = np.abs(quad_angles - np.pi / 2)
            angle_deviations.extend(quad_angle_devs.flatten())
        
        if np.any(tri_mask):
            tri_angles = cache.face_angles[tri_mask, :3]
            tri_angle_devs = np.abs(tri_angles - np.pi / 3)
            angle_deviations.extend(tri_angle_devs.flatten())
        
        if not angle_deviations:
            angle_deviations = [0.0]
        
        angle_deviations = np.array(angle_deviations)
        
        # Irregular vertex ratio
        target_valence = 4 if mesh.is_quad else 6
        irregular = np.sum(cache.valence != target_valence) / max(1, len(cache.valence))
        
        # Pole analysis using cached curvature
        pole_analysis = self._compute_pole_analysis_cached(mesh)
        
        # Edge loop metrics
        edge_loop_metrics = self._compute_edge_loop_metrics_cached(mesh)
        
        return QuadQualityMetrics(
            aspect_ratio_mean=float(np.mean(aspect_ratios)) if len(aspect_ratios) > 0 else 1.0,
            aspect_ratio_std=float(np.std(aspect_ratios)) if len(aspect_ratios) > 0 else 0.0,
            angle_deviation_mean=float(np.mean(angle_deviations)),
            angle_deviation_std=float(np.std(angle_deviations)),
            valence_histogram=cache.valence_histogram,
            irregular_vertex_ratio=float(irregular),
            pole_analysis=pole_analysis,
            edge_loop_metrics=edge_loop_metrics,
        )
    
    def _compute_topology_cached(self, mesh: Mesh) -> TopologyMetrics:
        """Compute topology metrics using pre-computed cache."""
        cache = self._topo_cache
        
        # Euler characteristic
        euler = cache.num_vertices - cache.num_edges + cache.num_faces
        
        # Manifold check
        is_manifold = cache.non_manifold_edges == 0
        
        # Boundaries
        n_boundaries = 1 if cache.boundary_edges > 0 else 0
        
        # Genus
        genus = max(0, (2 - euler - n_boundaries) // 2)
        
        # Pole quality score
        n3_poles = int(np.sum(cache.valence == 3))
        n5_poles = int(np.sum(cache.valence == 5))
        n6_plus = int(np.sum(cache.valence >= 6))
        
        if cache.num_vertices > 0:
            n6_plus_ratio = n6_plus / cache.num_vertices
            total_poles = n3_poles + n5_poles + n6_plus
            pole_density = total_poles / cache.num_vertices
            pole_penalty = (n6_plus_ratio * 50) + max(0, (pole_density - 0.5) * 20)
        else:
            pole_penalty = 0
        
        pole_quality_score = max(0.0, min(100.0, 100.0 - pole_penalty))
        
        # Edge flow score
        edge_flow_score = self._compute_edge_flow_score_cached(mesh)
        
        return TopologyMetrics(
            num_vertices=cache.num_vertices,
            num_faces=cache.num_faces,
            num_edges=cache.num_edges,
            euler_characteristic=euler,
            num_boundaries=n_boundaries,
            is_manifold=is_manifold,
            genus=genus,
            pole_quality_score=pole_quality_score,
            edge_flow_score=edge_flow_score,
        )
    
    def _compute_pole_analysis_cached(self, mesh: Mesh) -> PoleAnalysis:
        """Compute pole analysis using cached data."""
        cache = self._topo_cache
        
        n3_poles = int(np.sum(cache.valence == 3))
        n5_poles = int(np.sum(cache.valence == 5))
        n6_plus = int(np.sum(cache.valence >= 6))
        total_poles = n3_poles + n5_poles + n6_plus
        
        pole_ratio = total_poles / max(1, cache.num_vertices)
        
        # Find poles and check curvature
        pole_indices = np.where((cache.valence != 4) & (cache.valence > 0))[0]
        
        if len(cache.vertex_curvature) > 0 and len(pole_indices) > 0:
            curvature_threshold = np.percentile(cache.vertex_curvature, 75)
            poles_in_high_curve = int(np.sum(cache.vertex_curvature[pole_indices] > curvature_threshold))
        else:
            poles_in_high_curve = 0
        
        if total_poles > 0:
            pole_placement_score = 1.0 - (poles_in_high_curve / total_poles)
        else:
            pole_placement_score = 1.0
        
        return PoleAnalysis(
            total_poles=total_poles,
            n3_poles=n3_poles,
            n5_poles=n5_poles,
            n6_plus_poles=n6_plus,
            pole_ratio=pole_ratio,
            poles_in_high_curvature=poles_in_high_curve,
            pole_placement_score=pole_placement_score,
        )
    
    def _compute_edge_loop_metrics_cached(self, mesh: Mesh) -> EdgeLoopMetrics:
        """Compute edge loop metrics using cached data."""
        cache = self._topo_cache
        
        if not mesh.is_quad:
            return EdgeLoopMetrics(
                avg_loop_length=0.0,
                loop_continuity_score=0.5,
                loop_alignment_score=0.5,
            )
        
        # Count interior edges (those that can form loops)
        interior_edges = sum(1 for c in cache.edge_count.values() if c == 2)
        total_edges = len(cache.edge_count)
        
        if total_edges == 0:
            return EdgeLoopMetrics(
                avg_loop_length=0.0,
                loop_continuity_score=0.0,
                loop_alignment_score=0.0,
            )
        
        loop_continuity = interior_edges / total_edges
        avg_loop_length = 4.0 * loop_continuity * 10
        
        return EdgeLoopMetrics(
            avg_loop_length=avg_loop_length,
            loop_continuity_score=loop_continuity,
            loop_alignment_score=loop_continuity,
        )
    
    def _compute_edge_flow_score_cached(self, mesh: Mesh) -> float:
        """Compute edge flow score using cached data."""
        cache = self._topo_cache
        
        if not mesh.is_quad:
            return 75.0
        
        interior_edges = sum(1 for c in cache.edge_count.values() if c == 2)
        total_edges = len(cache.edge_count)
        
        if total_edges == 0:
            return 50.0
        
        return (interior_edges / total_edges) * 100.0
    
    def _compute_fidelity(
        self, 
        retopo: Mesh, 
        original: Mesh
    ) -> GeometricFidelityMetrics:
        """Compute geometric fidelity metrics."""
        # Sample points on both meshes
        retopo_samples = self._sample_mesh(retopo, self.sample_count)
        original_samples = self._sample_mesh(original, self.sample_count)
        
        # Build KD-trees
        retopo_tree = cKDTree(retopo_samples)
        original_tree = cKDTree(original_samples)
        
        # Distance from retopo to original
        dist_r2o, _ = original_tree.query(retopo_samples)
        
        # Distance from original to retopo
        dist_o2r, _ = retopo_tree.query(original_samples)
        
        # Hausdorff distance (symmetric)
        hausdorff = max(dist_r2o.max(), dist_o2r.max())
        
        # Mean and RMS distances
        all_distances = np.concatenate([dist_r2o, dist_o2r])
        mean_dist = float(np.mean(all_distances))
        rms_dist = float(np.sqrt(np.mean(all_distances**2)))
        
        # Normal deviation (simplified - compare at closest points)
        if retopo.normals is None:
            retopo.compute_normals()
        if original.normals is None:
            original.compute_normals()
        
        # Sample normal deviation at vertex positions
        _, closest_idx = original_tree.query(retopo.vertices)
        closest_original_verts = np.clip(closest_idx, 0, len(original.vertices)-1)
        
        # Compare normals
        normal_dots = np.sum(
            retopo.normals * original.normals[closest_original_verts % len(original.normals)],
            axis=1
        )
        normal_dots = np.clip(normal_dots, -1, 1)
        normal_deviations = np.arccos(np.abs(normal_dots))
        normal_dev_mean = float(np.mean(normal_deviations))
        
        # Coverage: what fraction of original surface is within threshold distance
        threshold = original.diagonal * 0.01  # 1% of diagonal
        coverage = float(np.mean(dist_o2r < threshold))
        
        return GeometricFidelityMetrics(
            hausdorff_distance=float(hausdorff),
            mean_distance=mean_dist,
            rms_distance=rms_dist,
            normal_deviation_mean=normal_dev_mean,
            coverage=coverage
        )
    
    def _sample_mesh(self, mesh: Mesh, n_samples: int) -> np.ndarray:
        """Uniformly sample points on mesh surface."""
        if not mesh.is_triangular:
            mesh = mesh.triangulate()
        
        # Compute face areas
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        
        face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        face_probs = face_areas / face_areas.sum()
        
        # Sample faces
        sampled_faces = np.random.choice(len(mesh.faces), size=n_samples, p=face_probs)
        
        # Random barycentric coordinates
        r1 = np.random.random(n_samples)
        r2 = np.random.random(n_samples)
        sqrt_r1 = np.sqrt(r1)
        
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2
        
        # Compute positions
        samples = (
            u[:, np.newaxis] * mesh.vertices[mesh.faces[sampled_faces, 0]] +
            v[:, np.newaxis] * mesh.vertices[mesh.faces[sampled_faces, 1]] +
            w[:, np.newaxis] * mesh.vertices[mesh.faces[sampled_faces, 2]]
        )
        
        return samples


def evaluate_retopology(
    retopo_mesh: Mesh,
    original_mesh: Optional[Mesh] = None
) -> RetopologyScore:
    """Convenience function to evaluate retopology."""
    evaluator = MeshEvaluator()
    return evaluator.evaluate(retopo_mesh, original_mesh)
