"""
Quality metrics for retopology evaluation.

Provides comprehensive metrics to objectively measure retopology quality,
enabling data-driven iteration and improvement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree

from neurotopo.core.mesh import Mesh
from neurotopo.evaluation.visual import VisualEvaluator, VisualQualityMetrics

logger = logging.getLogger("neurotopo.evaluation")


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
    """
    
    def __init__(self, sample_count: int = 10000, enable_visual: bool = True):
        self.sample_count = sample_count
        self.enable_visual = enable_visual
        self._visual_evaluator = None
    
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
        
        # Compute quad quality
        t0 = time.time()
        quad_quality = self._compute_quad_quality(retopo_mesh)
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
        
        # Compute topology metrics
        t0 = time.time()
        topology = self._compute_topology(retopo_mesh)
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
        
        return score
    
    def _compute_quad_quality(self, mesh: Mesh) -> QuadQualityMetrics:
        """Compute quad-specific quality metrics."""
        aspect_ratios = []
        angle_deviations = []
        quad_count = 0
        tri_count = 0
        
        for face in mesh.faces:
            unique_verts = len(set(face))
            vertices = mesh.vertices[face]
            
            # Handle degenerate quads (stored as [a,b,c,c]) - skip them
            # They are artifacts from triangle pairing that failed and shouldn't
            # contribute to quality metrics
            if len(face) == 4 and unique_verts == 4:
                # True quad - evaluate as quad
                quad_count += 1
                ar = self._quad_aspect_ratio(vertices)
                angles = self._quad_angles(vertices)
                # For quads, ideal angle is 90 degrees
                angle_deviations.extend([abs(a - np.pi/2) for a in angles])
                aspect_ratios.append(ar)
            elif len(face) == 4 and unique_verts == 3:
                # Degenerate quad (triangle with repeated vertex)
                # Skip these - they're artifacts, not real geometry
                # The remesher should ideally not produce these
                continue
            elif len(face) == 3:
                # Regular triangle
                tri_count += 1
                ar = self._triangle_aspect_ratio(vertices)
                angles = self._triangle_angles(vertices)
                angle_deviations.extend([abs(a - np.pi/3) for a in angles])
                aspect_ratios.append(ar)
            else:
                # Skip invalid faces
                continue
        
        # Compute valence histogram (use unique vertices per face)
        valence = np.zeros(mesh.num_vertices, dtype=int)
        for face in mesh.faces:
            for vi in set(face):  # Use set to handle degenerate quads
                valence[vi] += 1
        
        valence_hist = {}
        for v in valence:
            valence_hist[int(v)] = valence_hist.get(int(v), 0) + 1
        
        # Irregular vertex ratio (not valence 4 for interior, not 2-4 for boundary)
        target_valence = 4 if mesh.is_quad else 6
        irregular = np.sum(valence != target_valence) / len(valence)
        
        # Compute pole analysis (topology-informed metric)
        pole_analysis = self._compute_pole_analysis(mesh, valence)
        
        # Compute edge loop metrics
        edge_loop_metrics = self._compute_edge_loop_metrics(mesh)
        
        return QuadQualityMetrics(
            aspect_ratio_mean=float(np.mean(aspect_ratios)),
            aspect_ratio_std=float(np.std(aspect_ratios)),
            angle_deviation_mean=float(np.mean(angle_deviations)),
            angle_deviation_std=float(np.std(angle_deviations)),
            valence_histogram=valence_hist,
            irregular_vertex_ratio=float(irregular),
            pole_analysis=pole_analysis,
            edge_loop_metrics=edge_loop_metrics
        )
    
    def _quad_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Compute aspect ratio of a quad."""
        if len(vertices) != 4:
            return 1.0
        
        # Edge lengths
        edges = [
            np.linalg.norm(vertices[(i+1)%4] - vertices[i])
            for i in range(4)
        ]
        
        # Aspect ratio: max/min edge length
        if min(edges) < 1e-10:
            return 10.0  # Degenerate
        
        return max(edges) / min(edges)
    
    def _quad_angles(self, vertices: np.ndarray) -> list[float]:
        """Compute interior angles of a quad."""
        angles = []
        for i in range(4):
            p0 = vertices[(i-1) % 4]
            p1 = vertices[i]
            p2 = vertices[(i+1) % 4]
            
            e1 = p0 - p1
            e2 = p2 - p1
            
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if n1 < 1e-10 or n2 < 1e-10:
                angles.append(np.pi/2)
            else:
                cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_angle))
        
        return angles
    
    def _triangle_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Compute aspect ratio of a triangle."""
        edges = [
            np.linalg.norm(vertices[(i+1)%3] - vertices[i])
            for i in range(3)
        ]
        
        if min(edges) < 1e-10:
            return 10.0
        
        return max(edges) / min(edges)
    
    def _triangle_angles(self, vertices: np.ndarray) -> list[float]:
        """Compute interior angles of a triangle."""
        angles = []
        for i in range(3):
            p0 = vertices[(i-1) % 3]
            p1 = vertices[i]
            p2 = vertices[(i+1) % 3]
            
            e1 = p0 - p1
            e2 = p2 - p1
            
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if n1 < 1e-10 or n2 < 1e-10:
                angles.append(np.pi/3)
            else:
                cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_angle))
        
        return angles
    
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
    
    def _compute_topology(self, mesh: Mesh) -> TopologyMetrics:
        """Compute topological metrics."""
        n_vertices = mesh.num_vertices
        n_faces = mesh.num_faces
        
        # Helper to get unique vertices in a face (handles degenerate quads)
        def get_face_verts(face):
            """Get unique vertices in face preserving order."""
            seen = set()
            result = []
            for v in face:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
            return result
        
        # Count edges, properly handling degenerate quads
        edges = set()
        for face in mesh.faces:
            verts = get_face_verts(face)
            n = len(verts)
            if n < 3:
                continue  # Skip degenerate faces
            for i in range(n):
                v0, v1 = verts[i], verts[(i + 1) % n]
                if v0 != v1:  # Skip self-loops
                    edges.add((min(v0, v1), max(v0, v1)))
        n_edges = len(edges)
        
        # Euler characteristic: V - E + F
        euler = n_vertices - n_edges + n_faces
        
        # Check manifold (each edge should have exactly 2 adjacent faces)
        edge_count = {}
        for face in mesh.faces:
            verts = get_face_verts(face)
            n = len(verts)
            if n < 3:
                continue  # Skip degenerate faces
            for i in range(n):
                v0, v1 = verts[i], verts[(i + 1) % n]
                if v0 != v1:  # Skip self-loops
                    edge = (min(v0, v1), max(v0, v1))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary_edges = sum(1 for c in edge_count.values() if c == 1)
        non_manifold_edges = sum(1 for c in edge_count.values() if c > 2)
        is_manifold = non_manifold_edges == 0
        
        # Count boundary loops (simplified)
        n_boundaries = 1 if boundary_edges > 0 else 0
        
        # Genus: g = (2 - euler - boundaries) / 2
        genus = max(0, (2 - euler - n_boundaries) // 2)
        
        # Compute pole quality score (0-100)
        # Based on valence distribution - penalize high-valence poles heavily
        valence = np.zeros(mesh.num_vertices, dtype=int)
        for face in mesh.faces:
            for vi in set(face):
                valence[vi] += 1
        
        n3_poles = np.sum(valence == 3)
        n5_poles = np.sum(valence == 5)
        n6_plus = np.sum(valence >= 6)
        total_poles = n3_poles + n5_poles + n6_plus
        
        # Score: penalize high-valence poles, 3/5 poles are acceptable
        # 6+ poles are problematic (cause subdivision artifacts)
        # Use proportion-based penalty:
        # - 6+ poles: up to 50 points penalty based on their ratio to total vertices
        # - Total poles: small penalty for high pole density
        if n_vertices > 0:
            n6_plus_ratio = n6_plus / n_vertices  # 0 to 1
            pole_density = total_poles / n_vertices  # typically 0.3-0.9
            
            # 50% of vertices being 6+ poles = max 50 point penalty
            # High pole density (>50%) adds small penalty
            pole_penalty = (n6_plus_ratio * 50) + max(0, (pole_density - 0.5) * 20)
        else:
            pole_penalty = 0
        
        pole_quality_score = max(0.0, min(100.0, 100.0 - pole_penalty))
        
        # Compute edge flow score based on edge loop continuity
        # Better loops = higher score
        edge_flow_score = self._compute_edge_flow_score(mesh, edge_count)
        
        return TopologyMetrics(
            num_vertices=n_vertices,
            num_faces=n_faces,
            num_edges=n_edges,
            euler_characteristic=euler,
            num_boundaries=n_boundaries,
            is_manifold=is_manifold,
            genus=genus,
            pole_quality_score=pole_quality_score,
            edge_flow_score=edge_flow_score
        )
    
    def _compute_pole_analysis(self, mesh: Mesh, valence: np.ndarray) -> PoleAnalysis:
        """
        Analyze pole (non-valence-4 vertex) distribution and placement.
        
        Topology best practice: poles should be in flat areas, not deformation zones.
        """
        n_vertices = len(valence)
        
        # Count poles by type
        n3_poles = int(np.sum(valence == 3))
        n5_poles = int(np.sum(valence == 5))
        n6_plus = int(np.sum(valence >= 6))
        total_poles = n3_poles + n5_poles + n6_plus
        
        pole_ratio = total_poles / max(1, n_vertices)
        
        # Identify poles
        pole_indices = np.where((valence != 4) & (valence > 0))[0]
        
        # Compute curvature at each vertex to assess pole placement
        # High curvature = deformation area = bad for poles
        curvatures = self._estimate_vertex_curvature(mesh)
        
        # Count poles in high-curvature regions (top 25% of curvature)
        if len(curvatures) > 0 and len(pole_indices) > 0:
            curvature_threshold = np.percentile(curvatures, 75)
            poles_in_high_curve = int(np.sum(curvatures[pole_indices] > curvature_threshold))
        else:
            poles_in_high_curve = 0
        
        # Pole placement score: 1.0 if all poles in flat areas, 0.0 if all in curved areas
        if total_poles > 0:
            pole_placement_score = 1.0 - (poles_in_high_curve / total_poles)
        else:
            pole_placement_score = 1.0  # No poles = perfect
        
        return PoleAnalysis(
            total_poles=total_poles,
            n3_poles=n3_poles,
            n5_poles=n5_poles,
            n6_plus_poles=n6_plus,
            pole_ratio=pole_ratio,
            poles_in_high_curvature=poles_in_high_curve,
            pole_placement_score=pole_placement_score
        )
    
    def _estimate_vertex_curvature(self, mesh: Mesh) -> np.ndarray:
        """
        Estimate discrete curvature at each vertex.
        
        Uses angle deficit method: curvature = 2π - sum of angles at vertex.
        Higher absolute values = more curved = worse for poles.
        """
        curvatures = np.zeros(mesh.num_vertices)
        angle_sums = np.zeros(mesh.num_vertices)
        
        for face in mesh.faces:
            unique_verts = list(set(face))
            if len(unique_verts) < 3:
                continue
            
            vertices = mesh.vertices[unique_verts]
            n = len(unique_verts)
            
            for i in range(n):
                vi = unique_verts[i]
                p0 = vertices[(i - 1) % n]
                p1 = vertices[i]
                p2 = vertices[(i + 1) % n]
                
                e1 = p0 - p1
                e2 = p2 - p1
                n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                    angle_sums[vi] += np.arccos(cos_angle)
        
        # Angle deficit: for interior vertex, ideal is 2π
        curvatures = np.abs(2 * np.pi - angle_sums)
        
        return curvatures
    
    def _compute_edge_loop_metrics(self, mesh: Mesh) -> EdgeLoopMetrics:
        """
        Compute edge loop quality metrics.
        
        Good edge loops:
        - Form continuous paths around the mesh
        - Follow curvature directions
        - Enable easy ring selection
        """
        # Build edge-to-face and vertex-to-edge maps
        edge_faces = {}
        vertex_edges = {}
        
        for fi, face in enumerate(mesh.faces):
            unique_verts = list(set(face))
            n = len(unique_verts)
            if n < 3:
                continue
            
            for i in range(n):
                v0, v1 = unique_verts[i], unique_verts[(i + 1) % n]
                if v0 == v1:
                    continue
                edge = (min(v0, v1), max(v0, v1))
                
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(fi)
                
                for v in [v0, v1]:
                    if v not in vertex_edges:
                        vertex_edges[v] = set()
                    vertex_edges[v].add(edge)
        
        # For quad meshes, try to trace edge loops
        # An edge loop in quads follows edges that cross opposite sides of each quad
        if not mesh.is_quad:
            # For triangle meshes, edge loops are less defined
            return EdgeLoopMetrics(
                avg_loop_length=0.0,
                loop_continuity_score=0.5,  # Neutral for tri meshes
                loop_alignment_score=0.5
            )
        
        # Count how many edges are part of "continuable" loops
        # An edge is continuable if it has exactly one edge continuing on each end
        continuable_edges = 0
        total_interior_edges = 0
        
        for edge, faces in edge_faces.items():
            if len(faces) != 2:
                continue  # Skip boundary edges
            
            total_interior_edges += 1
            
            # Check if this edge can continue through each adjacent face
            v0, v1 = edge
            can_continue = True
            
            for fi in faces:
                face = mesh.faces[fi]
                unique_verts = list(set(face))
                if len(unique_verts) != 4:
                    can_continue = False
                    break
                
                # In a quad, an edge loop crosses to the opposite edge
                # Check if opposite edge exists
                idx0 = unique_verts.index(v0) if v0 in unique_verts else -1
                idx1 = unique_verts.index(v1) if v1 in unique_verts else -1
                
                if idx0 < 0 or idx1 < 0:
                    can_continue = False
                    break
            
            if can_continue:
                continuable_edges += 1
        
        # Loop continuity: ratio of edges that can form loops
        if total_interior_edges > 0:
            loop_continuity = continuable_edges / total_interior_edges
        else:
            loop_continuity = 0.0
        
        # Simplified loop length estimation (would need full tracing for accuracy)
        avg_loop_length = 4.0 * loop_continuity * 10  # Rough estimate
        
        return EdgeLoopMetrics(
            avg_loop_length=avg_loop_length,
            loop_continuity_score=loop_continuity,
            loop_alignment_score=loop_continuity  # Simplified: use same as continuity
        )
    
    def _compute_edge_flow_score(self, mesh: Mesh, edge_count: dict) -> float:
        """
        Compute edge flow quality score (0-100).
        
        Better edge flow = more edges can form continuous loops.
        """
        if not mesh.is_quad:
            return 75.0  # Neutral for triangle meshes
        
        # Count edges by their "loop potential"
        # Interior edges (count=2) have loop potential
        # Boundary edges (count=1) don't
        interior_edges = sum(1 for c in edge_count.values() if c == 2)
        total_edges = len(edge_count)
        
        if total_edges == 0:
            return 50.0
        
        # Higher ratio of interior edges = better edge flow
        interior_ratio = interior_edges / total_edges
        
        # Scale to 0-100
        return interior_ratio * 100.0


def evaluate_retopology(
    retopo_mesh: Mesh,
    original_mesh: Optional[Mesh] = None
) -> RetopologyScore:
    """Convenience function to evaluate retopology."""
    evaluator = MeshEvaluator()
    return evaluator.evaluate(retopo_mesh, original_mesh)
