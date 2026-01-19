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

from meshretopo.core.mesh import Mesh

logger = logging.getLogger("meshretopo.evaluation")


@dataclass
class QuadQualityMetrics:
    """Metrics specific to quad quality."""
    aspect_ratio_mean: float  # 1.0 = perfect squares
    aspect_ratio_std: float
    angle_deviation_mean: float  # Deviation from 90 degrees (radians)
    angle_deviation_std: float
    valence_histogram: dict[int, int]  # Vertex valence distribution
    irregular_vertex_ratio: float  # Ratio of non-valence-4 vertices
    
    def to_dict(self) -> dict:
        return {
            "aspect_ratio_mean": self.aspect_ratio_mean,
            "aspect_ratio_std": self.aspect_ratio_std,
            "angle_deviation_mean": self.angle_deviation_mean,
            "angle_deviation_std": self.angle_deviation_std,
            "valence_histogram": self.valence_histogram,
            "irregular_vertex_ratio": self.irregular_vertex_ratio,
        }


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
    
    def to_dict(self) -> dict:
        return {
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "num_edges": self.num_edges,
            "euler_characteristic": self.euler_characteristic,
            "num_boundaries": self.num_boundaries,
            "is_manifold": self.is_manifold,
            "genus": self.genus,
        }


@dataclass
class RetopologyScore:
    """Combined score for retopology quality."""
    quad_quality: QuadQualityMetrics
    geometric_fidelity: GeometricFidelityMetrics
    topology: TopologyMetrics
    
    # Composite scores (0-100)
    overall_score: float = 0.0
    quad_score: float = 0.0
    fidelity_score: float = 0.0
    topology_score: float = 0.0
    
    # Weights used for scoring
    weights: dict = field(default_factory=lambda: {
        "quad": 0.4,
        "fidelity": 0.4,
        "topology": 0.2
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
        
        # Topology score (0-100)
        manifold_score = 100 if self.topology.is_manifold else 0
        boundary_score = 100 if self.topology.num_boundaries <= 1 else max(0, 100 - self.topology.num_boundaries * 10)
        self.topology_score = (manifold_score + boundary_score) / 2
        
        # Overall weighted score
        self.overall_score = (
            self.weights["quad"] * self.quad_score +
            self.weights["fidelity"] * self.fidelity_score +
            self.weights["topology"] * self.topology_score
        )
    
    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "quad_score": self.quad_score,
            "fidelity_score": self.fidelity_score,
            "topology_score": self.topology_score,
            "quad_quality": self.quad_quality.to_dict(),
            "geometric_fidelity": self.geometric_fidelity.to_dict(),
            "topology": self.topology.to_dict(),
        }
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"=== Retopology Quality Score: {self.overall_score:.1f}/100 ===",
            f"  Quad Quality:      {self.quad_score:.1f}/100",
            f"    - Aspect ratio:  {self.quad_quality.aspect_ratio_mean:.3f} (ideal: 1.0)",
            f"    - Angle dev:     {np.degrees(self.quad_quality.angle_deviation_mean):.1f}° (ideal: 0°)",
            f"    - Irregular:     {self.quad_quality.irregular_vertex_ratio*100:.1f}%",
            f"  Geometric Fidelity: {self.fidelity_score:.1f}/100",
            f"    - Hausdorff:     {self.geometric_fidelity.hausdorff_distance:.6f}",
            f"    - Mean dist:     {self.geometric_fidelity.mean_distance:.6f}",
            f"    - Coverage:      {self.geometric_fidelity.coverage*100:.1f}%",
            f"  Topology:          {self.topology_score:.1f}/100",
            f"    - Vertices:      {self.topology.num_vertices}",
            f"    - Faces:         {self.topology.num_faces}",
            f"    - Manifold:      {self.topology.is_manifold}",
        ]
        return "\n".join(lines)


class MeshEvaluator:
    """
    Comprehensive mesh quality evaluator.
    
    Computes all metrics needed to assess retopology quality.
    """
    
    def __init__(self, sample_count: int = 10000):
        self.sample_count = sample_count
    
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
        
        # Create score and compute composites
        score = RetopologyScore(
            quad_quality=quad_quality,
            geometric_fidelity=geometric_fidelity,
            topology=topology
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
            vertices = mesh.vertices[face]
            
            if len(face) == 4:
                quad_count += 1
                ar = self._quad_aspect_ratio(vertices)
                angles = self._quad_angles(vertices)
                # For quads, ideal angle is 90 degrees
                angle_deviations.extend([abs(a - np.pi/2) for a in angles])
            else:
                tri_count += 1
                ar = self._triangle_aspect_ratio(vertices)
                angles = self._triangle_angles(vertices)
                # For triangles, ideal angle is 60 degrees
                angle_deviations.extend([abs(a - np.pi/3) for a in angles])
            
            aspect_ratios.append(ar)
        
        # Compute valence histogram
        valence = np.zeros(mesh.num_vertices, dtype=int)
        for face in mesh.faces:
            for vi in face:
                valence[vi] += 1
        
        valence_hist = {}
        for v in valence:
            valence_hist[int(v)] = valence_hist.get(int(v), 0) + 1
        
        # Irregular vertex ratio (not valence 4 for interior, not 2-4 for boundary)
        target_valence = 4 if mesh.is_quad else 6
        irregular = np.sum(valence != target_valence) / len(valence)
        
        return QuadQualityMetrics(
            aspect_ratio_mean=float(np.mean(aspect_ratios)),
            aspect_ratio_std=float(np.std(aspect_ratios)),
            angle_deviation_mean=float(np.mean(angle_deviations)),
            angle_deviation_std=float(np.std(angle_deviations)),
            valence_histogram=valence_hist,
            irregular_vertex_ratio=float(irregular)
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
        
        # Count edges
        edges = set()
        for face in mesh.faces:
            n = len(face)
            for i in range(n):
                v0, v1 = face[i], face[(i + 1) % n]
                edges.add((min(v0, v1), max(v0, v1)))
        n_edges = len(edges)
        
        # Euler characteristic: V - E + F
        euler = n_vertices - n_edges + n_faces
        
        # Check manifold (each edge should have exactly 2 adjacent faces)
        edge_count = {}
        for face in mesh.faces:
            n = len(face)
            for i in range(n):
                v0, v1 = face[i], face[(i + 1) % n]
                edge = (min(v0, v1), max(v0, v1))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary_edges = sum(1 for c in edge_count.values() if c == 1)
        non_manifold_edges = sum(1 for c in edge_count.values() if c > 2)
        is_manifold = non_manifold_edges == 0
        
        # Count boundary loops (simplified)
        n_boundaries = 1 if boundary_edges > 0 else 0
        
        # Genus: g = (2 - euler - boundaries) / 2
        genus = max(0, (2 - euler - n_boundaries) // 2)
        
        return TopologyMetrics(
            num_vertices=n_vertices,
            num_faces=n_faces,
            num_edges=n_edges,
            euler_characteristic=euler,
            num_boundaries=n_boundaries,
            is_manifold=is_manifold,
            genus=genus
        )


def evaluate_retopology(
    retopo_mesh: Mesh,
    original_mesh: Optional[Mesh] = None
) -> RetopologyScore:
    """Convenience function to evaluate retopology."""
    evaluator = MeshEvaluator()
    return evaluator.evaluate(retopo_mesh, original_mesh)
