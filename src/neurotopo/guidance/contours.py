"""
Contour-Based Seeding for Cylindrical Regions.

Implements RetopoFlow-inspired Contours tool concepts for automated retopology:
- Detect cylindrical/tubular regions (limbs, fingers, tails, horns, etc.)
- Automatically place edge loops perpendicular to tube axis
- Generate optimal quad strips along tubular forms

This is particularly useful for organic models with limb-like structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger("neurotopo.guidance.contours")


@dataclass
class TubularRegion:
    """Represents a detected cylindrical/tubular region of the mesh."""
    vertex_indices: np.ndarray  # Vertices belonging to this region
    axis_direction: np.ndarray  # 3D direction of tube axis
    axis_points: np.ndarray  # Points along the central axis
    radii: np.ndarray  # Approximate radius at each axis point
    start_center: np.ndarray  # Center of start cap
    end_center: np.ndarray  # Center of end cap
    
    @property
    def length(self) -> float:
        """Approximate length of the tubular region."""
        return np.linalg.norm(self.end_center - self.start_center)
    
    @property
    def average_radius(self) -> float:
        """Average radius of the tube."""
        return float(np.mean(self.radii))


@dataclass
class ContourLoop:
    """A single contour loop (edge ring) on a tubular surface."""
    center: np.ndarray  # Center point of the loop
    normal: np.ndarray  # Normal direction (along tube axis)
    radius: float  # Approximate radius
    vertex_count: int  # Target number of vertices in this loop
    vertices: Optional[np.ndarray] = None  # Generated vertex positions (Nx3)


@dataclass  
class ContourSeeding:
    """Result of contour-based seeding analysis."""
    tubular_regions: List[TubularRegion]
    suggested_loops: List[ContourLoop]
    loop_density_field: Optional[np.ndarray]  # Per-vertex suggested loop spacing


class TubularRegionDetector:
    """
    Detect cylindrical/tubular regions in a mesh.
    
    Uses geodesic eccentricity and shape analysis to identify
    limb-like structures that benefit from contour-based topology.
    """
    
    def __init__(
        self,
        mesh,  # neurotopo.core.mesh.Mesh
        min_region_size: int = 100,  # Min vertices to consider a region
        eccentricity_threshold: float = 0.7,  # How "tube-like" (0=sphere, 1=line)
        curvature_threshold: float = 0.3,  # Max curvature for tube detection
    ):
        self.mesh = mesh
        self.min_region_size = min_region_size
        self.eccentricity_threshold = eccentricity_threshold
        self.curvature_threshold = curvature_threshold
    
    def detect(self) -> List[TubularRegion]:
        """
        Detect all tubular regions in the mesh.
        
        Returns list of TubularRegion objects.
        """
        regions = []
        
        # Step 1: Compute local shape descriptors
        eccentricity = self._compute_local_eccentricity()
        
        # Step 2: Find connected regions with high eccentricity
        tubular_mask = eccentricity > self.eccentricity_threshold
        region_labels = self._connected_components(tubular_mask)
        
        # Step 3: Analyze each region
        unique_labels = np.unique(region_labels)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            region_verts = np.where(region_labels == label)[0]
            
            if len(region_verts) < self.min_region_size:
                continue
            
            # Analyze this region
            region = self._analyze_region(region_verts)
            if region is not None:
                regions.append(region)
        
        logger.debug(f"Detected {len(regions)} tubular regions")
        return regions
    
    def _compute_local_eccentricity(self) -> np.ndarray:
        """
        Compute local eccentricity for each vertex.
        
        Eccentricity measures how elongated the local neighborhood is.
        High eccentricity = tubular, Low eccentricity = spherical/flat.
        """
        verts = self.mesh.vertices
        n_verts = len(verts)
        eccentricity = np.zeros(n_verts)
        
        # Build adjacency for k-ring neighborhood
        adjacency = self._build_adjacency()
        
        for vi in range(n_verts):
            # Get 2-ring neighborhood
            neighbors = self._get_k_ring_neighbors(vi, adjacency, k=2)
            
            if len(neighbors) < 6:
                continue
            
            # PCA on neighborhood positions
            neighborhood = verts[list(neighbors)]
            centered = neighborhood - neighborhood.mean(axis=0)
            
            try:
                _, s, _ = np.linalg.svd(centered)
                
                # Eccentricity from singular value ratios
                # High if one direction dominates (tubular)
                if s[0] > 1e-10:
                    # Ratio of first to second singular value
                    eccentricity[vi] = 1.0 - (s[1] / s[0]) if len(s) > 1 else 1.0
            except Exception:
                continue
        
        return eccentricity
    
    def _build_adjacency(self) -> List[set]:
        """Build vertex adjacency list."""
        n_verts = self.mesh.num_vertices
        adjacency = [set() for _ in range(n_verts)]
        
        for face in self.mesh.faces:
            for i in range(len(face)):
                for j in range(len(face)):
                    if i != j:
                        adjacency[face[i]].add(face[j])
        
        return adjacency
    
    def _get_k_ring_neighbors(
        self, 
        vi: int, 
        adjacency: List[set], 
        k: int
    ) -> set:
        """Get k-ring neighborhood of a vertex."""
        current = {vi}
        all_neighbors = {vi}
        
        for _ in range(k):
            next_ring = set()
            for v in current:
                next_ring.update(adjacency[v])
            next_ring -= all_neighbors
            all_neighbors.update(next_ring)
            current = next_ring
        
        all_neighbors.remove(vi)
        return all_neighbors
    
    def _connected_components(self, mask: np.ndarray) -> np.ndarray:
        """Find connected components of masked vertices."""
        n_verts = len(mask)
        labels = np.zeros(n_verts, dtype=int)
        adjacency = self._build_adjacency()
        
        current_label = 0
        visited = set()
        
        for vi in range(n_verts):
            if vi in visited or not mask[vi]:
                continue
            
            # BFS to find connected component
            current_label += 1
            queue = [vi]
            
            while queue:
                v = queue.pop(0)
                if v in visited:
                    continue
                visited.add(v)
                
                if mask[v]:
                    labels[v] = current_label
                    for neighbor in adjacency[v]:
                        if neighbor not in visited and mask[neighbor]:
                            queue.append(neighbor)
        
        return labels
    
    def _analyze_region(self, vertex_indices: np.ndarray) -> Optional[TubularRegion]:
        """
        Analyze a region to determine its tubular properties.
        """
        verts = self.mesh.vertices[vertex_indices]
        
        # PCA to find principal axis
        centered = verts - verts.mean(axis=0)
        try:
            u, s, vh = np.linalg.svd(centered)
        except Exception:
            return None
        
        # Primary axis direction
        axis_direction = vh[0]
        
        # Project vertices onto axis to find extent
        projections = centered @ axis_direction
        min_proj = projections.min()
        max_proj = projections.max()
        
        # Create axis points
        n_axis_points = max(5, int((max_proj - min_proj) / (s[1] if s[1] > 0 else 1) * 2))
        n_axis_points = min(n_axis_points, 20)  # Cap for performance
        
        axis_params = np.linspace(min_proj, max_proj, n_axis_points)
        center = verts.mean(axis=0)
        axis_points = center + axis_params[:, np.newaxis] * axis_direction
        
        # Estimate radii at each axis point
        radii = []
        for ap in axis_points:
            # Find vertices close to this axis point (in perpendicular plane)
            perp_dists = np.abs(projections - np.dot(ap - center, axis_direction))
            close_mask = perp_dists < (max_proj - min_proj) / n_axis_points
            
            if close_mask.sum() > 0:
                close_verts = verts[close_mask]
                # Perpendicular distance from axis
                to_axis = close_verts - ap
                perp_component = to_axis - np.outer(to_axis @ axis_direction, axis_direction)
                r = np.linalg.norm(perp_component, axis=1).mean()
                radii.append(r)
            else:
                radii.append(s[1] / 2 if len(s) > 1 else 1.0)
        
        radii = np.array(radii)
        
        return TubularRegion(
            vertex_indices=vertex_indices,
            axis_direction=axis_direction,
            axis_points=axis_points,
            radii=radii,
            start_center=axis_points[0],
            end_center=axis_points[-1],
        )


class ContourLoopGenerator:
    """
    Generate contour loops for tubular regions.
    
    Places edge loops perpendicular to tube axis, similar to
    RetopoFlow's Contours tool behavior.
    """
    
    def __init__(
        self,
        target_loop_spacing: Optional[float] = None,
        min_loops_per_region: int = 3,
        max_loops_per_region: int = 50,
        vertices_per_loop: int = 8,  # Default circumference resolution
    ):
        self.target_loop_spacing = target_loop_spacing
        self.min_loops = min_loops_per_region
        self.max_loops = max_loops_per_region
        self.vertices_per_loop = vertices_per_loop
    
    def generate_loops(
        self,
        region: TubularRegion,
        target_face_count: Optional[int] = None,
    ) -> List[ContourLoop]:
        """
        Generate contour loops for a tubular region.
        
        Args:
            region: TubularRegion to generate loops for
            target_face_count: Overall target face count (for adaptive density)
            
        Returns:
            List of ContourLoop objects
        """
        # Determine number of loops based on length and target density
        if self.target_loop_spacing is not None:
            n_loops = max(self.min_loops, int(region.length / self.target_loop_spacing))
        else:
            # Adaptive based on radius ratio
            n_loops = max(self.min_loops, int(region.length / (region.average_radius * 2)))
        
        n_loops = min(n_loops, self.max_loops)
        
        # Generate loops along the axis
        loops = []
        
        # Interpolate along axis points
        for i in range(n_loops):
            t = i / max(n_loops - 1, 1)
            
            # Interpolate position and radius along axis
            axis_idx = t * (len(region.axis_points) - 1)
            idx_low = int(axis_idx)
            idx_high = min(idx_low + 1, len(region.axis_points) - 1)
            frac = axis_idx - idx_low
            
            center = (1 - frac) * region.axis_points[idx_low] + frac * region.axis_points[idx_high]
            radius = (1 - frac) * region.radii[idx_low] + frac * region.radii[idx_high]
            
            loop = ContourLoop(
                center=center,
                normal=region.axis_direction,
                radius=radius,
                vertex_count=self.vertices_per_loop,
            )
            
            # Generate actual vertex positions for this loop
            loop.vertices = self._generate_loop_vertices(loop)
            loops.append(loop)
        
        return loops
    
    def _generate_loop_vertices(self, loop: ContourLoop) -> np.ndarray:
        """Generate vertex positions around a contour loop."""
        # Build orthonormal frame
        normal = loop.normal / np.linalg.norm(loop.normal)
        
        # Find perpendicular vectors
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(normal, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])
        
        tangent1 = np.cross(normal, up)
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        
        # Generate vertices around the loop
        vertices = np.zeros((loop.vertex_count, 3))
        angles = np.linspace(0, 2 * np.pi, loop.vertex_count, endpoint=False)
        
        for i, theta in enumerate(angles):
            offset = (np.cos(theta) * tangent1 + np.sin(theta) * tangent2) * loop.radius
            vertices[i] = loop.center + offset
        
        return vertices


class ContourGuidanceGenerator:
    """
    Generate guidance fields from contour analysis.
    
    Converts detected tubular regions and contour loops into
    guidance for the remeshing pipeline.
    """
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.detector = TubularRegionDetector(mesh)
        self.loop_generator = ContourLoopGenerator()
    
    def generate(
        self,
        target_faces: Optional[int] = None,
    ) -> ContourSeeding:
        """
        Generate full contour seeding information.
        
        Args:
            target_faces: Target face count for density calculation
            
        Returns:
            ContourSeeding with detected regions and suggested loops
        """
        # Detect tubular regions
        regions = self.detector.detect()
        
        # Generate loops for each region
        all_loops = []
        for region in regions:
            loops = self.loop_generator.generate_loops(region, target_faces)
            all_loops.extend(loops)
        
        # Generate loop density field (for adaptive sizing)
        density_field = self._compute_loop_density_field(regions)
        
        return ContourSeeding(
            tubular_regions=regions,
            suggested_loops=all_loops,
            loop_density_field=density_field,
        )
    
    def _compute_loop_density_field(
        self,
        regions: List[TubularRegion],
    ) -> np.ndarray:
        """
        Compute per-vertex loop density suggestions.
        
        Vertices in tubular regions get higher density (smaller quads)
        perpendicular to the tube axis.
        """
        n_verts = self.mesh.num_vertices
        density = np.ones(n_verts)  # Default density = 1
        
        for region in regions:
            # Increase density for tubular region vertices
            # (they benefit from more circumferential loops)
            density[region.vertex_indices] *= 1.5
        
        return density


def detect_tubular_regions(mesh) -> List[TubularRegion]:
    """Convenience function to detect tubular regions."""
    detector = TubularRegionDetector(mesh)
    return detector.detect()


def generate_contour_seeding(
    mesh,
    target_faces: Optional[int] = None,
) -> ContourSeeding:
    """
    Convenience function to generate contour seeding.
    
    Args:
        mesh: Input mesh
        target_faces: Target face count
        
    Returns:
        ContourSeeding with regions and loops
    """
    generator = ContourGuidanceGenerator(mesh)
    return generator.generate(target_faces)
