"""
Visual quality evaluation for mesh retopology.

Renders the mesh from multiple viewpoints and analyzes the images
for visual quality indicators like shading smoothness, edge visibility,
and surface continuity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger("neurotopo.evaluation.visual")


@dataclass
class VisualQualityMetrics:
    """Metrics from visual/rendered evaluation."""
    shading_smoothness: float  # How smooth the shading appears (0-1)
    edge_visibility: float  # Visibility of mesh edges in render (0-1, lower is better for smooth surfaces)
    silhouette_quality: float  # Quality of mesh silhouette (0-1)
    render_consistency: float  # Consistency across multiple views (0-1)
    
    def to_dict(self) -> dict:
        return {
            "shading_smoothness": self.shading_smoothness,
            "edge_visibility": self.edge_visibility,
            "silhouette_quality": self.silhouette_quality,
            "render_consistency": self.render_consistency,
        }


class VisualEvaluator:
    """
    Evaluates mesh quality through rendered images.
    
    Uses multiple camera angles to assess visual quality
    including shading, silhouette, and surface continuity.
    """
    
    def __init__(
        self,
        resolution: tuple[int, int] = (512, 512),
        num_views: int = 4,
        render_wireframe: bool = True,
    ):
        self.resolution = resolution
        self.num_views = num_views
        self.render_wireframe = render_wireframe
        self._renderer = None
    
    def evaluate(
        self,
        mesh,  # Mesh object
        original_mesh=None,  # Optional original for comparison
    ) -> VisualQualityMetrics:
        """
        Evaluate visual quality of a mesh.
        
        Args:
            mesh: The retopologized mesh to evaluate
            original_mesh: Optional original mesh for comparison renders
            
        Returns:
            VisualQualityMetrics with all visual scores
        """
        try:
            # Render from multiple viewpoints
            renders = self._render_views(mesh)
            
            if len(renders) == 0:
                logger.warning("No renders generated, returning default metrics")
                return self._default_metrics()
            
            # Analyze shading smoothness
            shading_score = self._analyze_shading(renders)
            
            # Analyze edge visibility (wireframe overlay)
            edge_score = self._analyze_edges(mesh, renders)
            
            # Analyze silhouette quality
            silhouette_score = self._analyze_silhouette(renders)
            
            # Analyze consistency across views
            consistency_score = self._analyze_consistency(renders)
            
            return VisualQualityMetrics(
                shading_smoothness=shading_score,
                edge_visibility=edge_score,
                silhouette_quality=silhouette_score,
                render_consistency=consistency_score,
            )
            
        except Exception as e:
            logger.warning(f"Visual evaluation failed: {e}")
            return self._default_metrics()
    
    def _default_metrics(self) -> VisualQualityMetrics:
        """Return neutral default metrics when evaluation fails."""
        return VisualQualityMetrics(
            shading_smoothness=0.5,
            edge_visibility=0.5,
            silhouette_quality=0.5,
            render_consistency=0.5,
        )
    
    def _render_views(self, mesh) -> list[np.ndarray]:
        """Render mesh from multiple viewpoints."""
        renders = []
        
        try:
            import trimesh
            from trimesh import Scene
            
            # Convert to trimesh
            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False
            )
            
            # Compute vertex normals for smooth shading
            tm.fix_normals()
            
            # Camera angles (azimuth, elevation) in degrees
            angles = [
                (0, 0),      # Front
                (90, 0),     # Right
                (180, 0),    # Back
                (45, 30),    # Isometric
            ][:self.num_views]
            
            for azimuth, elevation in angles:
                try:
                    render = self._render_single_view(tm, azimuth, elevation)
                    if render is not None:
                        renders.append(render)
                except Exception as e:
                    logger.debug(f"View render failed: {e}")
                    
        except ImportError:
            logger.warning("trimesh not available for rendering")
        except Exception as e:
            logger.warning(f"Render setup failed: {e}")
        
        return renders
    
    def _render_single_view(
        self, 
        tm,  # trimesh.Trimesh
        azimuth: float, 
        elevation: float
    ) -> Optional[np.ndarray]:
        """Render a single view of the mesh."""
        try:
            import trimesh
            
            # Create scene with the mesh
            scene = trimesh.Scene(tm)
            
            # Set camera
            # Compute bounding sphere for camera distance
            center = tm.centroid
            radius = np.linalg.norm(tm.vertices - center, axis=1).max()
            distance = radius * 2.5
            
            # Convert angles to radians
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
            
            # Camera position
            cam_x = distance * np.cos(el_rad) * np.sin(az_rad)
            cam_y = distance * np.cos(el_rad) * np.cos(az_rad)
            cam_z = distance * np.sin(el_rad)
            
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = center + np.array([cam_x, cam_y, cam_z])
            
            # Try to render using pyrender if available
            try:
                import pyrender
                
                # Create pyrender scene
                pr_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
                
                # Add mesh
                pr_mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
                pr_scene.add(pr_mesh)
                
                # Add light
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
                pr_scene.add(light, pose=camera_pose)
                
                # Add camera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                pr_scene.add(camera, pose=camera_pose)
                
                # Render
                renderer = pyrender.OffscreenRenderer(*self.resolution)
                color, _ = renderer.render(pr_scene)
                renderer.delete()
                
                return color
                
            except ImportError:
                # Fallback: use trimesh's built-in rendering
                # This creates a simple depth/normal visualization
                return self._render_fallback(tm, azimuth, elevation)
                
        except Exception as e:
            logger.debug(f"Single view render failed: {e}")
            return None
    
    def _render_fallback(
        self, 
        tm,  # trimesh.Trimesh
        azimuth: float, 
        elevation: float
    ) -> Optional[np.ndarray]:
        """Fallback rendering using depth buffer analysis."""
        try:
            # Simple approach: project vertices and analyze
            center = tm.centroid
            radius = np.linalg.norm(tm.vertices - center, axis=1).max()
            
            # Create rotation matrix
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
            
            # Rotation matrices
            Rz = np.array([
                [np.cos(az_rad), -np.sin(az_rad), 0],
                [np.sin(az_rad), np.cos(az_rad), 0],
                [0, 0, 1]
            ])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(el_rad), -np.sin(el_rad)],
                [0, np.sin(el_rad), np.cos(el_rad)]
            ])
            R = Rx @ Rz
            
            # Transform vertices
            verts_centered = tm.vertices - center
            verts_rotated = verts_centered @ R.T
            
            # Project to 2D (orthographic)
            scale = (self.resolution[0] * 0.4) / radius
            verts_2d = verts_rotated[:, :2] * scale + np.array(self.resolution) / 2
            
            # Create simple depth image
            depth_image = np.zeros(self.resolution, dtype=np.float32)
            
            for face in tm.faces:
                pts = verts_2d[face].astype(np.int32)
                z = verts_rotated[face, 2].mean()
                
                # Simple triangle rasterization (approximate)
                min_x = max(0, pts[:, 0].min())
                max_x = min(self.resolution[0]-1, pts[:, 0].max())
                min_y = max(0, pts[:, 1].min())
                max_y = min(self.resolution[1]-1, pts[:, 1].max())
                
                for x in range(min_x, max_x+1):
                    for y in range(min_y, max_y+1):
                        if depth_image[y, x] < z:
                            depth_image[y, x] = z
            
            # Normalize to 0-255
            if depth_image.max() > depth_image.min():
                depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            
            # Convert to RGB
            rgb = np.stack([depth_image * 255] * 3, axis=-1).astype(np.uint8)
            return rgb
            
        except Exception as e:
            logger.debug(f"Fallback render failed: {e}")
            return None
    
    def _analyze_shading(self, renders: list[np.ndarray]) -> float:
        """
        Analyze shading smoothness in rendered images.
        
        Smooth shading = low gradient variation in non-edge regions.
        """
        smoothness_scores = []
        
        for render in renders:
            # Convert to grayscale
            if len(render.shape) == 3:
                gray = np.mean(render, axis=2)
            else:
                gray = render
            
            # Compute gradients
            gy, gx = np.gradient(gray)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            
            # Mask out background (very dark or uniform areas)
            foreground = gray > 10
            if foreground.sum() < 100:
                continue
            
            # Compute gradient statistics on foreground
            fg_gradients = gradient_mag[foreground]
            
            # Smoothness: inverse of gradient variance
            # Lower variance = smoother shading
            grad_std = np.std(fg_gradients)
            grad_mean = np.mean(fg_gradients)
            
            # Normalize: smooth mesh has low std relative to mean
            if grad_mean > 0:
                smoothness = 1.0 / (1.0 + grad_std / grad_mean)
            else:
                smoothness = 1.0
            
            smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.5
    
    def _analyze_edges(self, mesh, renders: list[np.ndarray]) -> float:
        """
        Analyze edge visibility.
        
        For smooth surfaces, visible edges indicate poor topology.
        Lower score = better (less visible edges).
        """
        # Compute edge sharpness from mesh geometry
        try:
            import trimesh
            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False
            )
            
            # Get face adjacency and compute dihedral angles
            if hasattr(tm, 'face_adjacency_angles'):
                angles = tm.face_adjacency_angles
                if len(angles) > 0:
                    # Sharp edges have large dihedral angles
                    sharp_threshold = np.radians(30)  # 30 degrees
                    sharp_ratio = np.sum(angles > sharp_threshold) / len(angles)
                    
                    # Invert: fewer sharp edges = better
                    return 1.0 - min(1.0, sharp_ratio)
        except Exception:
            pass
        
        return 0.5
    
    def _analyze_silhouette(self, renders: list[np.ndarray]) -> float:
        """
        Analyze silhouette quality.
        
        Good silhouette = smooth, not jagged edges.
        """
        silhouette_scores = []
        
        for render in renders:
            # Convert to grayscale
            if len(render.shape) == 3:
                gray = np.mean(render, axis=2)
            else:
                gray = render
            
            # Find silhouette (edges of non-background)
            foreground = gray > 10
            
            # Compute edge using simple difference
            edge_h = np.abs(np.diff(foreground.astype(float), axis=0))
            edge_v = np.abs(np.diff(foreground.astype(float), axis=1))
            
            # Pad to original size
            edge_h = np.vstack([edge_h, np.zeros((1, edge_h.shape[1]))])
            edge_v = np.hstack([edge_v, np.zeros((edge_v.shape[0], 1))])
            
            silhouette = np.maximum(edge_h, edge_v)
            
            # Compute silhouette smoothness
            # Smooth silhouette has consistent edge direction
            if silhouette.sum() < 10:
                continue
            
            # Use gradient of silhouette as roughness measure
            sy, sx = np.gradient(silhouette)
            roughness = np.mean(np.abs(sy) + np.abs(sx))
            
            # Invert: lower roughness = better
            smoothness = 1.0 / (1.0 + roughness * 10)
            silhouette_scores.append(smoothness)
        
        return np.mean(silhouette_scores) if silhouette_scores else 0.5
    
    def _analyze_consistency(self, renders: list[np.ndarray]) -> float:
        """
        Analyze consistency across multiple views.
        
        Good mesh looks consistent from all angles.
        """
        if len(renders) < 2:
            return 0.5
        
        # Compare brightness/contrast statistics across views
        stats = []
        for render in renders:
            if len(render.shape) == 3:
                gray = np.mean(render, axis=2)
            else:
                gray = render
            
            foreground = gray > 10
            if foreground.sum() < 100:
                continue
            
            fg_pixels = gray[foreground]
            stats.append({
                'mean': np.mean(fg_pixels),
                'std': np.std(fg_pixels),
            })
        
        if len(stats) < 2:
            return 0.5
        
        # Compute variance of statistics across views
        means = [s['mean'] for s in stats]
        stds = [s['std'] for s in stats]
        
        mean_variance = np.std(means) / (np.mean(means) + 1e-6)
        std_variance = np.std(stds) / (np.mean(stds) + 1e-6)
        
        # Low variance = consistent
        consistency = 1.0 / (1.0 + mean_variance + std_variance)
        
        return min(1.0, consistency)
    
    def compute_visual_score(self, metrics: VisualQualityMetrics) -> float:
        """
        Compute overall visual score from metrics.
        
        Returns score from 0-100.
        """
        # Weights for different aspects
        weights = {
            'shading': 0.35,
            'edges': 0.30,
            'silhouette': 0.20,
            'consistency': 0.15,
        }
        
        score = (
            weights['shading'] * metrics.shading_smoothness * 100 +
            weights['edges'] * metrics.edge_visibility * 100 +
            weights['silhouette'] * metrics.silhouette_quality * 100 +
            weights['consistency'] * metrics.render_consistency * 100
        )
        
        return min(100, max(0, score))
