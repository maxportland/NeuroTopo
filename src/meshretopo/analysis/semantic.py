"""
Semantic mesh segmentation using AI vision analysis.

Renders the mesh from multiple viewpoints, uses AI vision models to identify
semantic regions (eyes, mouth, nose, hands, etc.), and projects the
2D analysis back to 3D mesh faces.

This enables topology-aware remeshing that can apply rules like:
- Concentric loops around eyes
- Concentric loops around mouth
- Joint topology for elbows/knees
- Appropriate pole placement away from deformation zones
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

from meshretopo.core.mesh import Mesh
from meshretopo.core.fields import ScalarField, FieldLocation

logger = logging.getLogger("meshretopo.analysis.semantic")

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "meshretopo" / "semantic"


class SemanticRegion(Enum):
    """
    Semantic region types for topology-aware remeshing.
    
    Not limited to humans - these are general categories that apply
    to various organic and mechanical meshes.
    """
    # Facial features (organic)
    EYE = "eye"
    MOUTH = "mouth"
    NOSE = "nose"
    EAR = "ear"
    
    # Body parts (organic)
    HEAD = "head"
    NECK = "neck"
    TORSO = "torso"
    ARM = "arm"
    HAND = "hand"
    FINGER = "finger"
    LEG = "leg"
    FOOT = "foot"
    TAIL = "tail"
    
    # Joints (high deformation areas)
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"
    HIP = "hip"
    KNEE = "knee"
    ANKLE = "ankle"
    
    # General categories
    JOINT = "joint"  # Any joint/articulation point
    FEATURE = "feature"  # Important detail area
    FLAT = "flat"  # Flat/featureless area
    UNKNOWN = "unknown"


@dataclass
class RegionTopologyRules:
    """Topology rules for a semantic region."""
    
    # Loop requirements
    needs_concentric_loops: bool = False
    min_loop_count: int = 1
    ideal_loop_count: int = 3
    
    # Pole placement
    allow_poles: bool = True  # Can poles be placed here?
    ideal_pole_placement: bool = False  # Is this a good spot for poles?
    
    # Density
    relative_density: float = 1.0  # Multiplier for quad density
    
    # Edge flow
    edge_flow_direction: Optional[str] = None  # "radial", "parallel", "follow_contour"
    
    # Deformation priority
    deformation_priority: float = 0.5  # 0=static, 1=highly animated


# Default topology rules for each region type
REGION_TOPOLOGY_RULES: dict[SemanticRegion, RegionTopologyRules] = {
    # Facial features need concentric loops
    SemanticRegion.EYE: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=3,
        ideal_loop_count=5,
        allow_poles=False,  # No poles on eyelids!
        ideal_pole_placement=False,
        relative_density=1.5,
        edge_flow_direction="radial",
        deformation_priority=0.9,
    ),
    SemanticRegion.MOUTH: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=3,
        ideal_loop_count=5,
        allow_poles=False,  # No poles on lips!
        ideal_pole_placement=False,
        relative_density=1.5,
        edge_flow_direction="radial",
        deformation_priority=0.95,
    ),
    SemanticRegion.NOSE: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=2,
        ideal_loop_count=3,
        allow_poles=True,
        ideal_pole_placement=False,
        relative_density=1.3,
        edge_flow_direction="radial",
        deformation_priority=0.6,
    ),
    SemanticRegion.EAR: RegionTopologyRules(
        needs_concentric_loops=False,
        allow_poles=True,
        ideal_pole_placement=True,  # Good spot for poles
        relative_density=1.2,
        deformation_priority=0.2,
    ),
    
    # Body parts
    SemanticRegion.HEAD: RegionTopologyRules(
        relative_density=1.2,
        deformation_priority=0.5,
    ),
    SemanticRegion.TORSO: RegionTopologyRules(
        allow_poles=True,
        ideal_pole_placement=True,  # Flat areas good for poles
        relative_density=0.8,
        deformation_priority=0.4,
    ),
    SemanticRegion.HAND: RegionTopologyRules(
        relative_density=1.4,
        deformation_priority=0.8,
    ),
    SemanticRegion.FINGER: RegionTopologyRules(
        needs_concentric_loops=True,  # Rings around each segment
        min_loop_count=2,
        ideal_loop_count=3,
        allow_poles=False,
        relative_density=1.5,
        edge_flow_direction="parallel",
        deformation_priority=0.85,
    ),
    
    # Joints - critical deformation areas
    SemanticRegion.SHOULDER: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=3,
        ideal_loop_count=5,
        allow_poles=False,
        relative_density=1.5,
        deformation_priority=1.0,
    ),
    SemanticRegion.ELBOW: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=3,
        ideal_loop_count=4,
        allow_poles=False,
        relative_density=1.3,
        edge_flow_direction="parallel",
        deformation_priority=0.9,
    ),
    SemanticRegion.KNEE: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=3,
        ideal_loop_count=4,
        allow_poles=False,
        relative_density=1.3,
        edge_flow_direction="parallel",
        deformation_priority=0.9,
    ),
    SemanticRegion.WRIST: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=2,
        ideal_loop_count=3,
        allow_poles=False,
        relative_density=1.2,
        deformation_priority=0.85,
    ),
    SemanticRegion.ANKLE: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=2,
        ideal_loop_count=3,
        allow_poles=False,
        relative_density=1.2,
        deformation_priority=0.8,
    ),
    SemanticRegion.HIP: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=3,
        allow_poles=False,
        relative_density=1.4,
        deformation_priority=0.95,
    ),
    
    # Generic categories
    SemanticRegion.JOINT: RegionTopologyRules(
        needs_concentric_loops=True,
        min_loop_count=3,
        allow_poles=False,
        relative_density=1.3,
        deformation_priority=0.9,
    ),
    SemanticRegion.FEATURE: RegionTopologyRules(
        relative_density=1.2,
        deformation_priority=0.6,
    ),
    SemanticRegion.FLAT: RegionTopologyRules(
        allow_poles=True,
        ideal_pole_placement=True,
        relative_density=0.7,
        deformation_priority=0.2,
    ),
    SemanticRegion.UNKNOWN: RegionTopologyRules(
        relative_density=1.0,
        deformation_priority=0.5,
    ),
}


@dataclass
class SemanticSegment:
    """A detected semantic region in the mesh."""
    region_type: SemanticRegion
    face_indices: np.ndarray  # Indices of faces in this region
    vertex_indices: np.ndarray  # Indices of vertices in this region
    confidence: float  # Detection confidence (0-1)
    centroid: np.ndarray  # 3D centroid of the region
    bounds: tuple[np.ndarray, np.ndarray]  # (min, max) bounding box
    
    @property
    def rules(self) -> RegionTopologyRules:
        """Get topology rules for this region."""
        return REGION_TOPOLOGY_RULES.get(
            self.region_type, 
            REGION_TOPOLOGY_RULES[SemanticRegion.UNKNOWN]
        )


@dataclass
class SemanticSegmentation:
    """Complete semantic segmentation of a mesh."""
    mesh: Mesh
    segments: list[SemanticSegment] = field(default_factory=list)
    face_labels: np.ndarray = None  # Per-face region labels
    vertex_labels: np.ndarray = None  # Per-vertex region labels
    confidence_map: np.ndarray = None  # Per-face confidence
    
    def __post_init__(self):
        if self.face_labels is None:
            self.face_labels = np.full(self.mesh.num_faces, -1, dtype=int)
        if self.vertex_labels is None:
            self.vertex_labels = np.full(self.mesh.num_vertices, -1, dtype=int)
        if self.confidence_map is None:
            self.confidence_map = np.zeros(self.mesh.num_faces, dtype=float)
    
    def get_region_faces(self, region_type: SemanticRegion) -> np.ndarray:
        """Get face indices for a specific region type."""
        for segment in self.segments:
            if segment.region_type == region_type:
                return segment.face_indices
        return np.array([], dtype=int)
    
    def get_deformation_priority_field(self) -> ScalarField:
        """Generate a per-vertex field of deformation priority."""
        priorities = np.full(self.mesh.num_vertices, 0.5)
        
        for segment in self.segments:
            priority = segment.rules.deformation_priority
            priorities[segment.vertex_indices] = np.maximum(
                priorities[segment.vertex_indices], priority
            )
        
        return ScalarField(priorities, FieldLocation.VERTEX, "deformation_priority")
    
    def get_density_field(self) -> ScalarField:
        """Generate a per-vertex field of relative density."""
        densities = np.ones(self.mesh.num_vertices)
        
        for segment in self.segments:
            density = segment.rules.relative_density
            densities[segment.vertex_indices] = np.maximum(
                densities[segment.vertex_indices], density
            )
        
        return ScalarField(densities, FieldLocation.VERTEX, "semantic_density")
    
    def get_pole_penalty_field(self) -> ScalarField:
        """
        Generate a per-vertex field penalizing pole placement.
        
        Higher values = worse for poles.
        """
        penalties = np.zeros(self.mesh.num_vertices)
        
        for segment in self.segments:
            if not segment.rules.allow_poles:
                # High penalty - don't place poles here
                penalties[segment.vertex_indices] = 1.0
            elif not segment.rules.ideal_pole_placement:
                # Medium penalty
                penalties[segment.vertex_indices] = np.maximum(
                    penalties[segment.vertex_indices], 0.5
                )
            # Low/no penalty for ideal_pole_placement areas
        
        return ScalarField(penalties, FieldLocation.VERTEX, "pole_penalty")
    
    def get_loop_requirement_field(self) -> ScalarField:
        """
        Generate a per-vertex field indicating loop requirements.
        
        Higher values = needs more concentric loops.
        """
        requirements = np.zeros(self.mesh.num_vertices)
        
        for segment in self.segments:
            if segment.rules.needs_concentric_loops:
                # Normalize by ideal loop count
                value = segment.rules.ideal_loop_count / 5.0  # Max 5 loops
                requirements[segment.vertex_indices] = np.maximum(
                    requirements[segment.vertex_indices], value
                )
        
        return ScalarField(requirements, FieldLocation.VERTEX, "loop_requirement")


class SemanticAnalyzer:
    """
    Analyze mesh semantics using AI vision.
    
    Renders the mesh from multiple viewpoints, sends to an AI vision model,
    and maps the detected regions back to mesh faces.
    
    Rendering priority:
    1. Blender (best quality, runs as subprocess - no Python conflicts)
    2. pyrender (if available in current Python)
    3. PIL-based fallback (basic but works everywhere)
    
    Caching:
    - Renders and API responses are cached based on source file modification time
    - Set use_cache=False to bypass caching
    - Cache is stored in ~/.cache/meshretopo/semantic/ by default
    """
    
    def __init__(
        self,
        api_provider: str = "openai",  # or "anthropic"
        model: str = None,  # Auto-select based on provider
        resolution: tuple[int, int] = (1536, 1536),  # Higher res for AI vision
        num_views: int = 6,
        use_blender: bool = True,  # Prefer Blender when available
        blender_path: str = None,  # Custom Blender path
        use_cache: bool = True,  # Enable caching of renders and API responses
        cache_dir: Path = None,  # Custom cache directory
    ):
        self.api_provider = api_provider
        self.model = model or self._default_model(api_provider)
        self.resolution = resolution
        self.num_views = num_views
        self.use_blender = use_blender
        self.blender_path = blender_path
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._blender_renderer = None
        
        # Ensure cache directory exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
        }
        return defaults.get(provider, "gpt-4o")
    
    def _get_blender_renderer(self):
        """Get or create Blender renderer."""
        if self._blender_renderer is None:
            try:
                from meshretopo.analysis.blender_render import BlenderRenderer
                self._blender_renderer = BlenderRenderer(
                    blender_path=self.blender_path,
                    resolution=self.resolution,
                )
            except ImportError:
                self._blender_renderer = False  # Mark as unavailable
        return self._blender_renderer if self._blender_renderer else None
    
    # ========== Caching Methods ==========
    
    def _get_cache_key(self, mesh: Mesh, source_path: Optional[str] = None) -> str:
        """
        Generate a cache key based on mesh content and source file modification time.
        
        The key includes:
        - Source file path (if available)
        - Source file modification time (if available)
        - Mesh vertex/face count and bounds (as fallback hash)
        - Resolution and num_views settings
        """
        key_parts = []
        
        # Try to get source file info
        if source_path and Path(source_path).exists():
            source = Path(source_path)
            key_parts.append(f"path:{source.name}")
            mtime = source.stat().st_mtime
            key_parts.append(f"mtime:{mtime}")
        elif hasattr(mesh, '_source_path') and mesh._source_path:
            source = Path(mesh._source_path)
            if source.exists():
                key_parts.append(f"path:{source.name}")
                mtime = source.stat().st_mtime
                key_parts.append(f"mtime:{mtime}")
        
        # Fallback: use mesh content hash
        if not key_parts:
            # Hash based on mesh geometry
            mesh_data = f"{mesh.num_vertices}:{mesh.num_faces}"
            if mesh.vertices is not None and len(mesh.vertices) > 0:
                # Include bounds for uniqueness
                bounds_min = mesh.vertices.min(axis=0)
                bounds_max = mesh.vertices.max(axis=0)
                mesh_data += f":{bounds_min.tobytes().hex()[:16]}:{bounds_max.tobytes().hex()[:16]}"
            key_parts.append(f"mesh:{hashlib.md5(mesh_data.encode()).hexdigest()[:16]}")
        
        # Add settings that affect the output
        key_parts.append(f"res:{self.resolution[0]}x{self.resolution[1]}")
        key_parts.append(f"views:{self.num_views}")
        key_parts.append(f"blender:{self.use_blender}")
        
        # Create final hash
        full_key = "|".join(key_parts)
        return hashlib.sha256(full_key.encode()).hexdigest()[:32]
    
    def _get_render_cache_path(self, cache_key: str) -> Path:
        """Get path to cached renders."""
        return self.cache_dir / f"renders_{cache_key}.pkl"
    
    def _get_vision_cache_path(self, cache_key: str) -> Path:
        """Get path to cached vision API responses."""
        return self.cache_dir / f"vision_{cache_key}_{self.api_provider}_{self.model}.json"
    
    def _load_cached_renders(self, cache_key: str) -> Optional[tuple[list[np.ndarray], list[dict]]]:
        """Load cached renders if available."""
        cache_path = self._get_render_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded cached renders from {cache_path.name}")
                return data['renders'], data['camera_params']
            except Exception as e:
                logger.warning(f"Failed to load cached renders: {e}")
        return None
    
    def _save_cached_renders(
        self, 
        cache_key: str, 
        renders: list[np.ndarray], 
        camera_params: list[dict]
    ) -> None:
        """Save renders to cache."""
        cache_path = self._get_render_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'renders': renders,
                    'camera_params': camera_params,
                }, f)
            logger.info(f"Saved renders to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save renders to cache: {e}")
    
    def _load_cached_vision(self, cache_key: str) -> Optional[list[dict]]:
        """Load cached vision API response if available."""
        cache_path = self._get_vision_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                # Convert region_type strings back to enums
                for det in data:
                    if isinstance(det.get('region_type'), str):
                        det['region_type'] = self._parse_region_type(det['region_type'])
                logger.info(f"Loaded cached vision response from {cache_path.name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached vision response: {e}")
        return None
    
    def _save_cached_vision(self, cache_key: str, detections: list[dict]) -> None:
        """Save vision API response to cache."""
        cache_path = self._get_vision_cache_path(cache_key)
        try:
            # Convert enums to strings for JSON serialization
            serializable = []
            for det in detections:
                det_copy = det.copy()
                if hasattr(det_copy.get('region_type'), 'value'):
                    det_copy['region_type'] = det_copy['region_type'].value
                serializable.append(det_copy)
            
            with open(cache_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"Saved vision response to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save vision response to cache: {e}")
    
    def clear_cache(self, cache_key: str = None) -> int:
        """
        Clear cached data.
        
        Args:
            cache_key: If provided, only clear cache for this key.
                      If None, clear all cached data.
        
        Returns:
            Number of files deleted.
        """
        deleted = 0
        if cache_key:
            # Clear specific cache
            for path in [
                self._get_render_cache_path(cache_key),
                self._get_vision_cache_path(cache_key),
            ]:
                if path.exists():
                    path.unlink()
                    deleted += 1
        else:
            # Clear all cache
            if self.cache_dir.exists():
                for path in self.cache_dir.glob("*"):
                    if path.is_file():
                        path.unlink()
                        deleted += 1
        
        logger.info(f"Cleared {deleted} cached files")
        return deleted
    
    # ========== Analysis Methods ==========
    
    def analyze(
        self, 
        mesh: Mesh, 
        source_path: Optional[str] = None
    ) -> SemanticSegmentation:
        """
        Perform semantic segmentation of the mesh.
        
        Args:
            mesh: Input mesh to analyze
            source_path: Optional path to source file (for cache validation)
            
        Returns:
            SemanticSegmentation with detected regions
        """
        logger.info(f"Starting semantic analysis with {self.api_provider}/{self.model}")
        
        # Generate cache key
        cache_key = self._get_cache_key(mesh, source_path) if self.use_cache else None
        
        # Step 1: Try to load cached renders, or render from scratch
        renders = None
        camera_params = None
        
        if self.use_cache and cache_key:
            cached = self._load_cached_renders(cache_key)
            if cached:
                renders, camera_params = cached
        
        if renders is None:
            renders, camera_params = self._render_views(mesh)
            
            if self.use_cache and cache_key and renders:
                self._save_cached_renders(cache_key, renders, camera_params)
        
        if not renders:
            logger.warning("No renders generated, returning empty segmentation")
            return SemanticSegmentation(mesh)
        
        # Step 2: Try to load cached vision response, or call API
        region_detections = None
        
        if self.use_cache and cache_key:
            region_detections = self._load_cached_vision(cache_key)
        
        if region_detections is None:
            region_detections = self._analyze_with_vision(renders)
            
            if self.use_cache and cache_key and region_detections:
                self._save_cached_vision(cache_key, region_detections)
        
        if not region_detections:
            logger.warning("No regions detected by AI, returning empty segmentation")
            return SemanticSegmentation(mesh)
        
        # Step 3: Project 2D detections back to 3D mesh
        segmentation = self._project_to_mesh(mesh, region_detections, camera_params)
        
        logger.info(f"Detected {len(segmentation.segments)} semantic regions")
        return segmentation
    
    def _render_views(self, mesh: Mesh) -> tuple[list[np.ndarray], list[dict]]:
        """
        Render mesh from multiple viewpoints.
        
        Rendering priority:
        1. Blender (best quality, no Python conflicts)
        2. pyrender (if available)
        3. PIL-based fallback (basic but works)
        """
        # Try Blender first (best quality)
        if self.use_blender:
            blender_result = self._render_with_blender(mesh)
            if blender_result is not None:
                return blender_result
        
        # Fallback to internal rendering
        return self._render_internal(mesh)
    
    def _render_with_blender(self, mesh: Mesh) -> Optional[tuple[list[np.ndarray], list[dict]]]:
        """Try to render with Blender."""
        renderer = self._get_blender_renderer()
        if renderer is None or not renderer.available:
            logger.debug("Blender not available, using fallback renderer")
            return None
        
        try:
            import tempfile
            import trimesh
            
            # Save mesh to temp file
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
                temp_path = f.name
            
            mesh.to_file(temp_path)
            
            # Load mesh to get bounding info
            tm = trimesh.load(temp_path, force='mesh')
            center = tm.bounding_box.centroid
            bounds = tm.bounding_box.extents
            distance = max(bounds) * 2.5  # Camera distance
            
            # Render with Blender
            logger.info(f"Rendering with Blender at {self.resolution[0]}x{self.resolution[1]}")
            renders, blender_params = renderer.render_mesh(temp_path)
            
            # Convert Blender params to our format with full camera pose
            camera_params = []
            yfov = np.pi / 4  # 45 degree FOV (Blender default)
            
            for bp in blender_params:
                azimuth = np.radians(bp["azimuth"])
                elevation = np.radians(bp["elevation"])
                
                # Compute camera position
                x = distance * np.cos(elevation) * np.sin(azimuth)
                y = distance * np.sin(elevation)
                z = distance * np.cos(elevation) * np.cos(azimuth)
                camera_pos = center + np.array([x, y, z])
                
                # Build camera pose matrix (look at center)
                forward = center - camera_pos
                forward = forward / np.linalg.norm(forward)
                
                world_up = np.array([0, 1, 0])
                right = np.cross(world_up, forward)
                if np.linalg.norm(right) < 0.001:
                    right = np.array([1, 0, 0])
                right = right / np.linalg.norm(right)
                up = np.cross(forward, right)
                
                camera_pose = np.eye(4)
                camera_pose[:3, 0] = right
                camera_pose[:3, 1] = up
                camera_pose[:3, 2] = -forward
                camera_pose[:3, 3] = camera_pos
                
                camera_params.append({
                    "azimuth": bp["azimuth"],
                    "elevation": bp["elevation"],
                    "name": bp.get("name", ""),
                    "pose": camera_pose,
                    "yfov": yfov,
                })
            
            # Cleanup
            import os
            os.unlink(temp_path)
            
            logger.info(f"Blender rendered {len(renders)} views successfully")
            return renders[:self.num_views], camera_params[:self.num_views]
            
        except Exception as e:
            logger.warning(f"Blender rendering failed: {e}, using fallback")
            return None
    
    def _render_internal(self, mesh: Mesh) -> tuple[list[np.ndarray], list[dict]]:
        """Internal rendering using pyrender or PIL fallback."""
        renders = []
        camera_params = []
        
        try:
            import trimesh
            
            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False
            )
            tm.fix_normals()
            
            # Camera angles for comprehensive coverage
            # (azimuth, elevation) in degrees
            angles = [
                (0, 0),      # Front
                (180, 0),    # Back
                (90, 0),     # Right
                (-90, 0),    # Left
                (0, 60),     # Top-front
                (0, -30),    # Bottom-front
            ][:self.num_views]
            
            for i, (azimuth, elevation) in enumerate(angles):
                render, params = self._render_single_view(tm, azimuth, elevation)
                if render is not None:
                    renders.append(render)
                    camera_params.append(params)
                    
        except Exception as e:
            logger.warning(f"Internal render failed: {e}")
        
        return renders, camera_params
    
    def _render_single_view(
        self,
        tm,  # trimesh.Trimesh
        azimuth: float,
        elevation: float
    ) -> tuple[Optional[np.ndarray], Optional[dict]]:
        """Render a single view with camera parameters."""
        center = tm.centroid
        radius = np.linalg.norm(tm.vertices - center, axis=1).max()
        distance = radius * 2.5
        
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Camera position
        cam_x = distance * np.cos(el_rad) * np.sin(az_rad)
        cam_y = distance * np.cos(el_rad) * np.cos(az_rad)
        cam_z = distance * np.sin(el_rad)
        
        camera_pos = center + np.array([cam_x, cam_y, cam_z])
        
        # Build camera pose matrix (look at center)
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = camera_pos
        
        yfov = np.pi / 4.0
        
        # Store camera parameters for back-projection
        params = {
            "azimuth": azimuth,
            "elevation": elevation,
            "position": camera_pos,
            "pose": camera_pose,
            "yfov": yfov,
            "center": center,
            "distance": distance,
        }
        
        # Try pyrender first (best quality)
        try:
            import pyrender
            
            # Create pyrender scene
            scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
            
            # Add mesh with smooth shading
            pr_mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
            scene.add(pr_mesh)
            
            # Add lights
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=camera_pose)
            
            # Add camera
            camera = pyrender.PerspectiveCamera(yfov=yfov)
            scene.add(camera, pose=camera_pose)
            
            # Render
            renderer = pyrender.OffscreenRenderer(*self.resolution)
            color, depth = renderer.render(scene)
            renderer.delete()
            
            return color, params
            
        except ImportError:
            pass  # Fall through to fallback
        except Exception as e:
            logger.debug(f"pyrender render failed: {e}")
        
        # Fallback: use trimesh's scene rendering
        try:
            return self._render_fallback_trimesh(tm, camera_pos, center, params)
        except Exception as e:
            logger.debug(f"Trimesh fallback render failed: {e}")
        
        # Final fallback: generate a normal-shaded image manually
        try:
            return self._render_fallback_normals(tm, camera_pose, params)
        except Exception as e:
            logger.debug(f"Normal fallback render failed: {e}")
            return None, None
    
    def _render_fallback_trimesh(
        self,
        tm,  # trimesh.Trimesh
        camera_pos: np.ndarray,
        center: np.ndarray,
        params: dict
    ) -> tuple[Optional[np.ndarray], Optional[dict]]:
        """Fallback rendering using trimesh's scene."""
        import trimesh
        
        # Create scene
        scene = trimesh.Scene(tm)
        
        # Try to use scene.save_image if available (requires pyglet)
        try:
            # Set camera transform
            camera_transform = np.eye(4)
            camera_transform[:3, 3] = camera_pos
            scene.camera_transform = camera_transform
            
            # Get image data
            png_data = scene.save_image(resolution=self.resolution)
            
            # Convert to numpy array
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(png_data))
            color = np.array(img)
            
            return color, params
        except Exception as e:
            logger.debug(f"trimesh scene.save_image failed: {e}")
            raise
    
    def _render_fallback_normals(
        self,
        tm,  # trimesh.Trimesh
        camera_pose: np.ndarray,
        params: dict
    ) -> tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Ultimate fallback: render a normal-shaded image manually.
        
        Projects vertices, rasterizes triangles, and shades by normal direction.
        Not pretty, but gives AI something to work with.
        """
        from PIL import Image, ImageDraw
        
        width, height = self.resolution
        
        # Camera extrinsics (world to camera)
        R = camera_pose[:3, :3].T
        t = -R @ camera_pose[:3, 3]
        
        # Camera intrinsics (approximate)
        yfov = params["yfov"]
        focal = height / (2 * np.tan(yfov / 2))
        cx, cy = width / 2, height / 2
        
        # Project vertices to image
        verts_cam = (R @ tm.vertices.T).T + t
        
        # Perspective projection
        z = verts_cam[:, 2]
        valid = z > 0.01
        
        x_img = np.zeros(len(tm.vertices))
        y_img = np.zeros(len(tm.vertices))
        
        x_img[valid] = (verts_cam[valid, 0] / z[valid]) * focal + cx
        y_img[valid] = (verts_cam[valid, 1] / z[valid]) * focal + cy
        
        # Create image
        img = Image.new('RGB', (width, height), (50, 50, 50))  # Dark gray background
        draw = ImageDraw.Draw(img)
        
        # Compute face normals for shading
        face_normals = tm.face_normals
        
        # Light direction (from camera)
        light_dir = -camera_pose[:3, 2]
        
        # Sort faces by depth (painter's algorithm)
        face_centers = tm.vertices[tm.faces].mean(axis=1)
        face_z = (R @ face_centers.T).T[:, 2] + t[2]
        face_order = np.argsort(-face_z)  # Back to front
        
        for fi in face_order:
            face = tm.faces[fi]
            
            # Skip if any vertex is behind camera
            if not all(valid[face]):
                continue
            
            # Get projected vertices
            pts = [(int(x_img[vi]), int(y_img[vi])) for vi in face]
            
            # Compute shading (dot product with light)
            normal = face_normals[fi]
            shade = max(0.2, np.dot(normal, light_dir))
            
            # Color based on normal direction (for visibility)
            r = int(128 + 127 * normal[0] * shade)
            g = int(128 + 127 * normal[1] * shade)
            b = int(128 + 127 * normal[2] * shade)
            
            color = (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b))
            )
            
            # Draw filled polygon
            if len(pts) >= 3:
                draw.polygon(pts, fill=color, outline=color)
        
        return np.array(img), params
    
    def _analyze_with_vision(
        self,
        renders: list[np.ndarray]
    ) -> list[dict]:
        """
        Send renders to AI vision model for semantic analysis.
        
        Returns list of detections with:
        - region_type: SemanticRegion
        - view_index: which render view
        - bbox: [x1, y1, x2, y2] normalized coordinates
        - confidence: float
        """
        detections = []
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt()
        
        for view_idx, render in enumerate(renders):
            try:
                # Convert render to base64
                image_b64 = self._image_to_base64(render)
                
                # Call AI API
                response = self._call_vision_api(image_b64, prompt)
                
                # Parse response into detections
                view_detections = self._parse_vision_response(response, view_idx)
                detections.extend(view_detections)
                
            except Exception as e:
                logger.warning(f"Vision analysis failed for view {view_idx}: {e}")
        
        return detections
    
    def _build_analysis_prompt(self) -> str:
        """Build the prompt for vision analysis."""
        return """This is a 3D mesh rendering. Identify the visible anatomical/semantic regions and their approximate locations.

For each region, estimate the center position as normalized x,y coordinates (0,0 is top-left, 1,1 is bottom-right).

Return a JSON array with this exact format:
[{"region_type": "FACE", "center_2d": [0.5, 0.5], "approximate_radius": 0.3, "confidence": 0.9}]

Valid region types for organic/character meshes:
- Face/Head: FACE, EYE_SOCKET, NOSE, MOUTH, FOREHEAD, CHIN, CHEEK, NECK, EAR
- Body: TORSO, ARM, HAND, LEG, FOOT, TAIL
- Joints: SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE

For mechanical/geometric meshes, use: JOINT, FEATURE, FLAT

Focus on regions relevant for 3D modeling topology:
- Facial features that need concentric edge loops (eyes, mouth, nose)
- Joints that need special deformation topology (elbows, knees, shoulders)
- Body parts and their boundaries

If you cannot identify any meaningful regions, return an empty array: []

Respond with ONLY the JSON array, no other text."""
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image array to base64 string."""
        from PIL import Image
        
        img = Image.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _call_vision_api(self, image_b64: str, prompt: str) -> str:
        """Call the vision API with the image and prompt."""
        if self.api_provider == "openai":
            return self._call_openai_vision(image_b64, prompt)
        elif self.api_provider == "anthropic":
            return self._call_anthropic_vision(image_b64, prompt)
        else:
            raise ValueError(f"Unknown API provider: {self.api_provider}")
    
    def _call_openai_vision(self, image_b64: str, prompt: str) -> str:
        """Call OpenAI's vision API."""
        import openai
        
        client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic_vision(self, image_b64: str, prompt: str) -> str:
        """Call Anthropic's vision API."""
        import anthropic
        
        client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        return response.content[0].text
    
    def _parse_vision_response(
        self,
        response: str,
        view_index: int
    ) -> list[dict]:
        """Parse AI response into structured detections."""
        import json
        
        detections = []
        
        try:
            # Clean up response - extract JSON if wrapped in markdown
            text = response.strip()
            if text.startswith("```"):
                # Remove markdown code blocks
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines 
                    if not line.startswith("```")
                )
            
            data = json.loads(text)
            
            if not isinstance(data, list):
                logger.warning(f"Expected list, got {type(data)}")
                return []
            
            for item in data:
                try:
                    # Handle both "type" and "region_type" formats
                    type_str = item.get("region_type") or item.get("type", "unknown")
                    region_type = self._parse_region_type(type_str)
                    
                    # Handle both "bbox" and "center_2d" formats
                    if "bbox" in item:
                        bbox = item.get("bbox", [0, 0, 1, 1])
                    elif "center_2d" in item:
                        # Convert center + radius to bbox
                        center = item.get("center_2d", [0.5, 0.5])
                        radius = item.get("approximate_radius", 0.1)
                        bbox = [
                            max(0, center[0] - radius),
                            max(0, center[1] - radius),
                            min(1, center[0] + radius),
                            min(1, center[1] + radius),
                        ]
                    else:
                        bbox = [0, 0, 1, 1]
                    
                    confidence = float(item.get("confidence", 0.5))
                    
                    detections.append({
                        "region_type": region_type,
                        "view_index": view_index,
                        "bbox": bbox,
                        "confidence": confidence,
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse detection item: {e}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse vision response as JSON: {e}")
        except Exception as e:
            logger.warning(f"Error parsing vision response: {e}")
        
        return detections
    
    def _parse_region_type(self, type_str: str) -> SemanticRegion:
        """Parse string to SemanticRegion enum."""
        type_str = type_str.lower().strip().replace("_", "")
        
        # Direct mapping
        for region in SemanticRegion:
            if region.value == type_str:
                return region
            # Also check enum name (for uppercase inputs like "EYE_SOCKET")
            if region.name.lower().replace("_", "") == type_str:
                return region
        
        # Fuzzy matching for common variations
        mappings = {
            "eyes": SemanticRegion.EYE,
            "eyesocket": SemanticRegion.EYE,
            "eye_socket": SemanticRegion.EYE,
            "lips": SemanticRegion.MOUTH,
            "face": SemanticRegion.HEAD,
            "body": SemanticRegion.TORSO,
            "chest": SemanticRegion.TORSO,
            "back": SemanticRegion.TORSO,
            "belly": SemanticRegion.TORSO,
            "stomach": SemanticRegion.TORSO,
            "arms": SemanticRegion.ARM,
            "hands": SemanticRegion.HAND,
            "fingers": SemanticRegion.FINGER,
            "legs": SemanticRegion.LEG,
            "feet": SemanticRegion.FOOT,
            "toes": SemanticRegion.FINGER,  # Similar topology to fingers
            "ears": SemanticRegion.EAR,
            "shoulders": SemanticRegion.SHOULDER,
            "elbows": SemanticRegion.ELBOW,
            "wrists": SemanticRegion.WRIST,
            "hips": SemanticRegion.HIP,
            "knees": SemanticRegion.KNEE,
            "ankles": SemanticRegion.ANKLE,
            "articulation": SemanticRegion.JOINT,
            "bend": SemanticRegion.JOINT,
            "detail": SemanticRegion.FEATURE,
            "smooth": SemanticRegion.FLAT,
            "plain": SemanticRegion.FLAT,
            # Additional uppercase mappings
            "forehead": SemanticRegion.HEAD,  # Part of head
            "chin": SemanticRegion.HEAD,  # Part of head
            "cheek": SemanticRegion.HEAD,  # Part of head
        }
        
        if type_str in mappings:
            return mappings[type_str]
        
        return SemanticRegion.UNKNOWN
    
    def _project_to_mesh(
        self,
        mesh: Mesh,
        detections: list[dict],
        camera_params: list[dict]
    ) -> SemanticSegmentation:
        """
        Project 2D detections back to 3D mesh faces.
        
        Uses ray casting from camera through detection bbox to find
        which faces are in each region.
        """
        segmentation = SemanticSegmentation(mesh)
        
        # Group detections by region type
        region_detections: dict[SemanticRegion, list[dict]] = {}
        for det in detections:
            region_type = det["region_type"]
            if region_type not in region_detections:
                region_detections[region_type] = []
            region_detections[region_type].append(det)
        
        # Process each region type
        for region_type, dets in region_detections.items():
            try:
                segment = self._create_segment(mesh, region_type, dets, camera_params)
                if segment is not None and len(segment.face_indices) > 0:
                    segmentation.segments.append(segment)
                    
                    # Update labels
                    seg_idx = len(segmentation.segments) - 1
                    segmentation.face_labels[segment.face_indices] = seg_idx
                    segmentation.vertex_labels[segment.vertex_indices] = seg_idx
                    segmentation.confidence_map[segment.face_indices] = segment.confidence
                    
            except Exception as e:
                logger.warning(f"Failed to create segment for {region_type}: {e}")
        
        return segmentation
    
    def _create_segment(
        self,
        mesh: Mesh,
        region_type: SemanticRegion,
        detections: list[dict],
        camera_params: list[dict]
    ) -> Optional[SemanticSegment]:
        """Create a semantic segment from multiple view detections."""
        try:
            import trimesh
            
            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False
            )
            
            # Collect face votes from all detections
            face_votes = np.zeros(mesh.num_faces)
            confidence_sum = 0.0
            
            for det in detections:
                view_idx = det["view_index"]
                if view_idx >= len(camera_params):
                    continue
                    
                bbox = det["bbox"]
                confidence = det["confidence"]
                params = camera_params[view_idx]
                
                # Find faces visible in this bbox
                visible_faces = self._find_faces_in_bbox(
                    tm, bbox, params, self.resolution
                )
                
                face_votes[visible_faces] += confidence
                confidence_sum += confidence
            
            # Threshold to select faces
            if confidence_sum > 0:
                face_votes /= confidence_sum
            
            threshold = 0.3  # Face must have 30% of max votes
            selected_faces = np.where(face_votes > threshold * face_votes.max())[0]
            
            if len(selected_faces) == 0:
                return None
            
            # Get vertices in these faces
            selected_vertices = np.unique(mesh.faces[selected_faces].flatten())
            
            # Compute centroid and bounds
            face_centers = mesh.vertices[mesh.faces[selected_faces]].mean(axis=1)
            centroid = face_centers.mean(axis=0)
            bounds = (
                mesh.vertices[selected_vertices].min(axis=0),
                mesh.vertices[selected_vertices].max(axis=0)
            )
            
            # Average confidence
            avg_confidence = np.mean([d["confidence"] for d in detections])
            
            return SemanticSegment(
                region_type=region_type,
                face_indices=selected_faces,
                vertex_indices=selected_vertices,
                confidence=avg_confidence,
                centroid=centroid,
                bounds=bounds,
            )
            
        except Exception as e:
            logger.warning(f"Segment creation failed: {e}")
            return None
    
    def _find_faces_in_bbox(
        self,
        tm,  # trimesh.Trimesh
        bbox: list[float],  # [left, top, right, bottom] normalized
        camera_params: dict,
        resolution: tuple[int, int]
    ) -> np.ndarray:
        """
        Find mesh faces that project into the given 2D bounding box.
        
        Uses face centroid projection for efficiency.
        """
        try:
            # Get camera parameters
            pose = camera_params["pose"]  # camera-to-world transform
            yfov = camera_params["yfov"]
            
            # Camera intrinsics
            width, height = resolution
            focal = height / (2 * np.tan(yfov / 2))
            cx, cy = width / 2, height / 2
            
            # Camera extrinsics: world-to-camera is the inverse of pose
            # For a proper rigid transform: inv([R|t]) = [R^T | -R^T @ t]
            R_cam_to_world = pose[:3, :3]
            t_cam_to_world = pose[:3, 3]
            
            # World to camera transform
            R = R_cam_to_world.T
            t = -R @ t_cam_to_world
            
            # Project face centers
            face_centers = tm.vertices[tm.faces].mean(axis=1)  # (N, 3)
            
            # Transform to camera space
            cam_coords = (R @ face_centers.T).T + t  # (N, 3)
            
            # In OpenGL/Blender convention, camera looks down -Z
            # So objects in front of camera have NEGATIVE z in camera space
            # We need to negate z for the projection
            z = -cam_coords[:, 2]
            valid = z > 0.01  # Only faces in front of camera
            
            x = np.zeros(len(face_centers))
            y = np.zeros(len(face_centers))
            
            # Project to image plane (flip y for image coordinates)
            x[valid] = (cam_coords[valid, 0] / z[valid]) * focal + cx
            y[valid] = (-cam_coords[valid, 1] / z[valid]) * focal + cy
            
            # Normalize to 0-1
            x_norm = x / width
            y_norm = y / height
            
            # Convert bbox from [left, top, right, bottom] to check
            left, top, right, bottom = bbox
            
            # Find faces in bbox
            in_bbox = (
                valid &
                (x_norm >= left) & (x_norm <= right) &
                (y_norm >= top) & (y_norm <= bottom)
            )
            
            return np.where(in_bbox)[0]
            
        except Exception as e:
            logger.debug(f"Bbox face finding failed: {e}")
            return np.array([], dtype=int)


def analyze_mesh_semantics(
    mesh: Mesh,
    api_provider: str = "openai",
    model: str = None,
) -> SemanticSegmentation:
    """
    Convenience function to analyze mesh semantics.
    
    Args:
        mesh: Input mesh
        api_provider: "openai" or "anthropic"
        model: Specific model to use (default: auto-select)
        
    Returns:
        SemanticSegmentation with detected regions
    """
    analyzer = SemanticAnalyzer(api_provider=api_provider, model=model)
    return analyzer.analyze(mesh)
