"""
Mesh visualization utilities.

Provides visualization for:
- Mesh wireframes and surfaces
- Quality heatmaps
- Edge flow directions
- Comparison views
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from meshretopo.core.mesh import Mesh


def visualize_mesh(
    mesh: Mesh,
    title: str = "Mesh",
    show_wireframe: bool = True,
    face_color: str = "lightblue",
    edge_color: str = "black",
    alpha: float = 0.7,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
    elevation: float = 30,
    azimuth: float = 45,
) -> plt.Figure:
    """
    Visualize a mesh in 3D.
    
    Args:
        mesh: Mesh to visualize
        title: Plot title
        show_wireframe: Show edge wireframe
        face_color: Face fill color
        edge_color: Edge line color
        alpha: Face transparency
        figsize: Figure size
        save_path: Optional path to save image
        elevation: Camera elevation angle
        azimuth: Camera azimuth angle
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get triangulated version for display
    tri_mesh = mesh.triangulate() if not mesh.is_triangular else mesh
    
    # Create polygon collection
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    
    # Get face vertices
    face_verts = vertices[faces]
    
    # Create collection
    collection = Poly3DCollection(
        face_verts,
        alpha=alpha,
        facecolor=face_color,
        edgecolor=edge_color if show_wireframe else 'none',
        linewidth=0.5 if show_wireframe else 0,
    )
    ax.add_collection3d(collection)
    
    # Set axis limits
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2 * 1.2
    
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\n{mesh.num_vertices} verts, {mesh.num_faces} faces")
    
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_quality_heatmap(
    mesh: Mesh,
    quality_values: np.ndarray,
    title: str = "Quality Heatmap",
    cmap: str = "RdYlGn",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize per-face quality as a heatmap.
    
    Args:
        mesh: Mesh to visualize
        quality_values: Per-face quality values (0-1)
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    tri_mesh = mesh.triangulate() if not mesh.is_triangular else mesh
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    
    # Expand quality values if needed (for triangulated quads)
    if len(quality_values) != len(faces):
        # Assume original quality is per-quad, expand to triangles
        if mesh.is_quad:
            expanded = np.repeat(quality_values, 2)
            quality_values = expanded[:len(faces)]
        else:
            quality_values = np.ones(len(faces)) * quality_values.mean()
    
    # Normalize to 0-1
    vmin, vmax = quality_values.min(), quality_values.max()
    if vmax - vmin > 1e-10:
        normalized = (quality_values - vmin) / (vmax - vmin)
    else:
        normalized = np.ones_like(quality_values) * 0.5
    
    # Get colormap
    colormap = plt.cm.get_cmap(cmap)
    face_colors = colormap(normalized)
    
    # Create polygons
    face_verts = vertices[faces]
    
    collection = Poly3DCollection(
        face_verts,
        alpha=0.9,
        facecolor=face_colors,
        edgecolor='gray',
        linewidth=0.3,
    )
    ax.add_collection3d(collection)
    
    # Set limits
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2 * 1.2
    
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\nRange: [{vmin:.3f}, {vmax:.3f}]")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label='Quality')
    
    ax.set_box_aspect([1, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_comparison(
    original: Mesh,
    retopo: Mesh,
    title: str = "Comparison",
    figsize: tuple = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of original and retopologized mesh.
    
    Args:
        original: Original high-poly mesh
        retopo: Retopologized mesh
        title: Overall title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Original mesh
    ax1 = fig.add_subplot(131, projection='3d')
    _plot_mesh_on_axis(ax1, original, "Original", "lightgray")
    
    # Retopo mesh
    ax2 = fig.add_subplot(132, projection='3d')
    _plot_mesh_on_axis(ax2, retopo, "Retopology", "lightblue")
    
    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    _plot_mesh_on_axis(ax3, original, "", "lightgray", alpha=0.3, wireframe=False)
    _plot_mesh_on_axis(ax3, retopo, "Overlay", "blue", alpha=0.7, wireframe=True)
    
    fig.suptitle(f"{title}\nOriginal: {original.num_faces} faces → Retopo: {retopo.num_faces} faces")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _plot_mesh_on_axis(
    ax,
    mesh: Mesh,
    title: str,
    color: str,
    alpha: float = 0.7,
    wireframe: bool = True,
):
    """Helper to plot mesh on an axis."""
    tri_mesh = mesh.triangulate() if not mesh.is_triangular else mesh
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    
    face_verts = vertices[faces]
    
    collection = Poly3DCollection(
        face_verts,
        alpha=alpha,
        facecolor=color,
        edgecolor='black' if wireframe else 'none',
        linewidth=0.3 if wireframe else 0,
    )
    ax.add_collection3d(collection)
    
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2 * 1.2
    
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])


def compute_face_quality(mesh: Mesh) -> np.ndarray:
    """
    Compute per-face quality for visualization.
    
    Returns values in [0, 1] where 1 = high quality.
    """
    qualities = []
    
    for face in mesh.faces:
        verts = mesh.vertices[face]
        unique = list(set(face))
        
        if len(unique) < 3:
            qualities.append(0.0)
            continue
        
        verts = mesh.vertices[unique]
        
        if len(unique) == 4:
            # Quad quality
            edges = [np.linalg.norm(verts[(i+1)%4] - verts[i]) for i in range(4)]
            if min(edges) < 1e-10:
                qualities.append(0.0)
                continue
            
            # Aspect ratio score (1.0 = perfect)
            aspect = max(edges) / min(edges)
            aspect_score = 1.0 / aspect
            
            # Angle score
            angle_devs = []
            for i in range(4):
                p0 = verts[(i-1) % 4]
                p1 = verts[i]
                p2 = verts[(i+1) % 4]
                e1, e2 = p0 - p1, p2 - p1
                n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_angle = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                    angle = np.arccos(cos_angle)
                    angle_devs.append(abs(angle - np.pi/2))
            
            angle_score = 1.0 - (np.mean(angle_devs) / (np.pi/2)) if angle_devs else 0.5
            
            quality = (aspect_score + angle_score) / 2
        else:
            # Triangle quality (using aspect ratio)
            edges = [np.linalg.norm(verts[(i+1)%3] - verts[i]) for i in range(3)]
            if min(edges) < 1e-10:
                qualities.append(0.0)
                continue
            
            aspect = max(edges) / min(edges)
            quality = 1.0 / aspect
        
        qualities.append(max(0.0, min(1.0, quality)))
    
    return np.array(qualities)


def save_visualization_report(
    original: Mesh,
    retopo: Mesh,
    output_dir: str = "visualization",
    prefix: str = "retopo",
):
    """
    Generate and save a complete visualization report.
    
    Args:
        original: Original mesh
        retopo: Retopologized mesh
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Original mesh
    fig1 = visualize_mesh(original, "Original Mesh")
    fig1.savefig(output_dir / f"{prefix}_original.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Retopo mesh
    fig2 = visualize_mesh(retopo, "Retopology Result", face_color="lightgreen")
    fig2.savefig(output_dir / f"{prefix}_retopo.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Quality heatmap
    quality = compute_face_quality(retopo)
    fig3 = visualize_quality_heatmap(retopo, quality, "Face Quality")
    fig3.savefig(output_dir / f"{prefix}_quality.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Comparison
    fig4 = visualize_comparison(original, retopo)
    fig4.savefig(output_dir / f"{prefix}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"Saved visualization report to {output_dir}/")
