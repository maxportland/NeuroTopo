"""
Visualization utilities for retopology evaluation.

Provides visual debugging and result presentation tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.core.fields import ScalarField, FieldLocation
from neurotopo.evaluation.metrics import RetopologyScore


class MeshVisualizer:
    """
    Visualization tools for meshes and fields.
    
    Supports multiple backends (matplotlib, Open3D, HTML export).
    """
    
    def __init__(self, backend: str = "matplotlib"):
        self.backend = backend
    
    def show_mesh(
        self,
        mesh: Mesh,
        color_field: Optional[ScalarField] = None,
        wireframe: bool = True,
        title: str = "",
    ) -> None:
        """Display mesh with optional scalar field coloring."""
        if self.backend == "matplotlib":
            self._show_matplotlib(mesh, color_field, wireframe, title)
        elif self.backend == "open3d":
            self._show_open3d(mesh, color_field, wireframe, title)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _show_matplotlib(
        self,
        mesh: Mesh,
        color_field: Optional[ScalarField],
        wireframe: bool,
        title: str
    ) -> None:
        """Render mesh using matplotlib."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to triangles if needed
        if mesh.is_quad:
            mesh = mesh.triangulate()
        
        # Get vertex colors from field
        if color_field is not None:
            # Map to colormap
            cmap = plt.cm.viridis
            normalized = (color_field.values - color_field.min) / (color_field.max - color_field.min + 1e-10)
            vertex_colors = cmap(normalized)
        else:
            vertex_colors = None
        
        # Create polygon collection
        verts = mesh.vertices[mesh.faces]
        
        if vertex_colors is not None:
            # Average colors per face
            face_colors = vertex_colors[mesh.faces].mean(axis=1)
            poly = Poly3DCollection(verts, facecolors=face_colors, edgecolors='k' if wireframe else 'none', linewidths=0.5)
        else:
            poly = Poly3DCollection(verts, facecolors='lightblue', edgecolors='k' if wireframe else 'none', linewidths=0.5)
        
        ax.add_collection3d(poly)
        
        # Set axis limits
        min_b, max_b = mesh.bounds
        max_range = (max_b - min_b).max() / 2
        center = mesh.center
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title or mesh.name)
        
        plt.tight_layout()
        plt.show()
    
    def _show_open3d(
        self,
        mesh: Mesh,
        color_field: Optional[ScalarField],
        wireframe: bool,
        title: str
    ) -> None:
        """Render mesh using Open3D."""
        import open3d as o3d
        
        o3d_mesh = mesh.to_open3d()
        
        if color_field is not None:
            # Apply colors from field
            cmap = plt.cm.viridis
            normalized = (color_field.values - color_field.min) / (color_field.max - color_field.min + 1e-10)
            colors = cmap(normalized)[:, :3]
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        geometries = [o3d_mesh]
        
        if wireframe:
            lineset = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
            lineset.paint_uniform_color([0, 0, 0])
            geometries.append(lineset)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=title or mesh.name
        )
    
    def save_comparison(
        self,
        original: Mesh,
        retopo: Mesh,
        score: RetopologyScore,
        output_path: Union[str, Path],
    ) -> None:
        """Save a comparison visualization to file."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure(figsize=(16, 10))
        
        # Original mesh
        ax1 = fig.add_subplot(231, projection='3d')
        self._plot_mesh_to_axis(ax1, original, "Original")
        
        # Retopo mesh
        ax2 = fig.add_subplot(232, projection='3d')
        self._plot_mesh_to_axis(ax2, retopo, "Retopologized")
        
        # Wireframe comparison
        ax3 = fig.add_subplot(233, projection='3d')
        self._plot_mesh_to_axis(ax3, retopo, "Wireframe", wireframe_only=True)
        
        # Score breakdown
        ax4 = fig.add_subplot(234)
        categories = ['Overall', 'Quad', 'Fidelity', 'Topology']
        scores = [score.overall_score, score.quad_score, score.fidelity_score, score.topology_score]
        colors = ['#2ecc71' if s >= 70 else '#f39c12' if s >= 50 else '#e74c3c' for s in scores]
        
        bars = ax4.bar(categories, scores, color=colors)
        ax4.set_ylim(0, 100)
        ax4.set_ylabel('Score')
        ax4.set_title('Quality Scores')
        ax4.axhline(y=70, color='g', linestyle='--', alpha=0.5)
        ax4.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
        
        for bar, s in zip(bars, scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{s:.1f}', ha='center', va='bottom')
        
        # Metrics table
        ax5 = fig.add_subplot(235)
        ax5.axis('off')
        
        metrics_text = [
            f"Vertices: {original.num_vertices} → {retopo.num_vertices}",
            f"Faces: {original.num_faces} → {retopo.num_faces}",
            f"Quad Aspect Ratio: {score.quad_quality.aspect_ratio_mean:.3f}",
            f"Angle Deviation: {np.degrees(score.quad_quality.angle_deviation_mean):.1f}°",
            f"Irregular Vertices: {score.quad_quality.irregular_vertex_ratio*100:.1f}%",
            f"Hausdorff Distance: {score.geometric_fidelity.hausdorff_distance:.6f}",
            f"Mean Distance: {score.geometric_fidelity.mean_distance:.6f}",
            f"Coverage: {score.geometric_fidelity.coverage*100:.1f}%",
        ]
        
        ax5.text(0.1, 0.9, '\n'.join(metrics_text), transform=ax5.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top')
        ax5.set_title('Metrics')
        
        # Valence histogram
        ax6 = fig.add_subplot(236)
        valence_data = score.quad_quality.valence_histogram
        valences = sorted(valence_data.keys())
        counts = [valence_data[v] for v in valences]
        
        colors_hist = ['#2ecc71' if v == 4 else '#f39c12' for v in valences]
        ax6.bar([str(v) for v in valences], counts, color=colors_hist)
        ax6.set_xlabel('Valence')
        ax6.set_ylabel('Count')
        ax6.set_title('Vertex Valence Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_mesh_to_axis(
        self,
        ax,
        mesh: Mesh,
        title: str,
        wireframe_only: bool = False
    ) -> None:
        """Plot mesh to a matplotlib axis."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        if mesh.is_quad:
            display_mesh = mesh.triangulate()
        else:
            display_mesh = mesh
        
        verts = display_mesh.vertices[display_mesh.faces]
        
        if wireframe_only:
            poly = Poly3DCollection(verts, facecolors='white', edgecolors='blue', 
                                   linewidths=0.3, alpha=0.3)
        else:
            poly = Poly3DCollection(verts, facecolors='lightblue', edgecolors='k', 
                                   linewidths=0.2, alpha=0.9)
        
        ax.add_collection3d(poly)
        
        min_b, max_b = mesh.bounds
        max_range = (max_b - min_b).max() / 2
        center = mesh.center
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.set_title(title)


def visualize_comparison(
    original: Mesh,
    retopo: Mesh,
    score: RetopologyScore,
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """Convenience function to visualize comparison."""
    viz = MeshVisualizer()
    
    if output_path:
        viz.save_comparison(original, retopo, score, output_path)
    else:
        # Interactive display
        print(score.summary())
