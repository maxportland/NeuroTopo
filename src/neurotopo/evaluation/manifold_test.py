"""
Comprehensive manifold testing using Blender.

Provides detailed manifold analysis beyond simple edge counting,
including non-manifold vertices, boundary detection, and more.
"""

from __future__ import annotations

import logging
import tempfile
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from neurotopo.core.mesh import Mesh

logger = logging.getLogger("neurotopo.evaluation.manifold_test")


@dataclass
class ManifoldTestResult:
    """Results from comprehensive manifold testing."""
    
    # Overall status
    is_manifold: bool
    
    # Non-manifold vertices (pinch points, bowtie vertices)
    non_manifold_vertices: List[int] = field(default_factory=list)
    
    # Non-manifold edges (shared by >2 faces)
    non_manifold_edges: List[tuple] = field(default_factory=list)
    
    # Boundary edges (shared by only 1 face) - open mesh borders
    boundary_edges: List[tuple] = field(default_factory=list)
    
    # Wire edges (edges not connected to any face)
    wire_edges: List[tuple] = field(default_factory=list)
    
    # Interior faces (faces inside closed volumes)
    interior_faces: List[int] = field(default_factory=list)
    
    # Summary counts
    num_non_manifold_vertices: int = 0
    num_non_manifold_edges: int = 0
    num_boundary_edges: int = 0
    num_wire_edges: int = 0
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "✓ MANIFOLD" if self.is_manifold else "✗ NON-MANIFOLD"
        lines = [
            f"Manifold Test: {status}",
            f"  Non-manifold vertices: {self.num_non_manifold_vertices}",
            f"  Non-manifold edges: {self.num_non_manifold_edges}",
            f"  Boundary edges: {self.num_boundary_edges}",
            f"  Wire edges: {self.num_wire_edges}",
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_manifold": self.is_manifold,
            "num_non_manifold_vertices": self.num_non_manifold_vertices,
            "num_non_manifold_edges": self.num_non_manifold_edges,
            "num_boundary_edges": self.num_boundary_edges,
            "num_wire_edges": self.num_wire_edges,
            "non_manifold_vertices": self.non_manifold_vertices[:100],  # Limit for JSON
            "non_manifold_edges": [list(e) for e in self.non_manifold_edges[:100]],
            "boundary_edges": [list(e) for e in self.boundary_edges[:100]],
        }


def test_manifold_python(mesh: Mesh) -> ManifoldTestResult:
    """
    Test manifold properties using pure Python (no Blender required).
    
    This is a fast fallback when Blender is not available.
    """
    n_verts = mesh.num_vertices
    
    # Build edge -> face count mapping
    edge_faces = {}  # edge -> list of face indices
    vertex_edge_count = np.zeros(n_verts, dtype=np.int32)
    
    for fi, face in enumerate(mesh.faces):
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i + 1) % n]
            if v0 == v1:
                continue
            edge = (min(v0, v1), max(v0, v1))
            if edge not in edge_faces:
                edge_faces[edge] = []
                vertex_edge_count[v0] += 1
                vertex_edge_count[v1] += 1
            edge_faces[edge].append(fi)
    
    # Find non-manifold edges (>2 faces) and boundary edges (1 face)
    non_manifold_edges = []
    boundary_edges = []
    wire_edges = []
    
    for edge, faces in edge_faces.items():
        if len(faces) > 2:
            non_manifold_edges.append(edge)
        elif len(faces) == 1:
            boundary_edges.append(edge)
        elif len(faces) == 0:
            wire_edges.append(edge)
    
    # Find non-manifold vertices
    # A vertex is non-manifold if its surrounding faces don't form a fan/disk
    non_manifold_vertices = []
    
    # Build vertex -> faces mapping
    vertex_faces = [[] for _ in range(n_verts)]
    for fi, face in enumerate(mesh.faces):
        for vi in face:
            vertex_faces[vi].append(fi)
    
    for vi in range(n_verts):
        faces = vertex_faces[vi]
        if len(faces) < 2:
            continue
        
        # Check if faces around vertex form a connected fan
        # Build adjacency of faces at this vertex
        face_neighbors = {fi: set() for fi in faces}
        for fi in faces:
            face = mesh.faces[fi]
            idx = list(face).index(vi)
            n = len(face)
            prev_v = face[(idx - 1) % n]
            next_v = face[(idx + 1) % n]
            
            # Find faces that share prev_v or next_v edges
            for fj in faces:
                if fi == fj:
                    continue
                other_face = mesh.faces[fj]
                if prev_v in other_face or next_v in other_face:
                    face_neighbors[fi].add(fj)
        
        # BFS to check connectivity
        visited = set()
        queue = [faces[0]]
        visited.add(faces[0])
        
        while queue:
            f = queue.pop(0)
            for neighbor in face_neighbors[f]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # If not all faces are connected, vertex is non-manifold
        if len(visited) < len(faces):
            non_manifold_vertices.append(vi)
    
    # Determine overall manifold status
    # Note: boundary edges don't make a mesh non-manifold, they just mean it's open
    is_manifold = (
        len(non_manifold_vertices) == 0 and
        len(non_manifold_edges) == 0 and
        len(wire_edges) == 0
    )
    
    return ManifoldTestResult(
        is_manifold=is_manifold,
        non_manifold_vertices=non_manifold_vertices,
        non_manifold_edges=non_manifold_edges,
        boundary_edges=boundary_edges,
        wire_edges=wire_edges,
        num_non_manifold_vertices=len(non_manifold_vertices),
        num_non_manifold_edges=len(non_manifold_edges),
        num_boundary_edges=len(boundary_edges),
        num_wire_edges=len(wire_edges),
    )


def test_manifold_blender(mesh: Mesh) -> ManifoldTestResult:
    """
    Test manifold properties using Blender's comprehensive checks.
    
    Uses Blender's select_non_manifold operator and bmesh is_manifold property
    for accurate detection.
    """
    blender_path = "/Applications/Blender.app/Contents/MacOS/blender"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_obj = tmpdir / "input.obj"
        result_file = tmpdir / "manifold_result.txt"
        
        mesh.to_file(input_obj)
        
        script = f'''
import bpy
import bmesh
import json

# Clear and import
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{input_obj}")

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Create bmesh for detailed analysis
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
bm.faces.ensure_lookup_table()

# Find non-manifold vertices using bmesh
non_manifold_verts = []
for v in bm.verts:
    if not v.is_manifold:
        non_manifold_verts.append(v.index)

# Find non-manifold edges and boundary edges
non_manifold_edges = []
boundary_edges = []
wire_edges = []

for e in bm.edges:
    num_faces = len(e.link_faces)
    if num_faces > 2:
        non_manifold_edges.append([e.verts[0].index, e.verts[1].index])
    elif num_faces == 1:
        boundary_edges.append([e.verts[0].index, e.verts[1].index])
    elif num_faces == 0:
        wire_edges.append([e.verts[0].index, e.verts[1].index])

bm.free()

# Also use select_non_manifold for cross-validation
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_non_manifold(
    extend=False,
    use_wire=True,
    use_boundary=False,  # Don't count boundaries as non-manifold
    use_multi_face=True,
    use_non_contiguous=True,
    use_verts=True
)
bpy.ops.object.mode_set(mode='OBJECT')

# Get selected verts from select_non_manifold
selected_non_manifold = [v.index for v in obj.data.vertices if v.select]

# Combine results (bmesh is_manifold is more reliable)
all_non_manifold_verts = list(set(non_manifold_verts + selected_non_manifold))

# Write results
result = {{
    "non_manifold_vertices": all_non_manifold_verts,
    "non_manifold_edges": non_manifold_edges,
    "boundary_edges": boundary_edges,
    "wire_edges": wire_edges,
}}

with open(r"{result_file}", "w") as f:
    json.dump(result, f)

print(f"Manifold test complete: {{len(all_non_manifold_verts)}} non-manifold verts, "
      f"{{len(non_manifold_edges)}} non-manifold edges, {{len(boundary_edges)}} boundary edges")
'''
        
        script_path = tmpdir / "manifold_test.py"
        script_path.write_text(script)
        
        result = subprocess.run(
            [blender_path, "--background", "--python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if not result_file.exists():
            logger.warning(f"Blender manifold test failed: {result.stderr}")
            # Fall back to Python implementation
            return test_manifold_python(mesh)
        
        import json
        with open(result_file) as f:
            data = json.load(f)
        
        non_manifold_verts = data["non_manifold_vertices"]
        non_manifold_edges = [tuple(e) for e in data["non_manifold_edges"]]
        boundary_edges = [tuple(e) for e in data["boundary_edges"]]
        wire_edges = [tuple(e) for e in data["wire_edges"]]
        
        is_manifold = (
            len(non_manifold_verts) == 0 and
            len(non_manifold_edges) == 0 and
            len(wire_edges) == 0
        )
        
        return ManifoldTestResult(
            is_manifold=is_manifold,
            non_manifold_vertices=non_manifold_verts,
            non_manifold_edges=non_manifold_edges,
            boundary_edges=boundary_edges,
            wire_edges=wire_edges,
            num_non_manifold_vertices=len(non_manifold_verts),
            num_non_manifold_edges=len(non_manifold_edges),
            num_boundary_edges=len(boundary_edges),
            num_wire_edges=len(wire_edges),
        )


def test_manifold(
    mesh: Mesh,
    use_blender: bool = True
) -> ManifoldTestResult:
    """
    Test if a mesh is manifold.
    
    Args:
        mesh: The mesh to test
        use_blender: Whether to use Blender for comprehensive testing.
                     Falls back to Python if Blender is not available.
    
    Returns:
        ManifoldTestResult with detailed analysis
    """
    if use_blender:
        try:
            return test_manifold_blender(mesh)
        except Exception as e:
            logger.warning(f"Blender manifold test failed, using Python fallback: {e}")
            return test_manifold_python(mesh)
    else:
        return test_manifold_python(mesh)
