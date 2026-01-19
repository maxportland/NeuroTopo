"""
Sample mesh generators for testing.

Provides procedural test meshes with known properties for validation.
"""

from __future__ import annotations

import numpy as np
from meshretopo.core.mesh import Mesh


def create_sphere(
    radius: float = 1.0,
    subdivisions: int = 3,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """
    Create an icosphere mesh.
    
    Args:
        radius: Sphere radius
        subdivisions: Number of subdivision iterations (higher = more faces)
        center: Center position
        
    Returns:
        Triangulated sphere mesh
    """
    # Start with icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float64)
    
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = _subdivide_sphere(vertices, faces)
    
    # Scale and translate
    vertices = vertices * radius + np.array(center)
    
    return Mesh(vertices, faces, name="sphere")


def _subdivide_sphere(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide sphere mesh (loop-like subdivision)."""
    edge_midpoints = {}  # (v0, v1) -> midpoint index
    new_vertices = list(vertices)
    new_faces = []
    
    for face in faces:
        mid_indices = []
        
        for i in range(3):
            v0, v1 = face[i], face[(i + 1) % 3]
            edge = (min(v0, v1), max(v0, v1))
            
            if edge not in edge_midpoints:
                # Create midpoint
                midpoint = (vertices[v0] + vertices[v1]) / 2
                midpoint = midpoint / np.linalg.norm(midpoint)  # Project to sphere
                
                edge_midpoints[edge] = len(new_vertices)
                new_vertices.append(midpoint)
            
            mid_indices.append(edge_midpoints[edge])
        
        # Create 4 new triangles
        v0, v1, v2 = face
        m0, m1, m2 = mid_indices
        
        new_faces.extend([
            [v0, m0, m2],
            [m0, v1, m1],
            [m2, m1, v2],
            [m0, m1, m2],
        ])
    
    return np.array(new_vertices), np.array(new_faces)


def create_cube(
    size: float = 1.0,
    subdivisions: int = 2,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """
    Create a subdivided cube mesh.
    
    Args:
        size: Cube size (edge length)
        subdivisions: Subdivisions per face edge
        center: Center position
        
    Returns:
        Triangulated cube mesh
    """
    half = size / 2
    n = subdivisions + 1  # Vertices per edge
    
    vertices = []
    faces = []
    
    # Generate each face
    face_configs = [
        # (normal_axis, normal_sign, u_axis, v_axis)
        (2, 1, 0, 1),   # +Z
        (2, -1, 0, 1),  # -Z
        (1, 1, 0, 2),   # +Y
        (1, -1, 0, 2),  # -Y
        (0, 1, 1, 2),   # +X
        (0, -1, 1, 2),  # -X
    ]
    
    for normal_axis, normal_sign, u_axis, v_axis in face_configs:
        base_idx = len(vertices)
        
        # Generate vertices for this face
        for i in range(n):
            for j in range(n):
                u = -half + size * i / (n - 1)
                v = -half + size * j / (n - 1)
                
                pos = [0, 0, 0]
                pos[normal_axis] = half * normal_sign
                pos[u_axis] = u
                pos[v_axis] = v
                
                vertices.append(pos)
        
        # Generate faces
        for i in range(n - 1):
            for j in range(n - 1):
                v00 = base_idx + i * n + j
                v01 = base_idx + i * n + (j + 1)
                v10 = base_idx + (i + 1) * n + j
                v11 = base_idx + (i + 1) * n + (j + 1)
                
                if normal_sign > 0:
                    faces.extend([
                        [v00, v01, v11],
                        [v00, v11, v10],
                    ])
                else:
                    faces.extend([
                        [v00, v11, v01],
                        [v00, v10, v11],
                    ])
    
    vertices = np.array(vertices) + np.array(center)
    faces = np.array(faces)
    
    return Mesh(vertices, faces, name="cube")


def create_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    major_segments: int = 32,
    minor_segments: int = 16,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """
    Create a torus mesh.
    
    Args:
        major_radius: Distance from center to tube center
        minor_radius: Tube radius
        major_segments: Segments around the main ring
        minor_segments: Segments around the tube
        center: Center position
        
    Returns:
        Triangulated torus mesh
    """
    vertices = []
    faces = []
    
    for i in range(major_segments):
        theta = 2 * np.pi * i / major_segments
        
        # Center of tube cross-section
        cx = major_radius * np.cos(theta)
        cy = major_radius * np.sin(theta)
        
        for j in range(minor_segments):
            phi = 2 * np.pi * j / minor_segments
            
            # Point on tube surface
            x = cx + minor_radius * np.cos(phi) * np.cos(theta)
            y = cy + minor_radius * np.cos(phi) * np.sin(theta)
            z = minor_radius * np.sin(phi)
            
            vertices.append([x, y, z])
    
    # Generate faces
    for i in range(major_segments):
        i_next = (i + 1) % major_segments
        
        for j in range(minor_segments):
            j_next = (j + 1) % minor_segments
            
            v00 = i * minor_segments + j
            v01 = i * minor_segments + j_next
            v10 = i_next * minor_segments + j
            v11 = i_next * minor_segments + j_next
            
            faces.extend([
                [v00, v10, v11],
                [v00, v11, v01],
            ])
    
    vertices = np.array(vertices) + np.array(center)
    faces = np.array(faces)
    
    return Mesh(vertices, faces, name="torus")


def create_cylinder(
    radius: float = 0.5,
    height: float = 2.0,
    radial_segments: int = 32,
    height_segments: int = 4,
    caps: bool = True,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """
    Create a cylinder mesh.
    
    Args:
        radius: Cylinder radius
        height: Cylinder height
        radial_segments: Segments around circumference
        height_segments: Segments along height
        caps: Include end caps
        center: Center position
        
    Returns:
        Triangulated cylinder mesh
    """
    vertices = []
    faces = []
    
    half_h = height / 2
    
    # Generate side vertices
    for i in range(height_segments + 1):
        z = -half_h + height * i / height_segments
        
        for j in range(radial_segments):
            theta = 2 * np.pi * j / radial_segments
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            vertices.append([x, y, z])
    
    # Generate side faces
    for i in range(height_segments):
        for j in range(radial_segments):
            j_next = (j + 1) % radial_segments
            
            v00 = i * radial_segments + j
            v01 = i * radial_segments + j_next
            v10 = (i + 1) * radial_segments + j
            v11 = (i + 1) * radial_segments + j_next
            
            faces.extend([
                [v00, v01, v11],
                [v00, v11, v10],
            ])
    
    # Add caps
    if caps:
        # Bottom cap
        bottom_center_idx = len(vertices)
        vertices.append([0, 0, -half_h])
        
        bottom_start = 0
        for j in range(radial_segments):
            j_next = (j + 1) % radial_segments
            faces.append([bottom_center_idx, bottom_start + j_next, bottom_start + j])
        
        # Top cap
        top_center_idx = len(vertices)
        vertices.append([0, 0, half_h])
        
        top_start = height_segments * radial_segments
        for j in range(radial_segments):
            j_next = (j + 1) % radial_segments
            faces.append([top_center_idx, top_start + j, top_start + j_next])
    
    vertices = np.array(vertices) + np.array(center)
    faces = np.array(faces)
    
    return Mesh(vertices, faces, name="cylinder")


def create_plane(
    width: float = 2.0,
    height: float = 2.0,
    subdivisions_x: int = 10,
    subdivisions_y: int = 10,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """Create a flat plane mesh."""
    vertices = []
    faces = []
    
    nx = subdivisions_x + 1
    ny = subdivisions_y + 1
    
    half_w = width / 2
    half_h = height / 2
    
    for i in range(nx):
        for j in range(ny):
            x = -half_w + width * i / (nx - 1)
            y = -half_h + height * j / (ny - 1)
            vertices.append([x, y, 0])
    
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v01 = i * ny + (j + 1)
            v10 = (i + 1) * ny + j
            v11 = (i + 1) * ny + (j + 1)
            
            faces.extend([
                [v00, v01, v11],
                [v00, v11, v10],
            ])
    
    vertices = np.array(vertices) + np.array(center)
    faces = np.array(faces)
    
    return Mesh(vertices, faces, name="plane")


def add_noise(mesh: Mesh, amount: float = 0.01) -> Mesh:
    """Add random noise to vertex positions."""
    noise = np.random.randn(*mesh.vertices.shape) * amount * mesh.diagonal
    noisy_verts = mesh.vertices + noise
    
    return Mesh(
        noisy_verts,
        mesh.faces.copy(),
        name=f"{mesh.name}_noisy"
    )


def create_cone(
    radius: float = 0.5,
    height: float = 1.5,
    segments: int = 32,
    height_segments: int = 8,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """
    Create a cone mesh.
    
    Args:
        radius: Base radius
        height: Cone height
        segments: Radial segments
        height_segments: Height segments
        center: Center position
        
    Returns:
        Triangulated cone mesh
    """
    vertices = []
    faces = []
    
    # Apex at top
    apex_idx = 0
    vertices.append([0, 0, height])
    
    # Generate rings from top to bottom
    for i in range(height_segments):
        # Fraction from apex (0) to base (1)
        t = (i + 1) / height_segments
        z = height * (1 - t)
        r = radius * t
        
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append([x, y, z])
    
    # Generate side faces
    # Top triangles (connecting to apex)
    ring_start = 1
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([apex_idx, ring_start + j, ring_start + j_next])
    
    # Quad strips for rest of cone
    for i in range(height_segments - 1):
        start = 1 + i * segments
        next_start = start + segments
        
        for j in range(segments):
            j_next = (j + 1) % segments
            
            v00 = start + j
            v01 = start + j_next
            v10 = next_start + j
            v11 = next_start + j_next
            
            faces.extend([
                [v00, v10, v11],
                [v00, v11, v01],
            ])
    
    # Base cap
    base_center_idx = len(vertices)
    vertices.append([0, 0, 0])
    
    base_start = 1 + (height_segments - 1) * segments
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append([base_center_idx, base_start + j_next, base_start + j])
    
    vertices = np.array(vertices) + np.array(center)
    faces = np.array(faces)
    
    return Mesh(vertices, faces, name="cone")


def create_bunny_like(
    base_radius: float = 0.5,
    subdivisions: int = 3,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """
    Create a bunny-like organic mesh (simplified for testing).
    
    This creates an organic looking mesh with varying curvature,
    useful for testing curvature-adaptive remeshing.
    """
    # Start with sphere
    mesh = create_sphere(radius=base_radius, subdivisions=subdivisions, center=(0, 0, 0))
    vertices = mesh.vertices.copy()
    
    # Apply deformations to create organic shape
    for i, v in enumerate(vertices):
        x, y, z = v
        
        # Body elongation
        z_scale = 1.0 + 0.3 * np.exp(-3 * (z + 0.3) ** 2)
        
        # Head bulge
        head_factor = np.exp(-5 * ((z - 0.4) ** 2 + x ** 2))
        
        # Ear-like protrusions
        ear_left = 0.2 * np.exp(-10 * ((x + 0.15) ** 2 + (z - 0.5) ** 2)) * max(0, y)
        ear_right = 0.2 * np.exp(-10 * ((x - 0.15) ** 2 + (z - 0.5) ** 2)) * max(0, y)
        
        vertices[i] = [
            x * (1 + 0.1 * head_factor),
            y * (1 + 0.15 * head_factor) + ear_left + ear_right,
            z * z_scale
        ]
    
    vertices = vertices + np.array(center)
    
    return Mesh(vertices, mesh.faces.copy(), name="bunny_like")


def create_mechanical_part(
    size: float = 1.0,
    detail_level: int = 2,
    center: tuple[float, float, float] = (0, 0, 0)
) -> Mesh:
    """
    Create a mechanical-looking part with sharp edges and flat surfaces.
    
    Useful for testing feature detection and edge preservation.
    """
    # Create by combining primitives
    all_vertices = []
    all_faces = []
    
    # Main body (cube-ish)
    body = create_cube(size=size * 0.8, subdivisions=detail_level, center=(0, 0, 0))
    all_vertices.extend(body.vertices)
    all_faces.extend(body.faces)
    
    # Add a cylindrical hole simulation (actually a smaller cube subtracted)
    # For simplicity, we add a protruding cylinder
    cyl = create_cylinder(
        radius=size * 0.15,
        height=size * 0.4,
        radial_segments=16 * detail_level,
        height_segments=2,
        caps=True,
        center=(size * 0.3, 0, size * 0.5)
    )
    
    offset = len(all_vertices)
    all_vertices.extend(cyl.vertices)
    all_faces.extend(cyl.faces + offset)
    
    # Add another cylinder on the side
    cyl2 = create_cylinder(
        radius=size * 0.1,
        height=size * 0.3,
        radial_segments=12 * detail_level,
        height_segments=2,
        caps=True,
        center=(-size * 0.3, size * 0.4, 0)
    )
    
    offset = len(all_vertices)
    all_vertices.extend(cyl2.vertices)
    all_faces.extend(cyl2.faces + offset)
    
    vertices = np.array(all_vertices) + np.array(center)
    faces = np.array(all_faces)
    
    return Mesh(vertices, faces, name="mechanical_part")


# Collection of all test meshes
def get_test_meshes() -> dict[str, Mesh]:
    """Get a dictionary of all test meshes."""
    return {
        "sphere_low": create_sphere(subdivisions=2),
        "sphere_high": create_sphere(subdivisions=4),
        "cube_low": create_cube(subdivisions=1),
        "cube_high": create_cube(subdivisions=4),
        "torus": create_torus(),
        "cylinder": create_cylinder(),
        "cone": create_cone(),
        "plane": create_plane(),
        "bunny_like": create_bunny_like(),
        "mechanical_part": create_mechanical_part(),
    }


def save_test_meshes(output_dir: str = "test_meshes") -> None:
    """Save all test meshes to files."""
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meshes = get_test_meshes()
    for name, mesh in meshes.items():
        mesh.to_file(output_dir / f"{name}.obj")
        print(f"Saved {name}: {mesh.num_vertices} verts, {mesh.num_faces} faces")
