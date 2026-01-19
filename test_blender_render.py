#!/usr/bin/env python3
"""Test Blender rendering integration."""

from meshretopo.core.mesh import Mesh
from meshretopo.analysis.semantic import SemanticAnalyzer
from meshretopo.analysis.blender_render import BlenderRenderer, find_blender

# Check Blender
blender_path = find_blender()
print(f'Blender found: {blender_path}')

# Test Blender renderer
renderer = BlenderRenderer(resolution=(1024, 1024))
print(f'Blender available: {renderer.available}')

# Load mesh
mesh = Mesh.from_file('test_mesh.obj')
print(f'Mesh: {mesh.num_vertices:,} verts, {mesh.num_faces:,} faces')

# Test analyzer with Blender
print('\nTesting semantic analyzer with Blender...')
analyzer = SemanticAnalyzer(
    resolution=(1024, 1024),
    num_views=2,  # Quick test with 2 views
    use_blender=True,
)

renders, params = analyzer._render_views(mesh)
print(f'Rendered {len(renders)} views')
if renders:
    print(f'Render shape: {renders[0].shape}')
    print(f'View params: {params[0]}')
    
    # Save a sample
    from PIL import Image
    img = Image.fromarray(renders[0])
    img.save('blender_test_render.png')
    print('Saved blender_test_render.png')
else:
    print('No renders produced - check Blender output')
