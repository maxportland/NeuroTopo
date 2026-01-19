#!/usr/bin/env python3
"""Test the full SemanticAnalyzer with OpenAI Vision API."""

import os
import logging
logging.basicConfig(level=logging.DEBUG)

from neurotopo.core.mesh import Mesh
from neurotopo.analysis.semantic import SemanticAnalyzer

# Load mesh
mesh = Mesh.from_file('test_mesh.obj')
print(f'Mesh: {mesh.num_vertices:,} verts, {mesh.num_faces:,} faces')

# Create analyzer
analyzer = SemanticAnalyzer(
    api_provider='openai',
    model='gpt-4o',
    use_blender=True,
    resolution=(1024, 1024),
    num_views=2,  # Front and side views
)

print('Analyzing with GPT-4o vision...')
segmentation = analyzer.analyze(mesh)

print(f'\nDetected {len(segmentation.segments)} regions:')
for seg in segmentation.segments:
    print(f'  - {seg.region_type.value}: {seg.num_faces:,} faces, confidence: {seg.confidence:.2f}')
