#!/usr/bin/env python3
"""Test OpenAI Vision API directly."""

import os
import base64
from openai import OpenAI
from neurotopo.core.mesh import Mesh
from neurotopo.analysis.blender_render import BlenderRenderer

# Load mesh info
mesh = Mesh.from_file('test_mesh.obj')
print(f'Mesh: {mesh.num_vertices:,} verts')

# Render with Blender
renderer = BlenderRenderer(resolution=(1024, 1024))
renders, params = renderer.render_mesh('test_mesh.obj')

print(f'Rendered {len(renders)} views')
if renders:
    # Save first render for inspection
    from PIL import Image
    import numpy as np
    
    img = Image.fromarray(renders[0])
    img.save('test_render.png')
    print(f'Saved render to test_render.png')
    print(f'View 0: {params[0]}')
    
    # Convert to PNG bytes
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    
    # Call OpenAI Vision directly
    client = OpenAI()
    
    prompt = '''This is a 3D mesh rendering. Identify the visible anatomical regions and their approximate locations.

For each region, estimate the center position as normalized x,y coordinates (0,0 is top-left, 1,1 is bottom-right).

Return a JSON array with this exact format:
[{"region_type": "FACE", "center_2d": [0.5, 0.5], "approximate_radius": 0.3, "confidence": 0.9}]

Valid region types: FACE, EYE_SOCKET, NOSE, MOUTH, FOREHEAD, CHIN, CHEEK, NECK, EAR

Respond with ONLY the JSON array, no other text.'''
    
    # Encode image
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print('Calling GPT-4o...')
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}', 'detail': 'high'}}
            ]
        }],
        max_tokens=2000
    )
    
    # Print raw response
    print('\n=== RAW RESPONSE ===')
    print(response.choices[0].message.content)
    
    # Test parsing
    import json
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        # Remove markdown code blocks
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines 
            if not line.startswith("```")
        )
    
    print('\n=== PARSED JSON ===')
    data = json.loads(text)
    print(f'Got {len(data)} detections')
    for item in data:
        print(f'  - {item}')
