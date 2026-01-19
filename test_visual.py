#!/usr/bin/env python3
"""Test visual evaluation with render output."""
import os
import numpy as np
from meshretopo.core.mesh import Mesh
from meshretopo import RetopoPipeline
from meshretopo.evaluation.visual import VisualEvaluator

# Create output directory
os.makedirs('renders', exist_ok=True)

print("Loading mesh...")
mesh = Mesh.from_file('test_mesh.obj')

for backend in ['trimesh', 'hybrid']:
    print(f"\nProcessing with {backend} backend...")
    target = 5000 if backend == 'trimesh' else 3500
    pipeline = RetopoPipeline(backend=backend, target_faces=target)
    output, score = pipeline.process(mesh, enable_timing=False)
    
    print(f"  Output: {output.num_faces} faces")
    print(f"  Visual Score: {score.visual_score:.1f}/100")
    
    if score.visual_quality:
        print(f"    Shading:     {score.visual_quality.shading_smoothness*100:.1f}%")
        print(f"    Edge qual:   {score.visual_quality.edge_visibility*100:.1f}%")
        print(f"    Silhouette:  {score.visual_quality.silhouette_quality*100:.1f}%")
        print(f"    Consistency: {score.visual_quality.render_consistency*100:.1f}%")
    
    # Save a render
    evaluator = VisualEvaluator(resolution=(1024, 1024), num_views=1)
    renders = evaluator._render_views(output)
    
    if renders:
        try:
            from PIL import Image
            img = Image.fromarray(renders[0])
            img.save(f'renders/{backend}_front.png')
            print(f"  Saved render to renders/{backend}_front.png")
        except ImportError:
            print("  (PIL not available for saving renders)")
