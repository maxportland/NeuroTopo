#!/usr/bin/env python3
"""Save mesh results from benchmark tests."""

import os
from datetime import datetime
from neurotopo import RetopoPipeline
from neurotopo.core.mesh import Mesh

# Create timestamped folder
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f'results/{timestamp}'
os.makedirs(output_dir, exist_ok=True)

print(f'Saving results to: {output_dir}')

# Load input mesh
mesh = Mesh.from_file('test_mesh.obj')
print(f'Input: {mesh.num_vertices:,} vertices, {mesh.num_faces:,} faces')

results = []

# Run and save results for each backend
for backend, target in [('trimesh', 5000), ('hybrid', 3500)]:
    print(f'\nProcessing {backend}...')
    pipeline = RetopoPipeline(backend=backend, target_faces=target)
    output, score = pipeline.process(mesh, enable_timing=False)
    
    # Save mesh
    output_path = f'{output_dir}/{backend}_result.obj'
    output.to_file(output_path)
    print(f'  Saved: {output_path}')
    print(f'  Faces: {output.num_faces:,}, Score: {score.overall_score:.1f}')
    
    results.append({
        'backend': backend,
        'target': target,
        'vertices': output.num_vertices,
        'faces': output.num_faces,
        'score': score
    })

# Save detailed summary
summary_path = f'{output_dir}/summary.txt'
with open(summary_path, 'w') as f:
    f.write(f'NeuroTopo Results - {timestamp}\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'Input: test_mesh.obj\n')
    f.write(f'  Vertices: {mesh.num_vertices:,}\n')
    f.write(f'  Faces: {mesh.num_faces:,}\n\n')
    
    for r in results:
        score = r['score']
        f.write(f'{r["backend"].upper()} Backend (target={r["target"]})\n')
        f.write('-' * 40 + '\n')
        f.write(f'  Output: {r["vertices"]:,} vertices, {r["faces"]:,} faces\n')
        f.write(f'  Overall Score: {score.overall_score:.1f}/100\n')
        f.write(f'  Quad Score: {score.quad_score:.1f}/100\n')
        f.write(f'  Fidelity Score: {score.fidelity_score:.1f}/100\n')
        f.write(f'  Topology Score: {score.topology_score:.1f}/100\n')
        f.write(f'  Visual Score: {score.visual_score:.1f}/100\n')
        f.write(f'  Manifold: {score.topology.is_manifold}\n')
        f.write(f'  File: {r["backend"]}_result.obj\n\n')

print(f'\nSummary saved to: {summary_path}')
print(f'\nAll results saved to: {output_dir}/')

# List contents
print('\nFolder contents:')
for f in os.listdir(output_dir):
    path = os.path.join(output_dir, f)
    size = os.path.getsize(path)
    print(f'  {f}: {size:,} bytes')
