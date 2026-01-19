#!/usr/bin/env python3
"""Debug the remesh pipeline."""
import traceback
import numpy as np
from meshretopo.core.mesh import Mesh
from meshretopo import RetopoPipeline

try:
    mesh = Mesh.from_file('test_mesh.obj')
    pipeline = RetopoPipeline(backend='hybrid', target_faces=5000)
    output, score = pipeline.process(mesh, enable_timing=False)
    print(f"Success! {output.num_faces} faces")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
