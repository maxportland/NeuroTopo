#!/usr/bin/env python3
"""Compare source and retopo mesh topology using quad-aware analysis."""

from collections import Counter

def load_obj(path):
    """Load OBJ file preserving quads."""
    vertices = []
    faces = []
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'f':
            faces.append([int(x.split('/')[0])-1 for x in parts[1:]])
    return vertices, faces

def analyze(vertices, faces, name):
    """Analyze topology treating mesh as quad-dominant."""
    from collections import defaultdict
    
    # Build edge-to-vertex mapping (fast)
    vertex_edges = defaultdict(int)
    seen_edges = set()
    
    for face in faces:
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i+1) % n]
            if v0 != v1:
                edge = (min(v0, v1), max(v0, v1))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    vertex_edges[v0] += 1
                    vertex_edges[v1] += 1
    
    # Compute valence histogram
    valence = Counter()
    for vi in range(len(vertices)):
        valence[vertex_edges[vi]] += 1
    
    total = len(vertices)
    regular = valence.get(4, 0)
    high_poles = sum(valence.get(v, 0) for v in range(6, max(valence.keys())+1))
    
    # Count face types
    quads = sum(1 for f in faces if len(f) == 4 and (len(set(f)) == 4))
    degen_quads = sum(1 for f in faces if len(f) == 4 and len(set(f)) < 4)
    tris = sum(1 for f in faces if len(f) == 3)
    
    print(f"\n{name}")
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"  Face types: {quads} quads, {tris} tris, {degen_quads} degen")
    print(f"  Regular (valence 4): {regular} ({100*regular/total:.1f}%)")
    print(f"  High poles (valence 6+): {high_poles} ({100*high_poles/total:.1f}%)")
    
    return 100*regular/total, 100*high_poles/total

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("QUAD-AWARE TOPOLOGY COMPARISON")
    print("=" * 60)
    
    # Get paths from command line or use defaults
    if len(sys.argv) >= 3:
        source_path = sys.argv[1]
        retopo_path = sys.argv[2]
    else:
        source_path = '/tmp/source_mesh_quads.obj'
        retopo_path = '/Users/maxdavis/Projects/MeshRepair/results/20260119_192039/test_mesh2_retopo.obj'
    
    # Handle FBX files by converting through Blender first
    import os
    import tempfile
    import subprocess
    
    actual_source = source_path
    if source_path.lower().endswith('.fbx'):
        # Convert FBX to OBJ
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            temp_obj = f.name
        
        script = f'''
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"{source_path}")
bpy.ops.wm.obj_export(filepath=r"{temp_obj}", export_selected_objects=False, export_triangulated_mesh=False)
'''
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as sf:
            sf.write(script)
            script_path = sf.name
        
        subprocess.run(
            ["/Applications/Blender.app/Contents/MacOS/blender", "--background", "--python", script_path],
            capture_output=True, timeout=60
        )
        actual_source = temp_obj
        os.unlink(script_path)
    
    # Analyze source
    sv, sf = load_obj(actual_source)
    src_reg, src_high = analyze(sv, sf, f"SOURCE ({os.path.basename(source_path)})")
    
    # Clean up temp file if created
    if source_path.lower().endswith('.fbx'):
        os.unlink(actual_source)
    
    # Analyze retopo
    rv, rf = load_obj(retopo_path)
    ret_reg, ret_high = analyze(rv, rf, "RETOPO OUTPUT")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"Face reduction: {len(sf)} -> {len(rf)} ({100*len(rf)/len(sf):.1f}%)")
    print(f"Vertex reduction: {len(sv)} -> {len(rv)} ({100*len(rv)/len(sv):.1f}%)")
    print()
    
    reg_better = ret_reg >= src_reg  # Same or better
    high_better = ret_high <= src_high  # Same or better (fewer is better)
    
    print(f"Regular vertices: {src_reg:.1f}% -> {ret_reg:.1f}% ", end="")
    if ret_reg > src_reg:
        print("✓ BETTER")
    elif ret_reg == src_reg:
        print("= SAME")
    else:
        print("✗ WORSE")
    
    print(f"High poles: {src_high:.1f}% -> {ret_high:.1f}% ", end="")
    if ret_high < src_high:
        print("✓ BETTER")
    elif ret_high == src_high:
        print("= SAME")
    else:
        print("✗ WORSE")
    
    if ret_reg > src_reg and ret_high < src_high:
        print("\n🎉 RETOPO OUTPUT HAS BETTER TOPOLOGY THAN SOURCE!")
    elif reg_better and high_better:
        print("\n✓ RETOPO OUTPUT HAS EQUIVALENT OR BETTER TOPOLOGY")
    elif reg_better or high_better:
        # Check if this is acceptable given face reduction
        face_reduction = 1 - len(rf) / len(sf)
        
        if face_reduction > 0.3 and ret_reg > 80:
            print(f"\n✓ ACCEPTABLE: {face_reduction*100:.0f}% face reduction with {ret_reg:.1f}% regular vertices")
        else:
            print("\n⚠️  RETOPO OUTPUT HAS MIXED RESULTS")
    else:
        # Check if still acceptable
        face_reduction = 1 - len(rf) / len(sf)
        if face_reduction > 0.3 and ret_reg > 80:
            print(f"\n✓ ACCEPTABLE: {face_reduction*100:.0f}% face reduction with {ret_reg:.1f}% regular vertices")
        elif face_reduction > 0.5 and ret_reg > 70:
            print(f"\n✓ ACCEPTABLE: {face_reduction*100:.0f}% face reduction with {ret_reg:.1f}% regular vertices")
        else:
            print("\n❌ RETOPO OUTPUT HAS WORSE TOPOLOGY THAN SOURCE")
