#!/usr/bin/env python3
"""
MeshRetopo Status Report

Quick overview of the system capabilities and current performance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    from meshretopo.test_meshes import create_sphere, create_torus, create_cube
    from meshretopo.pipeline import RetopoPipeline
    from meshretopo.remesh import get_remesher
    
    print("=" * 60)
    print("MESHRETOPO - AI-Assisted Retopology System")
    print("=" * 60)
    
    # Available backends
    print("\n📦 Available Backends:")
    backends = ["trimesh", "hybrid", "guided_quad", "isotropic"]
    for b in backends:
        try:
            r = get_remesher(b)
            quad_support = "✓ quads" if r.supports_quads else "  tris"
            print(f"   • {b:15} [{quad_support}]")
        except Exception as e:
            print(f"   • {b:15} [unavailable: {e}]")
    
    # Quick benchmark
    print("\n📊 Quick Benchmark (sphere, 1280→~200 faces):")
    print("-" * 50)
    
    mesh = create_sphere(subdivisions=3)
    target = int(mesh.num_faces * 0.15)
    
    results = []
    for backend in ["trimesh", "hybrid", "guided_quad"]:
        try:
            pipeline = RetopoPipeline(backend=backend, target_faces=target)
            out, score = pipeline.process(mesh, evaluate=True)
            results.append({
                "backend": backend,
                "score": score.overall_score,
                "quad": score.quad_score,
                "fidelity": score.fidelity_score,
                "faces": out.num_faces
            })
            print(f"   {backend:15} Score: {score.overall_score:5.1f} "
                  f"(Q:{score.quad_score:4.1f} F:{score.fidelity_score:4.1f}) "
                  f"→ {out.num_faces} faces")
        except Exception as e:
            print(f"   {backend:15} FAILED: {e}")
    
    if results:
        best = max(results, key=lambda r: r["score"])
        print(f"\n🏆 Best: {best['backend']} ({best['score']:.1f}/100)")
    
    # Project stats
    print("\n📁 Project Structure:")
    src_path = Path(__file__).parent / "src" / "meshretopo"
    
    py_files = list(src_path.rglob("*.py"))
    total_lines = 0
    for f in py_files:
        total_lines += sum(1 for _ in open(f))
    
    print(f"   • Python files: {len(py_files)}")
    print(f"   • Total lines:  {total_lines:,}")
    
    # Next steps
    print("\n🔮 Suggested Next Steps:")
    print("   1. Improve quad quality metrics (currently ~25-40)")
    print("   2. Add real neural network when PyTorch available")
    print("   3. Implement field-aligned meshing algorithms")
    print("   4. Add edge flow optimization for animation")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
