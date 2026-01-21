"""
Hybrid remeshing backend.

Combines high-quality triangle generation with intelligent
quad conversion for production-quality output.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.guidance.composer import GuidanceFields
from neurotopo.remesh.base import Remesher, RemeshResult
from neurotopo.remesh.tri_to_quad import TriToQuadConverter, SmartQuadConverter

logger = logging.getLogger("neurotopo.remesh.hybrid")


class HybridRemesher(Remesher):
    """
    Hybrid remesher that produces high-quality quads.
    
    Strategy:
    1. Use fast decimation (fast_simplification or trimesh)
    2. Convert triangles to quads using optimal pairing with valence awareness
    3. Reduce high-valence poles
    4. Light optimization for quad shapes
    """
    
    def __init__(
        self,
        quad_ratio: float = 0.8,  # Target ratio of quads vs tris
        optimization_passes: int = 3,  # Reduced for speed
        preserve_boundary: bool = True,
        pole_reduction: bool = True,  # Enable pole reduction
        **kwargs
    ):
        self.quad_ratio = quad_ratio
        self.optimization_passes = optimization_passes
        self.preserve_boundary = preserve_boundary
        self.pole_reduction = pole_reduction
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    @property
    def supports_quads(self) -> bool:
        return True
    
    def remesh(
        self,
        mesh: Mesh,
        guidance: GuidanceFields,
        **kwargs
    ) -> RemeshResult:
        """Perform hybrid remeshing."""
        start_time = time.time()
        
        try:
            target_faces = guidance.target_face_count or mesh.num_faces // 4
            
            # Check if input is high-quality quads - if so, use topology-preserving path
            if mesh.is_quad:
                quality_score = self._assess_quad_quality(mesh)
                
                # For high-quality quads (>85%), use topology-preserving reduction
                # This removes edge loops in flat areas instead of collapsing edges
                if quality_score > 0.85:
                    logger.info(f"Input has high-quality quads ({quality_score:.1%} regular), "
                               f"using topology-preserving reduction")
                    return self._topology_preserving_reduce(mesh, target_faces, start_time, quality_score)
            
            # Standard path: triangulate and rebuild quads
            # Request 2x triangles since we'll pair them into quads
            target_tris = int(target_faces * 2)
            
            tri_mesh = mesh.triangulate() if not mesh.is_triangular else mesh
            verts, faces = self._fast_decimate(tri_mesh, target_tris)
            
            # Step 1.5: Improve triangle quality with edge relaxation
            try:
                verts, faces = self._improve_triangles(verts, faces, iterations=20)
            except Exception as e:
                logger.debug(f"Triangle improvement failed: {e}")
            
            # Step 2: Convert triangles to quads with valence-aware pairing
            converter = TriToQuadConverter(
                min_quality=0.35, 
                prefer_regular=True,
                valence_weight=0.4,  # Prioritize valence regularity
            )
            verts, quad_faces, remaining_tris = converter.convert(verts, faces)
            
            # Build output - store quads and triangles separately
            # Use object array to allow mixed face sizes
            all_faces = []
            
            for qf in quad_faces:
                all_faces.append(np.array(qf, dtype=np.int32))  # Quad as 4 vertices
            
            for tf in remaining_tris:
                all_faces.append(np.array(tf, dtype=np.int32))  # Triangle as 3 vertices
            
            # For now, still need to convert to homogeneous array for Mesh class
            # Convert triangles to degenerate quads for compatibility
            homo_faces = []
            for face in all_faces:
                if len(face) == 4:
                    homo_faces.append(face)
                elif len(face) == 3:
                    # Store as degenerate quad for array compatibility
                    homo_faces.append([face[0], face[1], face[2], face[2]])
            
            faces = np.array(homo_faces, dtype=np.int32)
            
            # Step 2.5: Pole reduction - reduce high-valence vertices
            if self.pole_reduction and len(faces) < 20000:
                try:
                    from neurotopo.postprocess.pole_reduction import PoleReducer
                    reducer = PoleReducer(
                        max_valence=5,  # Target: reduce vertices with valence > 5
                        iterations=2,
                        preserve_boundary=self.preserve_boundary,
                    )
                    verts, faces = reducer.reduce(verts, faces)
                    logger.debug(f"Pole reduction complete: {len(faces)} faces")
                except Exception as pr_err:
                    logger.debug(f"Pole reduction skipped: {pr_err}")
            
            # Step 3: Light optimization (only if small enough)
            # Skip for large meshes - the optimizer is too slow
            if len(faces) < 3000 and self.optimization_passes > 0:
                try:
                    import trimesh
                    original_tm = trimesh.Trimesh(
                        vertices=tri_mesh.vertices,
                        faces=tri_mesh.faces,
                        process=False
                    )
                    
                    from neurotopo.postprocess import QuadOptimizer
                    optimizer = QuadOptimizer(
                        iterations=min(self.optimization_passes, 2),
                        smoothing_weight=0.3,
                        surface_weight=0.7,
                    )
                    verts = optimizer.optimize(verts, faces, original_tm)
                except Exception as opt_err:
                    logger.debug(f"Optimization skipped: {opt_err}")
            
            # Final manifold repair after all processing
            from neurotopo.postprocess.manifold import ManifoldRepair
            repair = ManifoldRepair(verbose=False)
            verts, faces = repair.repair(verts, faces)
            
            output = Mesh(
                vertices=verts,
                faces=faces,
                name=f"{mesh.name}_hybrid"
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Hybrid remesh: {mesh.num_faces} -> {output.num_faces} faces "
                       f"({len(quad_faces)} quads, {len(remaining_tris)} tris) in {elapsed:.2f}s")
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                iterations=self.optimization_passes,
                time_seconds=elapsed,
                metadata={
                    "backend": "hybrid",
                    "quad_count": len(quad_faces),
                    "tri_count": len(remaining_tris)
                }
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Hybrid remesh failed after {elapsed:.2f}s: {e}")
            return RemeshResult(
                mesh=mesh,
                success=False,
                actual_face_count=mesh.num_faces,
                time_seconds=elapsed,
                metadata={"error": str(e)}
            )
    
    def _assess_quad_quality(self, mesh: Mesh) -> float:
        """Assess the quality of a quad mesh.
        
        Returns fraction of vertices with regular valence (4 edges).
        A mesh with >90% regular vertices is considered high-quality.
        """
        if not mesh.is_quad:
            return 0.0
        
        # Count edges per vertex using edge-based valence
        n_verts = mesh.num_vertices
        edge_valence = np.zeros(n_verts, dtype=np.int32)
        edge_set = set()
        
        for face in mesh.faces:
            # Handle degenerate quads (triangles stored as quads)
            unique_verts = list(dict.fromkeys(face))
            n_unique = len(unique_verts)
            
            for i in range(n_unique):
                v0, v1 = unique_verts[i], unique_verts[(i + 1) % n_unique]
                if v0 != v1:
                    edge = (min(v0, v1), max(v0, v1))
                    if edge not in edge_set:
                        edge_set.add(edge)
                        edge_valence[v0] += 1
                        edge_valence[v1] += 1
        
        # Count regular vertices (valence 4 for quads)
        regular_count = np.sum(edge_valence == 4)
        # Exclude boundary/isolated vertices
        non_zero = np.sum(edge_valence > 0)
        
        if non_zero == 0:
            return 0.0
        
        return regular_count / non_zero
    
    def _topology_preserving_reduce(
        self,
        mesh: Mesh, 
        target_faces: int,
        start_time: float,
        quality_score: float
    ) -> RemeshResult:
        """Reduce high-quality mesh by removing edge loops in low-curvature areas.
        
        For meshes with excellent topology (>95% regular), standard decimation
        degrades quality. Instead, we use Blender's edge loop removal which
        properly removes entire loops while preserving quad structure.
        
        Based on topology guidelines:
        - Add more polygons where there's curvature
        - Use fewer polygons in flat areas
        - Focus geometry on areas that affect the model's outline
        """
        # Use Blender's edge loop removal - it's much more robust than manual implementation
        try:
            output = self._blender_loop_decimate(mesh, target_faces)
            output_quality = self._assess_quad_quality(output)
            elapsed = time.time() - start_time
            
            logger.info(f"Loop-based decimate: {mesh.num_faces} -> {output.num_faces} faces "
                       f"({output_quality:.1%} regular) in {elapsed:.2f}s")
            
            if output_quality < quality_score * 0.8:
                logger.warning(f"Significant quality degradation: {quality_score:.1%} -> {output_quality:.1%}")
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                time_seconds=elapsed,
                metadata={
                    "backend": "hybrid_loop_removal",
                    "method": "blender_edge_loops",
                    "input_quality": quality_score,
                    "output_quality": output_quality,
                }
            )
        except Exception as e:
            logger.warning(f"Blender loop decimate failed: {e}")
            elapsed = time.time() - start_time
            
            # Return original mesh if all else fails
            return RemeshResult(
                mesh=mesh,
                success=True,
                actual_face_count=mesh.num_faces,
                time_seconds=elapsed,
                metadata={
                    "backend": "hybrid_passthrough",
                    "method": "fallback",
                    "input_quality": quality_score,
                    "error": str(e),
                }
            )
    
    def _blender_loop_decimate(self, mesh: Mesh, target_faces: int) -> Mesh:
        """Use Blender to reduce mesh using quad-friendly operations.
        
        Strategy:
        1. Use Un-Subdivide only - this is the BEST way to reduce subdivided meshes
           while preserving topology quality
        2. Calculate exact iterations needed
        3. Avoid dissolve/collapse which destroy topology
        """
        import tempfile
        import subprocess
        import json
        from pathlib import Path
        
        blender_path = "/Applications/Blender.app/Contents/MacOS/blender"
        
        # First, use neural network to classify poles
        pole_classifications = None
        try:
            from neurotopo.analysis.neural.pole_classifier import (
                HybridPoleClassifier, 
                export_pole_classifications_for_blender
            )
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "pole_classifier.pt"
            if model_path.exists():
                classifier = HybridPoleClassifier(model_path=str(model_path))
                # Get pole classifications using neural network
                poles = classifier.classify_poles(mesh)
                pole_classifications = {
                    "vertices_to_fix": [p.vertex_idx for p in poles 
                                       if p.prediction == 'fix' and p.confidence >= 0.7],
                    "vertices_to_keep": [p.vertex_idx for p in poles 
                                        if p.prediction == 'keep'],
                }
                logger.info(f"Neural network classified {len(pole_classifications['vertices_to_fix'])} "
                           f"poles to fix, {len(pole_classifications['vertices_to_keep'])} to keep")
        except Exception as e:
            logger.debug(f"Neural pole classifier not available: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_obj = tmpdir / "input.obj"
            output_obj = tmpdir / "output.obj"
            
            mesh.to_file(input_obj)
            
            # Calculate iterations: each un-subdivide removes ~75% of faces (1/4 remain)
            # So for ratio r, we need log_4(1/r) iterations
            import math
            ratio = target_faces / mesh.num_faces
            # Try to get closer to target, but cap at 2 to preserve quality
            iterations = max(1, min(2, int(math.ceil(math.log(1/ratio, 4)))))  
            logger.info(f"Un-subdivide: ratio={ratio:.3f}, iterations={iterations}")
            
            # Save pole classifications to file if available
            pole_data_json = "null"
            if pole_classifications:
                pole_data_json = json.dumps(pole_classifications)
            
            script = f'''
import bpy
import bmesh
import json

# Neural network pole classifications
POLE_CLASSIFICATIONS = json.loads('{pole_data_json}')
use_neural_network = POLE_CLASSIFICATIONS is not None
if use_neural_network:
    print(f"Using neural network: {{len(POLE_CLASSIFICATIONS.get('vertices_to_fix', []))}} poles to fix, "
          f"{{len(POLE_CLASSIFICATIONS.get('vertices_to_keep', []))}} to keep")

# Clear and import
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{input_obj}")

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

target_faces = {target_faces}
initial_faces = len(obj.data.polygons)
print(f"Initial: {{initial_faces}} faces, target: {{target_faces}}")

# Count initial quads and tris
initial_quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
initial_tris = sum(1 for p in obj.data.polygons if len(p.vertices) == 3)
print(f"Initial: {{initial_quads}} quads, {{initial_tris}} tris")

# ============================================================
# PRE-PROCESS: Fix source mesh defects BEFORE un-subdivide
# ============================================================
# Converting triangles to quads before un-subdivide prevents
# defects from being amplified during the reduction

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# First, try to convert any triangles to quads
if initial_tris > 0:
    print(f"Pre-process: Converting {{initial_tris}} triangles to quads")
    bpy.ops.mesh.tris_convert_to_quads(
        face_threshold=0.8,  # ~45 degrees - moderately selective
        shape_threshold=0.8,
        uvs=True,
        vcols=True,
        seam=True,
        sharp=True,
        materials=True
    )
    bm = bmesh.from_edit_mesh(obj.data)
    new_tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    print(f"After pre-process: {{new_tris}} triangles remaining")

# Remove any degenerate geometry
bpy.ops.mesh.remove_doubles(threshold=0.0001)
bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

bpy.ops.object.mode_set(mode='OBJECT')

# ============================================================
# UN-SUBDIVIDE: Reduce mesh while preserving edge loops
# ============================================================
iterations = {iterations}
print(f"Applying Un-Subdivide with {{iterations}} iterations")

decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
decimate.decimate_type = 'UNSUBDIV'
decimate.iterations = iterations
bpy.ops.object.modifier_apply(modifier="Decimate")

current_faces = len(obj.data.polygons)
current_quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
print(f"After un-subdivide: {{current_faces}} faces, {{current_quads}} quads ({{100*current_quads/max(1,current_faces):.1f}}%)")

# Clean up mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.0001)
bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

# Count face types
bm = bmesh.from_edit_mesh(obj.data)
quads = sum(1 for f in bm.faces if len(f.verts) == 4)
tris = sum(1 for f in bm.faces if len(f.verts) == 3)
print(f"Face types: {{quads}} quads, {{tris}} triangles")

# ============================================================
# POST-PROCESS: Clean up irregular topology after un-subdivide
# ============================================================

# Convert triangles back to quads
if tris > 0:
    print(f"Post-process: Converting {{tris}} triangles to quads")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.tris_convert_to_quads(
        face_threshold=0.8,
        shape_threshold=0.8,
        uvs=True,
        vcols=True,
        seam=True,
        sharp=True,
        materials=True
    )
    bm = bmesh.from_edit_mesh(obj.data)
    new_tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    print(f"After post-process: {{new_tris}} triangles remaining")

# ============================================================
# POLE CLEANUP: Fix irregular vertices that interrupt edge loops
# ============================================================
# When neural network classifications are available, use them.
# Otherwise fall back to rule-based V3-V5 pair detection.

bm = bmesh.from_edit_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
bm.faces.ensure_lookup_table()

def get_valence(v):
    return len(v.link_edges)

def is_on_boundary(v):
    return any(e.is_boundary for e in v.link_edges)

def vertex_curvature_estimate(v):
    # Estimate local curvature - high curvature means feature area
    if len(v.link_faces) < 2:
        return 1.0  # Boundary - treat as high importance
    normals = [f.normal for f in v.link_faces]
    if len(normals) < 2:
        return 0.0
    total = 0
    count = 0
    for i, n1 in enumerate(normals):
        for n2 in normals[i+1:]:
            total += n1.dot(n2)
            count += 1
    if count == 0:
        return 0.0
    avg_dot = total / count
    return 1.0 - max(0, avg_dot)

# Build a set of vertices that the neural network says to fix
# Note: vertex indices may shift after un-subdivide, so we use position-based matching
neural_fix_vertices = set()
if use_neural_network and POLE_CLASSIFICATIONS:
    # After un-subdivide, vertex indices have changed, so we need to match by valence+position
    # For now, use the rule-based approach but guided by neural network statistics
    vertices_to_fix = set(POLE_CLASSIFICATIONS.get('vertices_to_fix', []))
    vertices_to_keep = set(POLE_CLASSIFICATIONS.get('vertices_to_keep', []))
    print(f"Neural guidance: fix={{len(vertices_to_fix)}}, keep={{len(vertices_to_keep)}}")

# Multiple passes to fix poles iteratively
total_fixed = 0
max_fixes_per_pass = 50  # Limit to avoid over-reduction

for pass_num in range(3):  # Up to 3 passes
    fixed_this_pass = 0
    
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    
    # Strategy 1: Find V3-V5 pairs connected by an edge and collapse
    # This is the cleanest fix - reduces V3+V5 to V4
    v3_verts = []
    v5_verts = []
    
    for v in bm.verts:
        if not v.is_valid:
            continue
        valence = get_valence(v)
        if is_on_boundary(v):
            continue
        curv = vertex_curvature_estimate(v)
        if curv > 0.1:  # Skip curved areas (stricter threshold)
            continue
        
        if valence == 3:
            v3_verts.append(v)
        elif valence == 5:
            v5_verts.append(v)
    
    # Find connected V3-V5 pairs (limit how many we fix)
    pairs_found = 0
    for v3 in v3_verts:
        if pairs_found >= max_fixes_per_pass:
            break
        if not v3.is_valid:
            continue
        for e in v3.link_edges:
            if not e.is_valid:
                continue
            other = e.other_vert(v3)
            if not other.is_valid:
                continue
            if get_valence(other) == 5 and not is_on_boundary(other):
                # Found a V3-V5 pair! Try to collapse the edge
                # This merges the V3 into the V5, resulting in a V4
                try:
                    bmesh.ops.collapse(bm, edges=[e])
                    fixed_this_pass += 1
                    pairs_found += 1
                except:
                    pass
                break  # Move to next V3
    
    if pairs_found > 0:
        bmesh.update_edit_mesh(obj.data)
        bm = bmesh.from_edit_mesh(obj.data)
        print(f"Pass {{pass_num+1}}: Dissolved {{pairs_found}} V3-V5 pairs")
    
    # Strategy 2: For isolated V3 poles, try edge collapse toward V5 neighbor
    # Only do this if we haven't done many fixes yet
    if fixed_this_pass < max_fixes_per_pass // 2:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        
        v3_fixed = 0
        for v in list(bm.verts):
            if v3_fixed >= 20:  # Very limited
                break
            if not v.is_valid:
                continue
            valence = get_valence(v)
            if valence != 3 or is_on_boundary(v):
                continue
            curv = vertex_curvature_estimate(v)
            if curv > 0.1:
                continue
            
            # Find best edge to collapse - only toward V5
            best_edge = None
            
            for e in v.link_edges:
                if not e.is_valid:
                    continue
                other = e.other_vert(v)
                if not other.is_valid or is_on_boundary(other):
                    continue
                other_val = get_valence(other)
                
                # Only collapse toward V5
                if other_val == 5:
                    best_edge = e
                    break
            
            if best_edge and best_edge.is_valid:
                try:
                    bmesh.ops.collapse(bm, edges=[best_edge])
                    fixed_this_pass += 1
                    v3_fixed += 1
                except:
                    pass
    
    # Skip Strategy 3 (V5+ collapse) - too aggressive
    
    if fixed_this_pass > 0:
        bmesh.update_edit_mesh(obj.data)
        bm = bmesh.from_edit_mesh(obj.data)
    
    total_fixed += fixed_this_pass
    if fixed_this_pass == 0:
        break  # No more fixes possible

if total_fixed > 0:
    print(f"Total poles fixed: {{total_fixed}}")

# ============================================================
# END POLE CLEANUP
# ============================================================

# Re-count after pole cleanup
quads = sum(1 for f in bm.faces if len(f.verts) == 4)
tris = sum(1 for f in bm.faces if len(f.verts) == 3)
print(f"After pole cleanup: {{quads}} quads, {{tris}} triangles")

# Only convert tris to quads if there are PAIRS of triangles
# that can be cleanly joined (avoid creating degenerate quads)
if tris > 0 and tris % 2 == 0 and tris * 2 < quads:
    # Very selective conversion - high thresholds
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.tris_convert_to_quads(
        face_threshold=1.4,  # ~80 degrees - very selective
        shape_threshold=1.4,
        uvs=True,
        vcols=True,
        seam=True,
        sharp=True,
        materials=True
    )
    bm = bmesh.from_edit_mesh(obj.data)
    new_tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    print(f"After selective tri-to-quad: {{tris}} -> {{new_tris}} triangles")

bpy.ops.object.mode_set(mode='OBJECT')

# Final cleanup - remove any degenerate faces that Blender might create
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()

# Find and delete faces with duplicate vertices
to_delete = []
for face in bm.faces:
    verts = [v.index for v in face.verts]
    if len(verts) != len(set(verts)):
        to_delete.append(face)

if to_delete:
    print(f"Removing {{len(to_delete)}} degenerate faces")
    bmesh.ops.delete(bm, geom=to_delete, context='FACES')
    bmesh.update_edit_mesh(obj.data)

bpy.ops.object.mode_set(mode='OBJECT')

final_faces = len(obj.data.polygons)
final_quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
print(f"Final: {{final_faces}} faces, {{final_quads}} quads ({{100*final_quads/max(1,final_faces):.1f}}%)")

# Export - do NOT triangulate, let Blender export quads and tris as-is
bpy.ops.wm.obj_export(
    filepath=r"{output_obj}",
    export_selected_objects=True,
    export_triangulated_mesh=False,
    export_materials=False,
    export_normals=True,
    export_uv=True
)
print(f"Exported to {{r'{output_obj}'}}")
'''
            script_path = tmpdir / "loop_decimate.py"
            script_path.write_text(script)
            
            result = subprocess.run(
                [blender_path, "--background", "--python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if not output_obj.exists():
                raise RuntimeError(f"Blender loop decimation failed: {result.stderr}")
            
            # Load and clean the mesh
            output_mesh = Mesh.from_file(output_obj)
            
            # Filter out any degenerate faces that Blender might have created
            good_faces = []
            degen_count = 0
            for face in output_mesh.faces:
                if len(face) != len(set(face)):  # Has duplicate vertices
                    degen_count += 1
                else:
                    good_faces.append(face)
            
            if degen_count > 0:
                import logging
                logging.getLogger(__name__).info(f"Removed {degen_count} degenerate faces from Blender output")
                output_mesh = Mesh(output_mesh.vertices, good_faces)
            
            # Apply neural network-guided pole cleanup in Python
            output_mesh = self._neural_pole_cleanup(output_mesh)
            
            return output_mesh
    
    def _neural_pole_cleanup(self, mesh: Mesh, max_iterations: int = 5) -> Mesh:
        """
        Apply neural network-guided pole cleanup.
        
        Uses the trained model to identify which poles are defects vs structural,
        then applies targeted fixes to defect poles via Blender.
        """
        try:
            from neurotopo.analysis.neural.pole_classifier import HybridPoleClassifier
            from pathlib import Path
            import tempfile
            import subprocess
            
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "pole_classifier.pt"
            if not model_path.exists():
                logger.debug("Neural pole classifier model not found, skipping neural cleanup")
                return mesh
            
            classifier = HybridPoleClassifier(model_path=str(model_path))
            
            initial_quality = self._assess_quad_quality(mesh)
            logger.info(f"Neural pole cleanup: starting quality {initial_quality:.1%}")
            
            # Get fixable poles
            fixable = classifier.get_fixable_poles(mesh, min_confidence=0.75)
            
            if not fixable:
                logger.debug("No fixable poles found")
                return mesh
            
            logger.info(f"Neural pole cleanup: {len(fixable)} poles identified for fixing")
            
            # Group by fix type
            v3_poles = [p for p in fixable if p.valence == 3]
            v5_poles = [p for p in fixable if p.valence == 5]
            v6plus_poles = [p for p in fixable if p.valence >= 6]
            
            logger.info(f"  V3 defects: {len(v3_poles)}, V5 defects: {len(v5_poles)}, V6+ defects: {len(v6plus_poles)}")
            
            # Find V3-V5 pairs that are connected by an edge
            v3_indices = set(p.vertex_idx for p in v3_poles)
            v5_indices = set(p.vertex_idx for p in v5_poles)
            
            # Build edge adjacency from mesh faces
            edge_pairs = []
            for face in mesh.faces:
                n = len(face)
                for i in range(n):
                    v0, v1 = face[i], face[(i + 1) % n]
                    # Check if this edge connects a V3 defect to a V5 defect
                    if (v0 in v3_indices and v1 in v5_indices) or (v0 in v5_indices and v1 in v3_indices):
                        edge_pairs.append((min(v0, v1), max(v0, v1)))
            
            edge_pairs = list(set(edge_pairs))  # Deduplicate
            
            if not edge_pairs:
                logger.info("No V3-V5 edge pairs found for targeted cleanup")
                return mesh
            
            logger.info(f"Found {len(edge_pairs)} V3-V5 edge pairs to collapse")
            
            # Convert numpy types to Python ints for Blender script
            edge_pairs_clean = [(int(v0), int(v1)) for v0, v1 in edge_pairs[:100]]
            
            # Use Blender to collapse the identified edges - iterate until no improvement
            prev_quality = initial_quality
            for iteration in range(max_iterations):
                mesh = self._blender_collapse_edges(mesh, edge_pairs_clean)
                
                # Re-analyze
                fixable = classifier.get_fixable_poles(mesh, min_confidence=0.75)
                v3_indices = set(p.vertex_idx for p in fixable if p.valence == 3)
                v5_indices = set(p.vertex_idx for p in fixable if p.valence == 5)
                
                # Find new edge pairs
                edge_pairs = []
                for face in mesh.faces:
                    n = len(face)
                    for i in range(n):
                        v0, v1 = face[i], face[(i + 1) % n]
                        if (v0 in v3_indices and v1 in v5_indices) or (v0 in v5_indices and v1 in v3_indices):
                            edge_pairs.append((min(v0, v1), max(v0, v1)))
                
                edge_pairs = list(set(edge_pairs))
                edge_pairs_clean = [(int(v0), int(v1)) for v0, v1 in edge_pairs[:100]]
                
                curr_quality = self._assess_quad_quality(mesh)
                logger.info(f"  Iteration {iteration + 1}: quality {curr_quality:.1%}, {len(fixable)} poles to fix")
                
                if curr_quality <= prev_quality or len(edge_pairs_clean) == 0:
                    break
                prev_quality = curr_quality
            
            final_quality = self._assess_quad_quality(mesh)
            logger.info(f"Neural pole cleanup: final quality {final_quality:.1%}")
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Neural pole cleanup failed: {e}")
            import traceback
            traceback.print_exc()
            return mesh
    
    def _blender_collapse_edges(self, mesh: Mesh, edge_pairs: list) -> Mesh:
        """Use Blender to collapse specific edges identified by the neural network."""
        import tempfile
        import subprocess
        from pathlib import Path
        
        blender_path = "/Applications/Blender.app/Contents/MacOS/blender"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_obj = tmpdir / "input.obj"
            output_obj = tmpdir / "output.obj"
            
            mesh.to_file(input_obj)
            
            # Convert edge pairs to a Python list string for the script
            edges_str = str(edge_pairs)
            
            script = f'''
import bpy
import bmesh

# Clear and import
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{input_obj}")

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Edge pairs to collapse (V3-V5 pairs identified by neural network)
edge_pairs = {edges_str}
print(f"Collapsing {{len(edge_pairs)}} V3-V5 edge pairs")

bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()

# Build a lookup of edges by vertex pairs
edge_map = {{}}
for e in bm.edges:
    key = (min(e.verts[0].index, e.verts[1].index), max(e.verts[0].index, e.verts[1].index))
    edge_map[key] = e

collapsed = 0
for v0, v1 in edge_pairs:
    key = (v0, v1)
    if key in edge_map:
        e = edge_map[key]
        if e.is_valid and not any(v.is_boundary for v in e.verts):
            try:
                bmesh.ops.collapse(bm, edges=[e])
                collapsed += 1
            except:
                pass

print(f"Collapsed {{collapsed}} edges")
bmesh.update_edit_mesh(obj.data)

# Clean up
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.0001)
bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

# Convert any remaining triangles to quads
bm = bmesh.from_edit_mesh(obj.data)
tris = sum(1 for f in bm.faces if len(f.verts) == 3)
if tris > 0:
    bpy.ops.mesh.tris_convert_to_quads(
        face_threshold=0.8,
        shape_threshold=0.8,
    )

bpy.ops.object.mode_set(mode='OBJECT')

# Export
bpy.ops.wm.obj_export(
    filepath=r"{output_obj}",
    export_selected_objects=True,
    export_triangulated_mesh=False,
    export_materials=False,
    export_normals=True,
    export_uv=True
)

final_quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
final_faces = len(obj.data.polygons)
print(f"Final: {{final_faces}} faces, {{final_quads}} quads ({{100*final_quads/max(1,final_faces):.1f}}%)")
'''
            script_path = tmpdir / "collapse_edges.py"
            script_path.write_text(script)
            
            result = subprocess.run(
                [blender_path, "--background", "--python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if not output_obj.exists():
                logger.warning(f"Blender edge collapse failed: {result.stderr}")
                return mesh
            
            output_mesh = Mesh.from_file(output_obj)
            return output_mesh
    
    def _blender_curvature_decimate(
        self,
        mesh: Mesh,
        target_faces: int,
        start_time: float,
        input_quality: float
    ) -> RemeshResult:
        """Use Blender to decimate with curvature-awareness.
        
        Uses Blender's dissolve operations which work better for 
        preserving quad topology than collapse decimation.
        """
        try:
            output = self._blender_quad_decimate(mesh, target_faces)
            elapsed = time.time() - start_time
            
            output_quality = self._assess_quad_quality(output)
            logger.info(f"Blender decimate: {mesh.num_faces} -> {output.num_faces} faces "
                       f"({output_quality:.1%} regular) in {elapsed:.2f}s")
            
            if output_quality < input_quality * 0.8:
                logger.warning(f"Significant quality degradation: {input_quality:.1%} -> {output_quality:.1%}")
            
            return RemeshResult(
                mesh=output,
                success=True,
                actual_face_count=output.num_faces,
                time_seconds=elapsed,
                metadata={
                    "backend": "hybrid_blender",
                    "method": "blender_dissolve",
                    "input_quality": input_quality,
                    "output_quality": output_quality,
                }
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"Blender decimation failed: {e}")
            
            # Last resort: return original mesh
            return RemeshResult(
                mesh=mesh,
                success=True,
                actual_face_count=mesh.num_faces,
                time_seconds=elapsed,
                metadata={
                    "backend": "hybrid_passthrough",
                    "method": "fallback",
                    "input_quality": input_quality,
                    "error": str(e),
                }
            )

    def _blender_quad_decimate(self, mesh: Mesh, target_faces: int) -> Mesh:
        """Use Blender to reduce quad mesh while preserving topology.
        
        Strategy: Use edge loop/ring dissolution which is the quad-native way
        to reduce mesh complexity while maintaining regular topology.
        """
        import tempfile
        import subprocess
        from pathlib import Path
        
        # Blender path
        blender_path = "/Applications/Blender.app/Contents/MacOS/blender"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_obj = tmpdir / "input.obj"
            output_obj = tmpdir / "output.obj"
            
            # Export input mesh
            mesh.to_file(input_obj)
            
            # Calculate how much to reduce
            # We'll use iterative edge loop dissolution
            reduction_factor = mesh.num_faces / target_faces
            
            # Blender script for quad-preserving decimation via edge loops
            script = f'''
import bpy
import bmesh
import random

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import OBJ
bpy.ops.wm.obj_import(filepath=r"{input_obj}")

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Work in edit mode with bmesh for more control
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)

target_faces = {target_faces}
current_faces = len(bm.faces)
reduction_factor = {reduction_factor}

# For high-quality quad meshes, use very conservative approach:
# ONLY use dissolve modes which can preserve quad structure

# Strategy: Use DISSOLVE mode decimate which merges coplanar faces
# This is the only decimation that can somewhat preserve quads
bpy.ops.object.mode_set(mode='OBJECT')

# Calculate angle threshold based on desired reduction
# More reduction = larger angle threshold
if reduction_factor > 4:
    angle = 0.15  # ~8.5 degrees
elif reduction_factor > 2:
    angle = 0.08  # ~4.5 degrees  
elif reduction_factor > 1.5:
    angle = 0.04  # ~2.3 degrees
else:
    angle = 0.02  # ~1.1 degrees

decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
decimate.decimate_type = 'DISSOLVE'
decimate.angle_limit = angle
decimate.delimit = {{'UV', 'SEAM'}}  # Preserve UV seams

# Apply
bpy.ops.object.modifier_apply(modifier="Decimate")

bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)

# Clean up: convert any triangles back to quads
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.tris_convert_to_quads(
    face_threshold=0.7, 
    shape_threshold=0.7,
    uvs=True,
    vcols=False,
    seam=True,
    sharp=True,
    materials=True
)

# Remove doubles
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.0001)

# Handle N-gons: select them and triangulate, then convert back to quads
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')

# Select N-gons (faces with more than 4 vertices)
for p in obj.data.polygons:
    if len(p.vertices) > 4:
        p.select = True

bpy.ops.object.mode_set(mode='EDIT')

# If any N-gons selected, triangulate them then convert to quads
if bpy.context.object.data.total_face_sel > 0:
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.tris_convert_to_quads(face_threshold=0.6, shape_threshold=0.6)

bpy.ops.object.mode_set(mode='OBJECT')

# Export result (keep quads)
bpy.ops.wm.obj_export(
    filepath=r"{output_obj}",
    export_selected_objects=True,
    export_triangulated_mesh=False,
    export_materials=False
)
'''
            script_path = tmpdir / "decimate.py"
            script_path.write_text(script)
            
            # Run Blender
            result = subprocess.run(
                [blender_path, "--background", "--python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if not output_obj.exists():
                raise RuntimeError(f"Blender decimation failed: {result.stderr}")
            
            # Load output with quad-aware loader
            return Mesh.from_file(output_obj)

    def _fast_decimate(
        self,
        mesh: Mesh,
        target_faces: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fast mesh decimation using best available method."""
        # Try fast_simplification first (much faster)
        try:
            import fast_simplification
            
            target_ratio = min(0.99, max(0.01, target_faces / mesh.num_faces))
            verts, faces = fast_simplification.simplify(
                mesh.vertices.astype(np.float32),
                mesh.faces.astype(np.int32),
                target_reduction=1 - target_ratio
            )
            return verts, faces
        except ImportError:
            pass
        
        # Fallback to trimesh
        try:
            import trimesh
            tm = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                process=False
            )
            
            # Try quadric decimation
            try:
                simplified = tm.simplify_quadric_decimation(target_faces)
                return np.array(simplified.vertices), np.array(simplified.faces)
            except Exception:
                pass
            
            # Fallback: vertex clustering
            from scipy.spatial import cKDTree
            
            cluster_size = np.sqrt(tm.area / target_faces) * 1.5
            tree = cKDTree(mesh.vertices)
            clusters = tree.query_ball_tree(tree, cluster_size)
            
            used = set()
            new_verts = []
            vert_map = {}
            
            for i, cluster in enumerate(clusters):
                if i in used:
                    continue
                cluster_verts = mesh.vertices[cluster]
                centroid = cluster_verts.mean(axis=0)
                new_idx = len(new_verts)
                new_verts.append(centroid)
                for j in cluster:
                    vert_map[j] = new_idx
                    used.add(j)
            
            new_faces = []
            for face in mesh.faces:
                new_face = [vert_map.get(v, 0) for v in face]
                if len(set(new_face)) == 3:
                    new_faces.append(new_face)
            
            return np.array(new_verts), np.array(new_faces)
            
        except Exception as e:
            logger.warning(f"Decimation failed: {e}, returning original")
            return mesh.vertices.copy(), mesh.faces.copy()
    
    def _improve_triangles(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        iterations: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Improve triangle quality with Laplacian smoothing and edge flipping.
        
        Makes triangles more equilateral, which leads to better quads when paired.
        Also performs edge flips to improve Delaunay property (better for quad pairing).
        """
        verts = vertices.copy()
        faces = np.array(faces)
        
        # Build adjacency
        from collections import defaultdict
        
        neighbors = defaultdict(set)
        edge_count = defaultdict(int)
        
        for face in faces:
            for i in range(3):
                v0, v1 = int(face[i]), int(face[(i+1) % 3])
                neighbors[v0].add(v1)
                neighbors[v1].add(v0)
                e = tuple(sorted([v0, v1]))
                edge_count[e] += 1
        
        # Identify boundary vertices
        boundary_verts = set()
        for e, count in edge_count.items():
            if count == 1:
                boundary_verts.add(e[0])
                boundary_verts.add(e[1])
        
        # Laplacian smoothing iterations
        for _ in range(iterations):
            new_verts = verts.copy()
            for vi in range(len(verts)):
                if vi in boundary_verts:
                    continue
                if len(neighbors[vi]) == 0:
                    continue
                    
                # Uniform Laplacian: average of neighbors
                neighbor_pos = verts[list(neighbors[vi])]
                centroid = neighbor_pos.mean(axis=0)
                # Blend: move 30% toward centroid
                new_verts[vi] = verts[vi] * 0.7 + centroid * 0.3
            
            verts = new_verts
        
        # Edge flipping pass to improve Delaunay property
        # This creates more equilateral triangles which pair better into quads
        faces = self._delaunay_flip(verts, faces, iterations=3)
        
        return verts, faces
    
    def _delaunay_flip(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        iterations: int = 2
    ) -> np.ndarray:
        """Flip edges to improve Delaunay property (more equilateral triangles)."""
        faces = [list(f) for f in faces]
        
        for _ in range(iterations):
            # Build edge to face map
            edge_faces = {}
            for fi, face in enumerate(faces):
                for i in range(3):
                    v0, v1 = face[i], face[(i + 1) % 3]
                    edge = (min(v0, v1), max(v0, v1))
                    if edge not in edge_faces:
                        edge_faces[edge] = []
                    edge_faces[edge].append(fi)
            
            flips = 0
            flipped_edges = set()
            
            for edge, face_list in edge_faces.items():
                if len(face_list) != 2:
                    continue
                if edge in flipped_edges:
                    continue
                
                f0, f1 = face_list
                face0, face1 = faces[f0], faces[f1]
                
                # Get vertices
                v0, v1 = edge
                other0 = [v for v in face0 if v not in edge][0]
                other1 = [v for v in face1 if v not in edge][0]
                
                # Check if flip would improve local Delaunay property
                # by checking if opposite angles sum to > 180 degrees
                if self._should_flip_edge(vertices, v0, v1, other0, other1):
                    # Perform flip
                    faces[f0] = [other0, v0, other1]
                    faces[f1] = [other0, other1, v1]
                    flipped_edges.add(edge)
                    flips += 1
            
            if flips == 0:
                break
        
        return np.array(faces, dtype=np.int32)
    
    def _should_flip_edge(
        self,
        verts: np.ndarray,
        v0: int, v1: int,
        other0: int, other1: int
    ) -> bool:
        """Check if edge (v0,v1) should be flipped based on Delaunay criterion."""
        p0, p1 = verts[v0], verts[v1]
        q0, q1 = verts[other0], verts[other1]
        
        # Compute angles at other0 and other1 opposite to edge (v0,v1)
        def angle_at_vertex(p, a, b):
            """Angle at vertex p in triangle pab."""
            va = a - p
            vb = b - p
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-10 or nb < 1e-10:
                return np.pi / 2
            cos_ang = np.clip(np.dot(va, vb) / (na * nb), -1, 1)
            return np.arccos(cos_ang)
        
        # Angle at other0 opposite to edge v0-v1
        angle0 = angle_at_vertex(q0, p0, p1)
        # Angle at other1 opposite to edge v0-v1
        angle1 = angle_at_vertex(q1, p0, p1)
        
        # Delaunay criterion: flip if sum > pi
        if angle0 + angle1 > np.pi + 0.01:  # Small tolerance
            # Also check that flip won't create inverted triangles
            # New triangles would be: (other0, v0, other1) and (other0, other1, v1)
            def tri_area_2d(a, b, c):
                """Signed area of triangle abc."""
                return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
            
            # Project to 2D for orientation check (use first two axes)
            a0 = tri_area_2d(q0[:2], p0[:2], q1[:2])
            a1 = tri_area_2d(q0[:2], q1[:2], p1[:2])
            
            # Both should have same sign (same orientation)
            if a0 * a1 > 0:
                return True
        
        return False
