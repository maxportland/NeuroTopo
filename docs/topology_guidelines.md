# 3D Mesh Topology Guidelines

This document summarizes best practices for ideal 3D model topology, with special focus on human character modeling and face topology. These guidelines inform the quality metrics and evaluation criteria used in the MeshRetopo system.

## Table of Contents
1. [Core Topology Principles](#core-topology-principles)
2. [Polygon Types: Quads vs Triangles vs N-gons](#polygon-types-quads-vs-triangles-vs-n-gons)
3. [Edge Loops and Edge Flow](#edge-loops-and-edge-flow)
4. [Poles and Vertices](#poles-and-vertices)
5. [Human Face Topology](#human-face-topology)
6. [Body and Limb Topology](#body-and-limb-topology)
7. [Deformation Considerations](#deformation-considerations)
8. [Manifold Requirements](#manifold-requirements)
9. [Use Case Specific Guidelines](#use-case-specific-guidelines)
10. [References](#references)

---

## Core Topology Principles

**Topology** is the layout of a model—how vertices and edges are placed to create the mesh surface. Good topology is essential for:

- **Fast rendering** (especially for real-time applications)
- **Good deformation** (for animation)
- **Clean subdivision** (for high-poly detail work)
- **Efficient UV mapping** (for texturing)

### Key Principles

1. **Polygon Density Distribution**
   - Add more polygons where there's curvature
   - Use fewer polygons in flat areas
   - Balance density based on deformation and silhouette needs

2. **Silhouette Definition**
   - Polygons define the shape—don't waste them where they don't contribute
   - Focus geometry on areas that affect the model's outline

3. **Clean Mesh Flow**
   - Edges should flow naturally along the form
   - Avoid unnecessary edge terminations mid-surface
   - Support loops should follow major form changes

---

## Polygon Types: Quads vs Triangles vs N-gons

### Quadrilaterals (Quads) ⭐ Preferred

**Advantages:**
- Best for subdivision surface modeling
- Predictable deformation behavior
- Clean edge loops for animation
- Easier to modify and adjust
- Required for proper ZBrush/Mudbox sculpting

**When to use:**
- Character modeling
- Organic shapes
- Anything requiring subdivision
- Animation-ready models

### Triangles

**Acceptable uses:**
- Low-poly game models
- Real-time rendering (GPUs process triangles natively)
- Final export for game engines

**Disadvantages:**
- Unpredictable subdivision results
- Can cause pinching artifacts
- Harder to manage edge flow

### N-gons (5+ sided polygons)

**Generally avoid because:**
- Unpredictable subdivision behavior
- Can create artifacts in rendering
- Difficult to control deformation

**Exceptions:**
- Flat, non-deforming surfaces
- Areas hidden from view

---

## Edge Loops and Edge Flow

### What is an Edge Loop?

An edge loop is a continuous path of edges that forms a ring around the mesh. Good edge loops:

- Follow the natural contours of the form
- Support areas that will deform
- Enable easy selection and modification

### Edge Loop Placement Guidelines

1. **Around Deformation Areas**
   - Joints (elbows, knees, wrists, ankles)
   - Facial features (eyes, mouth, nose)
   - Any area that bends or stretches

2. **Following Muscle Flow**
   - Edges should align with major muscle groups
   - This ensures natural deformation during animation

3. **Support Loops for Sharpness**
   - Use supporting loops close to hard edges
   - Prevents unwanted smoothing during subdivision

### The "Flow" Concept

Edge flow describes how edges connect across the mesh surface. Good flow:
- Creates predictable results when subdividing
- Allows easy loop selection
- Produces clean deformation

---

## Poles and Vertices

### What is a Pole?

A **pole** is a vertex where more or fewer than 4 edges meet (in a quad-based mesh).

- **3-pole (E-pole)**: 3 edges meeting at a vertex
- **5-pole (N-pole)**: 5 edges meeting at a vertex
- **6+ poles**: Generally problematic

### Pole Placement Rules

1. **Avoid poles in deformation areas**
   - Keep them away from joints
   - Don't place at bend points

2. **Place poles in flat areas**
   - They can cause dimples/bumps in subdivided surfaces
   - Flat areas hide these artifacts

3. **Use poles to redirect edge flow**
   - Necessary for topology transitions
   - Strategic placement enables better flow

4. **Minimize pole count**
   - Fewer poles = cleaner subdivision
   - Each pole is a potential artifact source

### Vertex Valence

The **valence** of a vertex is the number of edges connected to it. In quad topology:
- **Valence 4** = Regular (ideal)
- **Valence 3** = 3-pole
- **Valence 5** = 5-pole
- **Higher** = Increasingly problematic

---

## Human Face Topology

Face topology is critical for animation because facial expressions require precise deformation. The key areas are:

### The Three Critical Loops

1. **Eye Loop (Orbicularis Oculi)**
   - Complete loop around each eye socket
   - Follows the orbicularis oculi muscle
   - Enables realistic blinking and squinting
   - Should be concentric (multiple nested loops)

2. **Mouth Loop (Orbicularis Oris)**  
   - Complete loop around the lips
   - Follows the orbicularis oris muscle
   - Enables realistic mouth movements
   - Multiple concentric loops for fine control

3. **Nose Loop**
   - Loop defining the nostril and nose bridge
   - Connects to the mouth and cheek regions
   - Important for expressions like snarling, flaring nostrils

### Face Topology Best Practices

Based on industry-standard topology (Pixar, BioWare, Epic Games):

```
Key Features:
├── Concentric loops around eyes (3-5 rings)
├── Concentric loops around mouth (3-5 rings)
├── Nasolabial fold definition
├── Brow ridge support
├── Cheek bone definition
└── Jawline edge loop
```

### Landmark Reference

| Feature | Topology Requirement |
|---------|---------------------|
| Eye corners | 5-pole placement (controls flow direction) |
| Mouth corners | 5-pole placement (allows expressions) |
| Nose bridge | Edge loop transition area |
| Cheeks | Diagonal flow connecting eye and mouth |
| Forehead | Horizontal loops for brow movement |
| Chin | Support loop for jaw opening |

### Common Mistakes to Avoid

- ❌ Triangles around eyes or mouth
- ❌ Poles on eyelids or lips
- ❌ Broken loops around features
- ❌ Uneven polygon density across face
- ❌ Missing support loops for expressions

---

## Body and Limb Topology

### Joint Topology (Knees, Elbows, etc.)

Joints require special attention because they undergo extreme deformation:

1. **Loop Count**
   - Minimum 3 edge loops across the joint
   - More loops = smoother bending
   - Place the central loop at the bend axis

2. **Back of Joint (Compression Side)**
   - Higher density for accordion-like folding
   - Edges should allow compression without intersection

3. **Front of Joint (Extension Side)**
   - Moderate density
   - Edges stretch smoothly during extension

### Shoulder Topology

The shoulder is one of the most complex areas due to its range of motion:

- Ball joint requires topology supporting 360° rotation
- Edge loops should follow the deltoid muscle shape
- Armpit area needs careful attention to prevent pinching

### Hand and Finger Topology

- Each finger joint follows the same rules as larger joints
- Edge loops should encircle each finger segment
- Web areas between fingers need careful density management

---

## Deformation Considerations

### Areas Requiring Most Attention

1. **High Deformation Areas**
   - Crotch/hips/butt
   - Shoulders/armpits
   - Mouth corners/cheeks
   - Knees and elbows
   - Hands and fingers

2. **Topology Requirements for Deformation**
   - Vertices and edges in specific positions
   - Proper edge flow following muscle structure
   - Adequate polygon density for smooth bending

### Testing Deformation

Before finalizing topology:
1. Test with extreme poses
2. Check for interpenetration
3. Look for pinching or tearing
4. Verify smooth transitions across joints

---

## Manifold Requirements

A **manifold mesh** is one that could theoretically exist in the physical world. Requirements:

### What Makes a Mesh Manifold

✅ **Valid:**
- Every edge shared by exactly 2 faces
- Closed surfaces (no holes except intentional)
- Consistent face orientation (normals pointing outward)
- No self-intersection

### Non-Manifold Elements to Avoid

❌ **Invalid:**
- T-vertices (vertex on an edge not at its endpoint)
- Doubled faces (overlapping coincident faces)
- Gaps in the surface
- Flipped faces (inconsistent normals)
- Internal/hidden faces
- Floating vertices (not connected to any face)
- Edges shared by more than 2 faces (non-manifold edges)

### Why Manifold Matters

- Required for 3D printing
- Needed for proper Boolean operations
- Essential for physics simulations
- Important for clean subdivision

---

## Use Case Specific Guidelines

### For Subdivision Modeling

- **All quads** whenever possible
- Proper edge loops essential
- Poles in strategic locations only
- Clean topology for predictable subdivision

### For Sculpting Base Mesh (ZBrush/Mudbox)

- Primarily quads to avoid pinching
- Even polygon distribution
- No need for animation-ready loops
- Can have higher topology density

### For Real-Time/Game Models

- Triangles acceptable in final export
- Focus on silhouette definition
- Minimize vertex count where possible
- Consider vertex split costs (UV seams, hard edges)

### For Animation

- Animation-ready edge loops mandatory
- Focus on deformation areas
- Test with intended rig before approval
- Higher density at joints and face

---

## Quality Metrics

Based on the research, ideal topology should score well on:

| Metric | Ideal Value | Importance |
|--------|-------------|------------|
| Quad percentage | 100% | Critical for subdivision |
| Regular vertices (valence 4) | >80% | Smooth surfaces |
| Manifold status | True | Required for many uses |
| Pole placement | Away from deform areas | Animation quality |
| Edge loop continuity | Unbroken | Deformation |
| Polygon density variation | Smooth gradients | Visual quality |

---

## References

### Primary Sources

1. **Polycount Wiki - Topology**
   - http://wiki.polycount.com/wiki/Topology
   - Comprehensive overview of topology principles

2. **Polycount Wiki - FaceTopology**
   - http://wiki.polycount.com/wiki/FaceTopology
   - Face-specific topology examples from industry

3. **Polycount Wiki - Limb Topology**
   - http://wiki.polycount.com/wiki/Limb_Topology
   - Joint and limb deformation examples

4. **Polycount Forum - Face Topology Breakdown**
   - Community thread with professional breakdowns
   - Examples from Gears of War, Mass Effect, Digic Pictures

### Industry Examples

- **Pixar** - Hippydrome articulation guides by Brian Tindall
- **BioWare** - Mass Effect character topology
- **Epic Games** - Gears of War topology breakdowns
- **Digic Pictures** - Cinematic character topology

### Additional Reading

- Wikipedia: Polygon Mesh
  - https://en.wikipedia.org/wiki/Polygon_mesh
  - Technical overview of mesh representations

- Polycount Forum "How Do I Model This?"
  - Ongoing community resource for specific topology challenges

---

## Applying to MeshRetopo

These guidelines inform our quality evaluation metrics:

1. **Quad Percentage** - Higher is better, especially for animation
2. **Vertex Regularity** - Prefer valence-4 vertices
3. **Manifold Status** - Must be manifold for production use
4. **Edge Flow Quality** - Loops should be continuous and purposeful
5. **Pole Placement** - Evaluate based on proximity to deformation areas

### Implemented Enhancements

Based on this research, the following enhancements have been implemented:

#### Topology-Informed Metrics (`metrics.py`)
- **Pole Analysis** - Detects and penalizes high-valence poles (6+)
- **Curvature-Aware Pole Placement** - Uses angle deficit method to identify high-curvature areas and penalize poles placed there
- **Edge Loop Metrics** - Evaluates edge loop continuity potential
- **Edge Flow Score** - Scores based on interior edge ratio

#### AI-Powered Semantic Analysis (`analysis/semantic.py`)
- **Mesh Rendering** - Renders mesh from multiple viewpoints
- **AI Vision Analysis** - Uses GPT-4o or Claude to identify semantic regions
- **Region Types Detected**:
  - Facial features: eye, mouth, nose, ear
  - Body parts: head, neck, torso, arm, hand, leg, foot
  - Joints: shoulder, elbow, wrist, hip, knee, ankle
  - Generic: feature, flat, unknown

#### Topology Rules by Region (`REGION_TOPOLOGY_RULES`)
| Region | Concentric Loops | Allow Poles | Density | Deformation Priority |
|--------|-----------------|-------------|---------|---------------------|
| Eye | Yes (5 ideal) | No | 1.5x | 0.9 |
| Mouth | Yes (5 ideal) | No | 1.5x | 0.95 |
| Elbow | Yes (4 ideal) | No | 1.3x | 0.9 |
| Shoulder | Yes (5 ideal) | No | 1.5x | 1.0 |
| Torso | No | Yes (ideal) | 0.8x | 0.4 |

#### Semantic Guidance Generation (`guidance/semantic.py`)
- Converts semantic regions to per-vertex guidance fields
- **Density Field** - Higher density at important regions
- **Pole Penalty Field** - Prevents poles at deformation zones
- **Loop Requirement Field** - Indicates where concentric loops are needed

### Usage

```python
from meshretopo import RetopoPipeline

# Enable AI semantic analysis
pipeline = RetopoPipeline(
    backend='hybrid',
    target_faces=5000,
    semantic_analysis=True,  # Enable AI vision
    semantic_api_provider='openai',  # or 'anthropic'
)

output_mesh, score = pipeline.process('input.obj')
```

---

## Reference Images

The following reference images are available in the `OutcomeTargets` folder, showing examples of ideal face topology:

- `good_result1.jpg` - Wireframe showing ideal face topology
- `good_result2.jpg` - Wireframe showing ideal face topology  
- `good_result3.jpg` - Wireframe showing ideal face topology
- `good_result4.jpg` - Wireframe showing ideal face topology

These images demonstrate:
- Concentric edge loops around eyes and mouth
- Clean quad-based topology
- Proper pole placement at eye and mouth corners
- Smooth edge flow following facial contours
- Appropriate polygon density (higher around features, lower on flat areas)

Use these as visual references when evaluating retopology output quality.

---

*Document compiled from industry resources and academic references for the MeshRetopo project.*
*Last updated: January 2026*
