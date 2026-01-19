# AI Integration Roadmap

This document outlines opportunities for leveraging AI vision models (OpenAI GPT-4o, Anthropic Claude) to improve retopology quality.

## Current Integration

### Semantic Segmentation (Implemented ✅)
- **Location**: `src/meshretopo/analysis/semantic.py`
- **Purpose**: Identifies anatomical/semantic regions (eyes, mouth, nose, joints)
- **Usage**: Informs topology rules (pole placement, edge loop density)
- **Caching**: Renders and API responses cached based on source file mtime

## Planned Integrations

### 1. AI Quality Assessment (In Progress 🔄)
**Location**: `src/meshretopo/evaluation/ai_quality.py`

Have AI visually evaluate the retopologized mesh and provide:
- Specific topology critiques ("edge loops around eyes too sparse")
- Problem area identification with 2D bounding boxes
- Severity ratings for each issue
- Actionable recommendations

**Benefits**:
- Catches visual issues that metrics miss
- Provides human-readable feedback
- Can identify animation-specific problems

### 2. Edge Flow Direction Guidance (Planned 📋)
**Location**: `src/meshretopo/guidance/ai_flow.py`

AI analyzes the mesh and suggests optimal edge flow directions:
- "Edges should follow the nasolabial fold"
- "Horizontal loops needed around mouth for speech"
- "Radial flow around eye sockets"

**Implementation**:
- Render mesh with UV/tangent visualization
- AI identifies natural flow lines
- Convert to vector field for remeshing guidance

### 3. Mesh Type Auto-Classification (Planned 📋)
**Location**: `src/meshretopo/analysis/mesh_classifier.py`

Automatically classify mesh type to select optimal settings:
- **Character/Organic**: Quad-dominant, animation-ready, deformation topology
- **Hard Surface/Mechanical**: Sharp edge preservation, uniform quads
- **Architectural**: Grid preference, planar alignment
- **Organic Props**: Balanced approach

**Implementation**:
- Single render from canonical view
- Classification with confidence scores
- Auto-select pipeline presets

### 4. Iterative Refinement Loop (Planned 📋)
**Location**: `src/meshretopo/pipeline/ai_refinement.py`

Closed-loop optimization using AI feedback:

```
Initial Retopo → Render → AI Critique → Identify Problem Areas
                                              ↓
                              Targeted Local Refinement
                                              ↓
                                    Re-render → Verify
                                              ↓
                                   (Repeat if needed)
```

**Benefits**:
- Automated quality improvement
- Focuses compute on problem areas
- Self-correcting pipeline

## API Cost Considerations

| Feature | Renders/Mesh | API Calls | Est. Cost |
|---------|-------------|-----------|-----------|
| Semantic Segmentation | 6 | 6 | ~$0.15 |
| Quality Assessment | 6 | 1 | ~$0.03 |
| Edge Flow Guidance | 2 | 1 | ~$0.02 |
| Mesh Classification | 1 | 1 | ~$0.01 |
| Iterative (per loop) | 6 | 2 | ~$0.05 |

**Caching Strategy**: All renders and responses cached by source file mtime + settings hash to minimize redundant API calls.

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    RetopoPipeline                            │
├─────────────────────────────────────────────────────────────┤
│  1. Load Mesh                                                │
│         ↓                                                    │
│  2. [AI] Mesh Classification → Select Presets               │
│         ↓                                                    │
│  3. [AI] Semantic Segmentation → Region Detection           │
│         ↓                                                    │
│  4. [AI] Edge Flow Guidance → Flow Field                    │
│         ↓                                                    │
│  5. Analysis (curvature, features)                          │
│         ↓                                                    │
│  6. Remeshing (with semantic + flow guidance)               │
│         ↓                                                    │
│  7. [AI] Quality Assessment → Issues & Scores               │
│         ↓                                                    │
│  8. [AI] Iterative Refinement (if issues found)             │
│         ↓                                                    │
│  9. Output                                                   │
└─────────────────────────────────────────────────────────────┘
```
