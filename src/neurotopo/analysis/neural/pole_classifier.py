"""
PyTorch-based pole classification for topology cleanup.

This module uses a Graph Neural Network (GNN) to classify irregular vertices:
- Is this pole a defect (should be fixed)?
- Or is it structural (should be preserved)?

The key insight is that defect poles exist in flat areas and break edge loops,
while structural poles exist at feature boundaries and direct edge flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from neurotopo.core.mesh import Mesh

logger = logging.getLogger(__name__)


@dataclass
class PoleInfo:
    """Information about an irregular vertex (pole)."""
    vertex_idx: int
    valence: int
    curvature: float
    is_boundary: bool
    neighbor_valences: List[int]
    face_angles: List[float]  # Angles at this vertex in each face
    prediction: Optional[str] = None  # 'fix', 'keep', or None
    confidence: float = 0.0


class PoleFeatureExtractor:
    """Extract features from mesh vertices for pole classification."""
    
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self._build_adjacency()
        self._compute_vertex_data()
    
    def _build_adjacency(self):
        """Build vertex adjacency structures (edge-connected neighbors only)."""
        n = self.mesh.num_vertices
        self.vertex_neighbors = [set() for _ in range(n)]
        self.vertex_faces = [[] for _ in range(n)]
        
        for fi, face in enumerate(self.mesh.faces):
            n_verts = len(face)
            for i in range(n_verts):
                vi = face[i]
                self.vertex_faces[vi].append(fi)
                
                # Only add edge-connected neighbors (adjacent in face)
                prev_vi = face[(i - 1) % n_verts]
                next_vi = face[(i + 1) % n_verts]
                self.vertex_neighbors[vi].add(prev_vi)
                self.vertex_neighbors[vi].add(next_vi)
        
        # Convert to lists for consistent ordering
        self.vertex_neighbors = [list(s) for s in self.vertex_neighbors]
    
    def _compute_vertex_data(self):
        """Precompute per-vertex data."""
        n = self.mesh.num_vertices
        
        # Valence (edge count)
        self.valences = np.array([len(self.vertex_neighbors[i]) for i in range(n)])
        
        # Boundary detection
        edge_count = {}
        for face in self.mesh.faces:
            for i in range(len(face)):
                v0, v1 = face[i], face[(i+1) % len(face)]
                edge = (min(v0, v1), max(v0, v1))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary_verts = set()
        for edge, count in edge_count.items():
            if count == 1:  # Boundary edge
                boundary_verts.add(edge[0])
                boundary_verts.add(edge[1])
        self.is_boundary = np.array([i in boundary_verts for i in range(n)])
        
        # Curvature estimation (simplified mean curvature)
        self.curvatures = self._estimate_curvatures()
    
    def _estimate_curvatures(self) -> np.ndarray:
        """Estimate mean curvature at each vertex."""
        n = self.mesh.num_vertices
        curvatures = np.zeros(n)
        
        # Ensure normals exist
        if self.mesh.normals is None:
            self.mesh.compute_normals()
        
        for vi in range(n):
            neighbors = self.vertex_neighbors[vi]
            if len(neighbors) < 2:
                continue
            
            # Get normals of this vertex and neighbors
            n_self = self.mesh.normals[vi]
            n_neighbors = self.mesh.normals[neighbors]
            
            # Curvature estimated from normal variation
            dot_products = np.dot(n_neighbors, n_self)
            curvatures[vi] = 1.0 - np.mean(np.clip(dot_products, -1, 1))
        
        return curvatures
    
    def get_pole_features(self, vertex_idx: int) -> np.ndarray:
        """
        Extract feature vector for a single vertex.
        
        Features:
        - Valence (normalized)
        - Is boundary (0/1)
        - Local curvature
        - Neighbor valence statistics (mean, std, min, max)
        - Face angle statistics
        - Curvature gradient (difference from neighbors)
        """
        vi = vertex_idx
        neighbors = self.vertex_neighbors[vi]
        
        # Basic features
        valence = self.valences[vi]
        is_boundary = float(self.is_boundary[vi])
        curvature = self.curvatures[vi]
        
        # Neighbor valence statistics
        neighbor_valences = self.valences[neighbors] if neighbors else np.array([4])
        val_mean = np.mean(neighbor_valences)
        val_std = np.std(neighbor_valences)
        val_min = np.min(neighbor_valences)
        val_max = np.max(neighbor_valences)
        
        # Count of V3, V4, V5+ neighbors
        n_v3 = np.sum(neighbor_valences == 3)
        n_v4 = np.sum(neighbor_valences == 4)
        n_v5plus = np.sum(neighbor_valences >= 5)
        
        # Curvature gradient
        neighbor_curvatures = self.curvatures[neighbors] if neighbors else np.array([0])
        curv_gradient = curvature - np.mean(neighbor_curvatures)
        
        # Face angles at this vertex
        face_angles = self._compute_face_angles(vi)
        angle_mean = np.mean(face_angles) if face_angles else np.pi/2
        angle_std = np.std(face_angles) if len(face_angles) > 1 else 0
        angle_min = np.min(face_angles) if face_angles else np.pi/2
        angle_max = np.max(face_angles) if face_angles else np.pi/2
        
        # Normalize valence (4 is regular)
        valence_normalized = (valence - 4) / 4.0
        
        features = np.array([
            valence_normalized,
            is_boundary,
            curvature,
            curv_gradient,
            val_mean / 4.0,
            val_std / 2.0,
            val_min / 4.0,
            val_max / 4.0,
            n_v3 / max(1, len(neighbors)),
            n_v4 / max(1, len(neighbors)),
            n_v5plus / max(1, len(neighbors)),
            angle_mean / np.pi,
            angle_std / np.pi,
            angle_min / np.pi,
            angle_max / np.pi,
        ], dtype=np.float32)
        
        return features
    
    def _compute_face_angles(self, vertex_idx: int) -> List[float]:
        """Compute the angle at vertex_idx in each of its faces."""
        angles = []
        vi = vertex_idx
        v_pos = self.mesh.vertices[vi]
        
        for fi in self.vertex_faces[vi]:
            face = self.mesh.faces[fi]
            # Find position of vi in face
            idx = list(face).index(vi)
            n = len(face)
            
            # Get adjacent vertices in face
            prev_vi = face[(idx - 1) % n]
            next_vi = face[(idx + 1) % n]
            
            # Compute vectors
            vec1 = self.mesh.vertices[prev_vi] - v_pos
            vec2 = self.mesh.vertices[next_vi] - v_pos
            
            # Compute angle
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1, 1)
                angles.append(np.arccos(cos_angle))
        
        return angles
    
    def get_all_pole_info(self, curvature_threshold: float = 0.15) -> List[PoleInfo]:
        """Get information about all irregular vertices."""
        poles = []
        
        for vi in range(self.mesh.num_vertices):
            valence = self.valences[vi]
            if valence == 4:  # Regular vertex
                continue
            
            neighbors = self.vertex_neighbors[vi]
            neighbor_valences = [self.valences[n] for n in neighbors]
            face_angles = self._compute_face_angles(vi)
            
            pole = PoleInfo(
                vertex_idx=vi,
                valence=valence,
                curvature=self.curvatures[vi],
                is_boundary=self.is_boundary[vi],
                neighbor_valences=neighbor_valences,
                face_angles=face_angles,
            )
            poles.append(pole)
        
        return poles


if TORCH_AVAILABLE:
    class PoleClassifierNet(nn.Module):
        """
        Neural network to classify poles as defects or structural.
        
        Architecture: MLP with residual connections
        Input: 15-dimensional feature vector per vertex
        Output: 2 classes (fix, keep)
        """
        
        def __init__(self, input_dim: int = 15, hidden_dim: int = 64):
            super().__init__()
            
            self.input_norm = nn.BatchNorm1d(input_dim)
            
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
            
            self.fc_out = nn.Linear(hidden_dim // 2, 2)
            
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(x)
            
            # First block
            h = F.relu(self.bn1(self.fc1(x)))
            h = self.dropout(h)
            
            # Second block with residual (if dimensions match)
            h2 = F.relu(self.bn2(self.fc2(h)))
            h2 = self.dropout(h2)
            h = h + h2  # Residual connection
            
            # Third block
            h = F.relu(self.bn3(self.fc3(h)))
            h = self.dropout(h)
            
            # Output
            return self.fc_out(h)
        
        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Get class probabilities."""
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class RuleBasedPoleClassifier:
    """
    Rule-based pole classifier (fallback when PyTorch not available).
    
    Uses curvature and neighbor analysis to classify poles.
    """
    
    def __init__(self, curvature_threshold: float = 0.15):
        self.curvature_threshold = curvature_threshold
    
    def classify(self, pole: PoleInfo) -> Tuple[str, float]:
        """
        Classify a pole as 'fix' or 'keep'.
        
        Returns: (classification, confidence)
        """
        # Boundary vertices should be kept
        if pole.is_boundary:
            return ('keep', 0.9)
        
        # High curvature areas - likely structural
        if pole.curvature > self.curvature_threshold:
            return ('keep', 0.7 + 0.3 * min(1, pole.curvature / 0.5))
        
        # Flat area with irregular valence - likely defect
        # V3 poles in flat areas are almost always defects
        if pole.valence == 3:
            # Check if there's a nearby V5 (common V3-V5 pair from triangles)
            has_v5_neighbor = 5 in pole.neighbor_valences
            if has_v5_neighbor:
                return ('fix', 0.85)
            return ('fix', 0.7)
        
        # V5 poles in flat areas
        if pole.valence == 5:
            has_v3_neighbor = 3 in pole.neighbor_valences
            if has_v3_neighbor:
                return ('fix', 0.85)
            return ('fix', 0.6)
        
        # V6+ poles are usually defects unless at feature corners
        if pole.valence >= 6:
            if pole.curvature < self.curvature_threshold * 0.5:
                return ('fix', 0.9)
            return ('fix', 0.5)
        
        # V2 poles (boundary-like)
        if pole.valence == 2:
            return ('keep', 0.5)  # Uncertain
        
        return ('keep', 0.5)  # Default: uncertain


class HybridPoleClassifier:
    """
    Hybrid classifier that uses neural network when available,
    falls back to rule-based otherwise.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.rule_classifier = RuleBasedPoleClassifier()
        
        if TORCH_AVAILABLE and model_path:
            try:
                self.model = PoleClassifierNet()
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                logger.info(f"Loaded pole classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}, using rule-based classifier")
                self.model = None
    
    def classify_poles(self, mesh: Mesh) -> List[PoleInfo]:
        """Classify all poles in the mesh."""
        extractor = PoleFeatureExtractor(mesh)
        poles = extractor.get_all_pole_info()
        
        if self.model is not None and TORCH_AVAILABLE:
            # Use neural network
            features = np.array([extractor.get_pole_features(p.vertex_idx) for p in poles])
            
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32)
                probs = self.model.predict_proba(x)
                predictions = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1).values
            
            classes = ['fix', 'keep']
            for i, pole in enumerate(poles):
                pole.prediction = classes[predictions[i].item()]
                pole.confidence = confidences[i].item()
        else:
            # Use rule-based classifier
            for pole in poles:
                pred, conf = self.rule_classifier.classify(pole)
                pole.prediction = pred
                pole.confidence = conf
        
        return poles
    
    def get_fixable_poles(
        self, 
        mesh: Mesh, 
        min_confidence: float = 0.6
    ) -> List[PoleInfo]:
        """Get poles that should be fixed, sorted by confidence."""
        poles = self.classify_poles(mesh)
        
        fixable = [p for p in poles if p.prediction == 'fix' and p.confidence >= min_confidence]
        
        # Sort by confidence (highest first)
        fixable.sort(key=lambda p: -p.confidence)
        
        return fixable


def export_pole_classifications_for_blender(
    mesh: Mesh,
    classifier: HybridPoleClassifier,
    output_path: str,
    min_confidence: float = 0.6
) -> dict:
    """
    Export pole classifications in a format Blender can use.
    
    Returns a dict with vertex indices and their classifications.
    Also saves to JSON file for Blender script.
    """
    import json
    
    poles = classifier.classify_poles(mesh)
    
    # Build export data
    export_data = {
        "vertices_to_fix": [],
        "vertices_to_keep": [],
        "all_poles": [],
    }
    
    for pole in poles:
        pole_data = {
            "vertex_idx": int(pole.vertex_idx),
            "valence": int(pole.valence),
            "curvature": float(pole.curvature),
            "is_boundary": bool(pole.is_boundary),
            "prediction": pole.prediction,
            "confidence": float(pole.confidence),
        }
        export_data["all_poles"].append(pole_data)
        
        if pole.prediction == 'fix' and pole.confidence >= min_confidence:
            export_data["vertices_to_fix"].append(int(pole.vertex_idx))
        elif pole.prediction == 'keep':
            export_data["vertices_to_keep"].append(int(pole.vertex_idx))
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported {len(export_data['vertices_to_fix'])} fixable poles, "
                f"{len(export_data['vertices_to_keep'])} keeper poles to {output_path}")
    
    return export_data


def create_training_data(
    high_quality_meshes: List[Mesh],
    low_quality_meshes: List[Mesh],
    output_path: str
) -> None:
    """
    Create training data from pairs of high/low quality meshes.
    
    High quality meshes: poles are mostly structural (label: keep)
    Low quality meshes: poles in flat areas are defects (label: fix based on curvature)
    
    This is a semi-supervised approach using curvature as a guide.
    """
    features_list = []
    labels_list = []
    
    # From high-quality meshes: poles are likely structural
    for mesh in high_quality_meshes:
        extractor = PoleFeatureExtractor(mesh)
        poles = extractor.get_all_pole_info()
        
        for pole in poles:
            features = extractor.get_pole_features(pole.vertex_idx)
            # High-quality mesh poles: label based on curvature
            # High curvature = keep, low curvature = still likely structural
            label = 1  # keep
            features_list.append(features)
            labels_list.append(label)
    
    # From low-quality meshes: poles in flat areas are likely defects
    for mesh in low_quality_meshes:
        extractor = PoleFeatureExtractor(mesh)
        poles = extractor.get_all_pole_info()
        
        for pole in poles:
            features = extractor.get_pole_features(pole.vertex_idx)
            # Low-quality mesh: flat area poles are defects
            label = 0 if pole.curvature < 0.15 else 1  # fix or keep
            features_list.append(features)
            labels_list.append(label)
    
    # Save as numpy arrays
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    
    np.savez(output_path, features=features, labels=labels)
    logger.info(f"Saved training data: {len(features)} samples to {output_path}")
