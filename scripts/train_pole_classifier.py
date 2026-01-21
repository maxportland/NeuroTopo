#!/usr/bin/env python3
"""
Train the pole classifier neural network.

This script trains the PoleClassifierNet on labeled data.
Training data can come from:
1. Automatic labeling based on curvature (semi-supervised)
2. Manual labels from artist-reviewed meshes
"""

import sys
sys.path.insert(0, '/Users/maxdavis/Projects/MeshRepair/src')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

from neurotopo.core.mesh import Mesh
from neurotopo.analysis.neural.pole_classifier import (
    PoleClassifierNet,
    PoleFeatureExtractor,
    RuleBasedPoleClassifier,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_training_data_from_mesh(
    mesh: Mesh,
    label_source: str = 'curvature'  # 'curvature' or 'rules'
) -> tuple:
    """
    Create training data from a single mesh.
    
    Returns:
        features: np.ndarray of shape (n_poles, n_features)
        labels: np.ndarray of shape (n_poles,) with 0=fix, 1=keep
    """
    extractor = PoleFeatureExtractor(mesh)
    poles = extractor.get_all_pole_info()
    
    if len(poles) == 0:
        return np.array([]), np.array([])
    
    features = np.array([extractor.get_pole_features(p.vertex_idx) for p in poles])
    
    if label_source == 'curvature':
        # Label based on curvature threshold
        labels = np.array([
            1 if p.curvature > 0.15 or p.is_boundary else 0 
            for p in poles
        ])
    elif label_source == 'rules':
        # Use rule-based classifier
        classifier = RuleBasedPoleClassifier()
        labels = np.array([
            0 if classifier.classify(p)[0] == 'fix' else 1
            for p in poles
        ])
    else:
        raise ValueError(f"Unknown label source: {label_source}")
    
    return features, labels


def train_pole_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray = None,
    val_labels: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = None,
) -> PoleClassifierNet:
    """
    Train the pole classifier network.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")
    
    # Create model
    input_dim = train_features.shape[1]
    model = PoleClassifierNet(input_dim=input_dim)
    model.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_features is not None:
        val_dataset = TensorDataset(
            torch.tensor(val_features, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Handle class imbalance with weighted loss
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        len(train_labels) / (len(class_counts) * class_counts),
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        val_acc = 0
        if val_features is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100. * val_correct / val_total
            scheduler.step(1 - val_acc/100)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    if save_path:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")
    
    return model


def main():
    """Train on available mesh data."""
    
    # Collect training data from mesh files
    mesh_dir = Path('/Users/maxdavis/Projects/MeshRepair/results')
    
    all_features = []
    all_labels = []
    
    # Find all retopo output meshes
    for obj_file in mesh_dir.rglob('*_retopo.obj'):
        logger.info(f"Processing {obj_file}")
        try:
            mesh = Mesh.from_file(obj_file)
            features, labels = create_training_data_from_mesh(mesh, label_source='rules')
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                logger.info(f"  -> {len(features)} poles")
        except Exception as e:
            logger.warning(f"  -> Error: {e}")
    
    if not all_features:
        logger.error("No training data found!")
        return
    
    # Combine all data
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    logger.info(f"Total training samples: {len(features)}")
    logger.info(f"Class distribution: fix={np.sum(labels==0)}, keep={np.sum(labels==1)}")
    
    # Split into train/val
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    
    train_features = features[indices[:split]]
    train_labels = labels[indices[:split]]
    val_features = features[indices[split:]]
    val_labels = labels[indices[split:]]
    
    # Train
    model = train_pole_classifier(
        train_features, train_labels,
        val_features, val_labels,
        epochs=100,
        save_path='/Users/maxdavis/Projects/MeshRepair/models/pole_classifier.pt'
    )
    
    # Test final accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(val_features, dtype=torch.float32))
        _, predicted = test_outputs.max(1)
        accuracy = (predicted.numpy() == val_labels).mean()
        logger.info(f"Final validation accuracy: {accuracy*100:.1f}%")


if __name__ == '__main__':
    main()
