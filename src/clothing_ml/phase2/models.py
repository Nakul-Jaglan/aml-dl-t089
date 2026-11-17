"""Small convolutional neural network architecture for Fashion-MNIST clothing category recognition.

Phase 2 model interface owned by Puneet.
Supports depth=2 (CNN-2 reference) and depth=3 (CNN-3 architecture comparison),
configurable dropout, and provides a 64-dimensional feature extraction interface
via encode(x) for hybrid evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Compact CNN architecture for 28x28 grayscale clothing category recognition.

    Args:
        depth: Number of convolutional blocks (2 or 3).
               depth=2: 2 blocks -> [B, 64, 7, 7] -> 220,234 parameters.
               depth=3: 3 blocks -> [B, 64, 3, 3] -> 93,322 parameters.
        dropout: Dropout probability applied to the 64-dimensional embedding
                 before the 10-way classification head.
    """

    def __init__(self, depth: int = 2, dropout: float = 0.30):
        super().__init__()
        if depth not in (2, 3):
            raise ValueError("depth must be 2 or 3")

        self.depth = depth
        self.dropout_rate = float(dropout)

        channels = (1, 32, 64, 64)[: depth + 1]
        layers = []
        for incoming, outgoing in zip(channels, channels[1:]):
            layers += [
                nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        self.blocks = nn.Sequential(*layers)

        side = 28 // (2**depth)
        self.spatial_side = side
        self.flattened_features = 64 * side * side
        self.projection = nn.Linear(self.flattened_features, 64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 10)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned 64-value pre-dropout feature representations per image.

        Args:
            x: Input tensor shaped [B, 1, 28, 28].

        Returns:
            Embeddings shaped [B, 64] after linear projection and ReLU activation.
        """
        features = self.blocks(x).flatten(1)
        return torch.relu(self.projection(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting class logits.

        Args:
            x: Input tensor shaped [B, 1, 28, 28].

        Returns:
            Unnormalized class logits shaped [B, 10].
        """
        embedding = self.encode(x)
        return self.classifier(self.dropout(embedding))


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_breakdown(model: SmallCNN) -> Dict[str, Any]:
    """Calculate exact layer-by-layer parameter counts and theoretical arithmetic."""
    breakdown = {}
    total = 0

    # Block 1: Conv2d(1, 32, 3, pad=1) -> 32 * (1 * 3 * 3 + 1) = 320
    b1_conv = model.blocks[0]
    b1_count = sum(p.numel() for p in b1_conv.parameters())
    breakdown["block1_conv"] = {
        "in_channels": 1,
        "out_channels": 32,
        "kernel": (3, 3),
        "params": b1_count,
        "formula": "32 * (1 * 3 * 3 + 1) = 320",
    }
    total += b1_count

    # Block 2: Conv2d(32, 64, 3, pad=1) -> 64 * (32 * 3 * 3 + 1) = 18,496
    b2_conv = model.blocks[3]
    b2_count = sum(p.numel() for p in b2_conv.parameters())
    breakdown["block2_conv"] = {
        "in_channels": 32,
        "out_channels": 64,
        "kernel": (3, 3),
        "params": b2_count,
        "formula": "64 * (32 * 3 * 3 + 1) = 18,496",
    }
    total += b2_count

    if model.depth == 3:
        # Block 3: Conv2d(64, 64, 3, pad=1) -> 64 * (64 * 3 * 3 + 1) = 36,928
        b3_conv = model.blocks[6]
        b3_count = sum(p.numel() for p in b3_conv.parameters())
        breakdown["block3_conv"] = {
            "in_channels": 64,
            "out_channels": 64,
            "kernel": (3, 3),
            "params": b3_count,
            "formula": "64 * (64 * 3 * 3 + 1) = 36,928",
        }
        total += b3_count

    # Dense Projection: Linear(64 * side * side, 64)
    # side=7 for depth=2 -> 64 * (3136 + 1) = 200,768
    # side=3 for depth=3 -> 64 * (576 + 1) = 36,928
    proj_count = sum(p.numel() for p in model.projection.parameters())
    breakdown["projection_dense"] = {
        "in_features": model.flattened_features,
        "out_features": 64,
        "params": proj_count,
        "formula": f"64 * ({model.flattened_features} + 1) = {proj_count}",
    }
    total += proj_count

    # Classifier: Linear(64, 10) -> 10 * (64 + 1) = 650
    clf_count = sum(p.numel() for p in model.classifier.parameters())
    breakdown["classifier_head"] = {
        "in_features": 64,
        "out_features": 10,
        "params": clf_count,
        "formula": "10 * (64 + 1) = 650",
    }
    total += clf_count

    return {
        "depth": model.depth,
        "spatial_side": model.spatial_side,
        "layers": breakdown,
        "total_trainable_parameters": total,
        "verified": total == count_parameters(model),
    }


def save_checkpoint(
    path: Union[str, Path],
    model: SmallCNN,
    config: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint including architecture configuration and weights."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth": model.depth,
        "dropout": model.dropout_rate,
        "state_dict": model.state_dict(),
        "trainable_parameters": count_parameters(model),
        "config": config or {},
        "epoch": epoch,
        "metrics": metrics or {},
    }
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(target_path)


def load_checkpoint(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Tuple[SmallCNN, Dict[str, Any]]:
    """Load model checkpoint across devices and reconstruct SmallCNN instance."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    depth = payload.get("depth", 2)
    dropout = payload.get("dropout", 0.30)
    model = SmallCNN(depth=depth, dropout=dropout)
    model.load_state_dict(payload["state_dict"])
    return model, payload
