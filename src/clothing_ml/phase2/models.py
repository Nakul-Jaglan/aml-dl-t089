"""Small convolutional neural network architecture for Fashion-MNIST clothing category recognition.

Phase 2 model interface owned by Puneet.
Supports depth=2 (CNN-2 reference) and depth=3 (CNN-3 architecture comparison),
configurable dropout, and provides a 64-dimensional feature extraction interface
via embed(x) / encode(x) for hybrid evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

CHANNELS = (32, 64, 128)
EMBEDDING_DIM = 64


class SmallCNN(nn.Module):
    """Compact CNN architecture for 28x28 grayscale clothing category recognition.

    Args:
        blocks: Number of convolutional blocks (2 or 3).
        dropout: Dropout probability applied to the 64-dimensional embedding.
        embedding_dim: Dimensionality of pre-classifier latent projection (default: 64).
        depth: Alias for blocks.
        channels: Custom channel sequence (defaults to (32, 64, 128)[:blocks]).
    """

    def __init__(
        self,
        blocks: int = 2,
        dropout: float = 0.30,
        embedding_dim: int = EMBEDDING_DIM,
        depth: Optional[int] = None,
        channels: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        num_blocks = depth if depth is not None else blocks
        if num_blocks not in (2, 3):
            raise ValueError("depth must be 2 or 3")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")

        self.depth = num_blocks
        self.blocks_count = num_blocks
        self.dropout_rate = float(dropout)
        self.embedding_dim = embedding_dim

        if channels is not None:
            channel_list = list(channels[:num_blocks])
        else:
            channel_list = list(CHANNELS[:num_blocks])
        self.channel_list = channel_list

        layers: List[nn.Module] = []
        channels_in = 1
        for ch in channel_list:
            layers += [
                nn.Conv2d(channels_in, ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            channels_in = ch
        self.features = nn.Sequential(*layers)

        side = 28 // (2**num_blocks)
        self.spatial_side = side
        self.flattened_features = channels_in * side * side

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_features, embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, 10)

        self.arch = {
            "name": "SmallCNN",
            "blocks": num_blocks,
            "dropout": dropout,
            "channels": channel_list,
            "embedding_dim": embedding_dim,
        }

    @property
    def blocks(self) -> nn.Sequential:
        """Alias for features matching Puneet's Phase 2 work guide."""
        return self.features

    @property
    def projection(self) -> nn.Linear:
        """Alias for dense[1] Linear projection matching Puneet's Phase 2 work guide."""
        return self.dense[1]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned 64-value pre-dropout feature representations per image."""
        return self.dense(self.features(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for embed(x) matching Puneet's Phase 2 interface contract."""
        return self.embed(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting unnormalized class logits [B, 10]."""
        return self.classifier(self.dropout(self.embed(x)))


def build_model(arch: Dict[str, Any]) -> SmallCNN:
    """Build SmallCNN instance from an architecture metadata dictionary."""
    if arch.get("name", "SmallCNN") != "SmallCNN":
        raise ValueError(f"Unknown architecture {arch.get('name')}")
    channels = arch.get("channels")
    return SmallCNN(
        blocks=arch.get("blocks", 2),
        dropout=arch.get("dropout", 0.30),
        embedding_dim=arch.get("embedding_dim", EMBEDDING_DIM),
        channels=channels,
    )


def parameter_count(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


count_parameters = parameter_count


def shape_table(model: SmallCNN) -> List[Dict[str, Any]]:
    """Layer-by-layer output shapes for one image; used in report tables."""
    rows = []
    x = torch.zeros(1, 1, 28, 28)
    with torch.inference_mode():
        for name, layer in list(model.features.named_children()) + [
            ("dense", model.dense),
            ("dropout", model.dropout),
            ("classifier", model.classifier),
        ]:
            x = layer(x)
            rows.append({
                "layer": f"{name}:{type(layer).__name__}",
                "output_shape": list(x.shape[1:]),
                "parameters": int(sum(p.numel() for p in layer.parameters())),
            })
    return rows


def get_parameter_breakdown(model: SmallCNN) -> Dict[str, Any]:
    """Calculate exact layer-by-layer parameter counts and theoretical arithmetic."""
    breakdown = {}
    total = 0

    # Conv blocks
    conv_idx = 1
    for i in range(0, len(model.features), 3):
        conv_layer = model.features[i]
        count = sum(p.numel() for p in conv_layer.parameters())
        breakdown[f"block{conv_idx}_conv"] = {
            "in_channels": conv_layer.in_channels,
            "out_channels": conv_layer.out_channels,
            "kernel": conv_layer.kernel_size,
            "params": count,
            "formula": (
                f"{conv_layer.out_channels} * "
                f"({conv_layer.in_channels} * {conv_layer.kernel_size[0]} * {conv_layer.kernel_size[1]} + 1) = {count}"
            ),
        }
        total += count
        conv_idx += 1

    # Dense Projection
    proj_layer = model.dense[1]
    proj_count = sum(p.numel() for p in proj_layer.parameters())
    breakdown["projection_dense"] = {
        "in_features": proj_layer.in_features,
        "out_features": proj_layer.out_features,
        "params": proj_count,
        "formula": f"{proj_layer.out_features} * ({proj_layer.in_features} + 1) = {proj_count}",
    }
    total += proj_count

    # Classifier
    clf_count = sum(p.numel() for p in model.classifier.parameters())
    breakdown["classifier_head"] = {
        "in_features": model.classifier.in_features,
        "out_features": model.classifier.out_features,
        "params": clf_count,
        "formula": f"{model.classifier.out_features} * ({model.classifier.in_features} + 1) = {clf_count}",
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
        "arch": model.arch,
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
    payload = torch.load(path, map_location=map_location, weights_only=False)
    arch = payload.get("arch")
    if arch is not None:
        model = build_model(arch)
    else:
        depth = payload.get("depth", 2)
        dropout = payload.get("dropout", 0.30)
        model = SmallCNN(depth=depth, dropout=dropout)
    model.load_state_dict(payload["state_dict"])
    return model, payload
