"""Small CNN designs (CNN-2 / CNN-3) with the 64-value embedding interface.

Stand-in implementation of the agreed contract until Puneet's module lands:
forward(x) -> logits [B,10]; embed(x) -> [B,64] after the dense ReLU and before
dropout. Architecture metadata is saved with every checkpoint so inference can
rebuild the exact network.
"""
from __future__ import annotations

import torch
from torch import nn

CHANNELS = (32, 64, 128)
EMBEDDING_DIM = 64


class SmallCNN(nn.Module):
    def __init__(self, blocks=2, dropout=0.30, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        if blocks not in (2, 3):
            raise ValueError("Declared designs use 2 or 3 conv blocks")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")
        layers, channels_in = [], 1
        for channels in CHANNELS[:blocks]:
            # 3x3 conv keeps 28x28 with padding 1; 2x2 max-pool halves it: 28 -> 14 -> 7 -> 3
            layers += [nn.Conv2d(channels_in, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            channels_in = channels
        self.features = nn.Sequential(*layers)
        side = 28 // 2 ** blocks
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(channels_in * side * side, embedding_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, 10)
        self.arch = {"name": "SmallCNN", "blocks": blocks, "dropout": dropout,
                     "channels": list(CHANNELS[:blocks]), "embedding_dim": embedding_dim}

    def embed(self, x):
        return self.dense(self.features(x))

    def forward(self, x):
        return self.classifier(self.dropout(self.embed(x)))


def build_model(arch):
    if arch.get("name", "SmallCNN") != "SmallCNN":
        raise ValueError(f"Unknown architecture {arch.get('name')}")
    return SmallCNN(blocks=arch["blocks"], dropout=arch["dropout"], embedding_dim=arch.get("embedding_dim", EMBEDDING_DIM))


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def shape_table(model):
    """Layer-by-layer output shapes for one image; used in the report's architecture table."""
    rows, x = [], torch.zeros(1, 1, 28, 28)
    with torch.inference_mode():
        for name, layer in list(model.features.named_children()) + [("dense", model.dense), ("dropout", model.dropout), ("classifier", model.classifier)]:
            x = layer(x)
            rows.append({"layer": f"{name}:{type(layer).__name__}", "output_shape": list(x.shape[1:]),
                         "parameters": int(sum(p.numel() for p in layer.parameters()))})
    return rows
