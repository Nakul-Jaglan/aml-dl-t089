"""CNN data adapter: yields (image, label, original_id) from the saved Phase 1 splits.

Stand-in implementation of the agreed handoff contract. When Sudeshhna's adapter
lands, keep the same return types so the runner does not change:
image float32 [1,28,28] in [0,1], label int64, original official_train row ID int64.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..features import normalized_images
from ..utils import ROOT, sha256

SPLIT_FILES = {"smoke": "data/smoke_indices.npz", "full": "data/full_indices.npz"}
AUGMENTATIONS = {"none", "translate2"}
MAX_SHIFT = 2  # the one predeclared fit-only policy: integer shifts in [-2, 2] px, zero fill


def split_ids(profile):
    """Original official_train IDs for fit and validation plus the split file digest."""
    if profile not in SPLIT_FILES:
        raise ValueError(f"Unknown Phase 2 profile {profile!r}")
    path = ROOT / SPLIT_FILES[profile]
    with np.load(path) as saved:
        fit, val = saved["train_idx"].astype(np.int64), saved["val_idx"].astype(np.int64)
    if np.intersect1d(fit, val).size:
        raise ValueError("Fit and validation IDs overlap")
    return fit, val, sha256(path)


def translate(image, dx, dy):
    """Shift a [1,28,28] tensor by (dx, dy) pixels with zero fill; size is preserved."""
    if max(abs(dx), abs(dy)) > MAX_SHIFT:
        raise ValueError("Shift outside the declared policy")
    padded = torch.nn.functional.pad(image, (MAX_SHIFT,) * 4)
    top, left = MAX_SHIFT - dy, MAX_SHIFT - dx
    return padded[:, top:top + 28, left:left + 28]


class FashionIDDataset(Dataset):
    """In-memory dataset; IDs travel with every example so predictions can be joined."""

    def __init__(self, images, labels, ids, augmentation="none", seed=42):
        if augmentation not in AUGMENTATIONS:
            raise ValueError(f"Unknown augmentation {augmentation!r}")
        ids = np.array(ids, dtype=np.int64)  # owned, writable copy
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate original IDs")
        self.x = torch.from_numpy(normalized_images(images[ids])).unsqueeze(1)
        self.y = torch.from_numpy(np.asarray(labels)[ids].astype(np.int64))
        self.ids = torch.from_numpy(ids)
        self.augmentation = augmentation
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.x[index]
        if self.augmentation == "translate2":
            dx, dy = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (2,), generator=self.generator).tolist()
            x = translate(x, dx, dy)
        return x, self.y[index], self.ids[index]


def make_loader(dataset, batch_size, shuffle, seed=42):
    """num_workers=0 keeps shuffling and augmentation reproducible on one machine."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)
