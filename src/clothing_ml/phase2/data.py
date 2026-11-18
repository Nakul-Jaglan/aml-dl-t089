"""CNN data adapter (Sudeshhna): (image, label, original_id) from the saved Phase 1 splits.

Boundaries
- HOG keeps receiving raw uint8 images through the Phase 1 feature code.
- The CNN receives float32 [1, 28, 28] per image, divided by 255 exactly once here.
- Every item carries its original row ID in the source partition (official_train
  for fit/validation), so predictions can be joined across models however a
  DataLoader shuffles. Stage and source split live in the run/prediction metadata.

Augmentation policies (fit loader only; validation, test and embedding extraction
always use the canonical image):
- "none"           canonical image.
- "translate2"     legacy stand-in used by full_cnn2_aug_s42_fdeb788e: every draw
                   picks dx, dy uniformly in [-2, 2]. Kept bit-for-bit so that run
                   stays reproducible.
- "translate2_p50" the work-guide policy: with probability 0.5 pick dx, dy uniformly
                   in [-2, 2] (zero fill, no wraparound, 28x28 kept); otherwise
                   return the same input.

The public names used by runner.py and fusion.py (split_ids, translate,
FashionIDDataset, make_loader, AUGMENTATIONS) keep their earlier contract.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data import CLASS_NAMES
from ..utils import ROOT, sha256

SPLIT_FILES = {"smoke": "data/smoke_indices.npz", "full": "data/full_indices.npz"}
MAX_SHIFT = 2
POLICIES = {
    "none": None,
    "translate2": {"max_shift": MAX_SHIFT, "p": 1.0, "fill": 0.0, "wraparound": False},
    "translate2_p50": {"max_shift": MAX_SHIFT, "p": 0.5, "fill": 0.0, "wraparound": False},
}
AUGMENTATIONS = set(POLICIES)


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
    """Shift a [1,28,28] tensor right by dx and down by dy pixels with zero fill.

    Pixels pushed past the frame are dropped (no wraparound); the uncovered
    border is 0, so shape and the [0, 1] range are preserved.
    """
    if max(abs(dx), abs(dy)) > MAX_SHIFT:
        raise ValueError("Shift outside the declared policy")
    padded = torch.nn.functional.pad(image, (MAX_SHIFT,) * 4)
    top, left = MAX_SHIFT - dy, MAX_SHIFT - dx
    return padded[:, top:top + 28, left:left + 28]


class RandomShift:
    """Seeded fit-only translation. With probability p, shift by (dx, dy) in [-s, s]^2.

    `last_offset` records the (dx, dy) applied to the most recent call, or None
    when the image was returned unchanged; useful for captions and audits.
    """

    def __init__(self, max_shift=MAX_SHIFT, p=0.5, seed=42):
        if not 0 <= max_shift <= MAX_SHIFT:
            raise ValueError("max_shift outside the declared policy")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be a probability")
        self.max_shift, self.p, self.seed = int(max_shift), float(p), int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.last_offset = None

    def __call__(self, image):
        # p == 1 draws only the offsets, which reproduces the legacy "translate2" stream exactly.
        if self.p < 1.0 and float(torch.rand(1, generator=self.generator)) >= self.p:
            self.last_offset = None
            return image
        dx, dy = torch.randint(-self.max_shift, self.max_shift + 1, (2,), generator=self.generator).tolist()
        self.last_offset = (dx, dy)
        return translate(image, dx, dy)

    def describe(self):
        return {"type": "RandomShift", "max_shift": self.max_shift, "p": self.p, "seed": self.seed,
                "fill": 0.0, "wraparound": False, "offsets": "uniform integers, independent dx and dy"}


def make_transform(augmentation, seed=42):
    if augmentation not in POLICIES:
        raise ValueError(f"Unknown augmentation {augmentation!r}")
    policy = POLICIES[augmentation]
    return None if policy is None else RandomShift(policy["max_shift"], policy["p"], seed)


def _validate_raw(raw, labels):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (28, 28):
        raise ValueError("Expected raw image array of shape (N, 28, 28)")
    if raw.dtype != np.uint8:
        raise TypeError("Expected raw uint8 images; the adapter divides by 255 once")
    labels = np.asarray(labels)
    if labels.shape != (len(raw),):
        raise ValueError("Labels must align with raw images")
    return raw, labels


class ClothingImages(Dataset):
    """Raw uint8 partition + original IDs -> (float32 [1,28,28] in [0,1], int label, int ID)."""

    def __init__(self, raw, labels, ids, transform=None):
        self.raw, self.labels = _validate_raw(raw, labels)
        ids = np.array(ids, dtype=np.int64)  # owned, writable copy
        if ids.ndim != 1:
            raise ValueError("IDs must be one-dimensional")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate original IDs")
        if len(ids) and (ids.min() < 0 or ids.max() >= len(self.raw)):
            raise IndexError("Original ID outside the source partition")
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        original_id = int(self.ids[index])
        image = torch.from_numpy(self.raw[original_id]).float()
        image = image.unsqueeze(0).div(255.0)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[original_id]), original_id


class FashionIDDataset(ClothingImages):
    """Runner-facing name: ClothingImages with the transform chosen by augmentation name."""

    def __init__(self, images, labels, ids, augmentation="none", seed=42):
        super().__init__(images, labels, ids, make_transform(augmentation, seed))
        self.augmentation = augmentation


def make_loader(dataset, batch_size, shuffle, seed=42):
    """num_workers=0 and a seeded generator keep shuffling and augmentation reproducible."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, drop_last=False)


def make_loaders(profile="full", augmentation="none", seed=42, batch_size=128, eval_batch_size=512,
                 images=None, labels=None):
    """The one loader factory handed to model and training code.

    Returns {"fit", "val"} DataLoaders and a metadata dict (split file, digest,
    sizes, policy). Only the fit loader shuffles and augments.
    """
    if images is None:
        from ..data import load_official
        images, labels = load_official("train")
    fit_ids, val_ids, split_sha = split_ids(profile)
    fit_set = FashionIDDataset(images, labels, fit_ids, augmentation, seed=seed)
    val_set = FashionIDDataset(images, labels, val_ids, "none")
    meta = {"profile": profile, "source_split": "official_train", "split_file": SPLIT_FILES[profile],
            "split_sha256": split_sha, "fit_n": len(fit_ids), "val_n": len(val_ids),
            "seed": seed, "batch_size": batch_size, "eval_batch_size": eval_batch_size,
            "augmentation": augmentation,
            "fit_transform": None if fit_set.transform is None else fit_set.transform.describe(),
            "val_transform": None, "num_workers": 0, "drop_last": False}
    return {"fit": make_loader(fit_set, batch_size, shuffle=True, seed=seed),
            "val": make_loader(val_set, eval_batch_size, shuffle=False)}, meta


def check_batch(batch, raw, labels, canonical=True):
    """Handoff checks for one collated batch; raises AssertionError on any violation.

    canonical=True also requires each image to equal raw[id] / 255 exactly (use it
    for validation/test or the unaugmented fit loader).
    """
    images, targets, ids = batch
    assert images.dtype == torch.float32 and images.ndim == 4 and tuple(images.shape[1:]) == (1, 28, 28), images.shape
    assert targets.dtype == torch.int64 and tuple(targets.shape) == (len(images),), targets.dtype
    assert ids.dtype == torch.int64 and tuple(ids.shape) == (len(images),), ids.dtype
    assert bool(torch.isfinite(images).all()) and float(images.min()) >= 0.0 and float(images.max()) <= 1.0
    id_array = ids.numpy()
    assert len(np.unique(id_array)) == len(id_array), "duplicate IDs in a batch"
    assert np.array_equal(targets.numpy(), np.asarray(labels)[id_array]), "labels do not match raw labels by ID"
    if canonical:
        expected = torch.from_numpy(np.asarray(raw)[id_array]).float().unsqueeze(1).div(255.0)
        assert torch.equal(images, expected), "image does not match raw[id] / 255"
    return {"shape": list(images.shape), "image_dtype": str(images.dtype).replace("torch.", ""),
            "label_dtype": str(targets.dtype).replace("torch.", ""), "min": float(images.min()),
            "max": float(images.max()), "finite": True,
            "examples": [{"original_id": int(i), "label": int(y), "class": CLASS_NAMES[int(y)]}
                         for i, y in zip(id_array[:5], targets.numpy()[:5])]}
