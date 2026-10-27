"""Checksum-verified official IDX data; saved IDs preserve the official test split."""
from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .utils import ROOT, sha256, write_json

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def download_file(filename, data_dir=None):
    directory = Path(data_dir) if data_dir else ROOT / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists():
        print(f"Downloading official data: {filename}", flush=True)
        request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "FashionMNIST-course-project"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    actual = hashlib.md5(target.read_bytes()).hexdigest()  # published content checksum, not security
    if actual != FILES[filename]:
        raise ValueError(f"Dataset checksum mismatch: {target}; remove and download again")
    return target


def parse_idx(payload: bytes):
    if len(payload) < 8:
        raise ValueError("Truncated IDX header")
    magic, count = struct.unpack(">II", payload[:8])
    if magic == 2051:
        if len(payload) < 16:
            raise ValueError("Truncated image header")
        rows, columns = struct.unpack(">II", payload[8:16])
        if (rows, columns) != (28, 28) or len(payload) != 16 + count * rows * columns:
            raise ValueError("Unexpected dimensions or truncated IDX images")
        return np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(count, rows, columns).copy()
    if magic == 2049:
        if len(payload) != 8 + count:
            raise ValueError("Unexpected IDX label length")
        return np.frombuffer(payload, dtype=np.uint8, offset=8).astype(np.int64)
    raise ValueError(f"Unsupported IDX magic number {magic}")


def load_official(split="train", data_dir=None):
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    prefix = "train" if split == "train" else "t10k"
    image_file = download_file(f"{prefix}-images-idx3-ubyte.gz", data_dir)
    label_file = download_file(f"{prefix}-labels-idx1-ubyte.gz", data_dir)
    images = parse_idx(gzip.decompress(image_file.read_bytes()))
    labels = parse_idx(gzip.decompress(label_file.read_bytes()))
    expected = 60000 if split == "train" else 10000
    if images.shape != (expected, 28, 28) or labels.shape != (expected,):
        raise ValueError("Unexpected official partition size")
    if not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError("Expected all ten intact integer labels")
    return images, labels


def stratified_subset(ids, labels, size, seed=42):
    ids = np.asarray(ids)
    if size == len(ids):
        return ids.copy()
    if not 0 < size < len(ids):
        raise ValueError("Subset size outside partition")
    selected, _ = train_test_split(ids, train_size=size, random_state=seed, stratify=labels[ids])
    return selected


def saved_split(labels, directory=None):
    directory = Path(directory) if directory else ROOT / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "split_indices.npz"
    ids = np.arange(len(labels))
    expected_train, expected_val = train_test_split(ids, test_size=0.2, random_state=42, stratify=labels)
    if path.exists():
        with np.load(path) as data:
            train, val = data["train_idx"], data["val_idx"]
        if not np.array_equal(train, expected_train) or not np.array_equal(val, expected_val):
            raise ValueError("Saved split differs from declared seed/protocol; do not overwrite silently")
    else:
        train, val = expected_train, expected_val
        np.savez(path, train_idx=train, val_idx=val)
    if np.intersect1d(train, val).size or not np.array_equal(np.sort(np.r_[train, val]), ids):
        raise ValueError("Invalid split coverage or overlap")
    return train, val, path


def audit_training(images, labels, train, val):
    groups = defaultdict(list)
    for index, image in enumerate(images):
        groups[hashlib.sha256(image.tobytes()).digest()].append(index)
    duplicates = [g for g in groups.values() if len(g) > 1]
    train_set, val_set = set(train.tolist()), set(val.tolist())
    cross = [g for g in duplicates if any(i in train_set for i in g) and any(i in val_set for i in g)]
    conflicts = [g for g in duplicates if len(set(labels[g].tolist())) > 1]
    return {
        "shape": list(images.shape), "dtype": str(images.dtype),
        "pixel_min": int(images.min()), "pixel_max": int(images.max()),
        "finite": bool(np.isfinite(images).all()),
        "all_zero_images": int((images.max(axis=(1, 2)) == 0).sum()),
        "class_counts": np.bincount(labels, minlength=10).tolist(),
        "train_counts": np.bincount(labels[train], minlength=10).tolist(),
        "val_counts": np.bincount(labels[val], minlength=10).tolist(),
        "exact_duplicate_groups": len(duplicates),
        "extra_duplicate_images": sum(len(g) - 1 for g in duplicates),
        "conflicting_label_duplicate_groups": len(conflicts),
        "fit_val_duplicate_groups": len(cross),
        "cross_duplicate_examples": cross[:10],
        "decision": "Retain official examples and seeded split; disclose exact-duplicate overlap. No near-duplicate or semantic relabeling audit claimed.",
        "test_content_explored_during_eda": False,
    }


def prepare():
    images, labels = load_official("train")
    train, val, path = saved_split(labels)
    write_json(ROOT / "data/training_audit.json", audit_training(images, labels, train, val))
    directory = ROOT / "data"
    for profile, n_train, n_val in [("smoke", 2000, 1000), ("budget", 12000, 12000), ("full", 48000, 12000)]:
        selected_train = stratified_subset(train, labels, n_train)
        selected_val = stratified_subset(val, labels, n_val)
        target = directory / f"{profile}_indices.npz"
        if target.exists():
            with np.load(target) as old:
                if not np.array_equal(old["train_idx"], selected_train) or not np.array_equal(old["val_idx"], selected_val):
                    raise ValueError(f"Existing {profile} split is inconsistent")
        else:
            np.savez(target, train_idx=selected_train, val_idx=selected_val)
    write_json(directory / "provenance.json", {
        "source": "https://github.com/zalandoresearch/fashion-mnist", "download_base": BASE_URL,
        "published_md5": FILES, "seed": 42, "validation_fraction": 0.2,
        "split_sha256": sha256(path), "sample_id_rule": "source_split plus original row index",
    })
    return images, labels

