"""Fixed per-image features. Learned standardization belongs in model Pipelines."""
from __future__ import annotations

import numpy as np
from skimage.feature import hog

from .utils import ROOT, read_json, sha256, stable_hash, write_json

HOG_SETTINGS = dict(orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                    block_norm="L2-Hys", transform_sqrt=False, feature_vector=True, channel_axis=None)


def normalized_images(raw):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (28, 28):
        raise ValueError("Expected raw image batch shape (N, 28, 28)")
    if raw.dtype != np.uint8:
        raise TypeError("Expected raw uint8 images; avoid double normalization")
    return raw.astype(np.float32) / 255.0


def make_features(raw, feature_name):
    images = normalized_images(raw)
    if feature_name == "pixels":
        features = images.reshape(len(images), 784)
    elif feature_name == "hog":
        features = np.asarray([hog(image, **HOG_SETTINGS) for image in images], dtype=np.float32)
    else:
        raise ValueError(f"Unknown feature representation {feature_name}")
    expected = 784 if feature_name == "pixels" else 1296
    if features.shape != (len(images), expected) or not np.isfinite(features).all():
        raise ValueError("Invalid feature matrix")
    return features


def cached_features(raw, ids, source_split, feature_name):
    """Content-addressed cache includes raw image bytes, IDs, settings and version."""
    from importlib.metadata import version
    import hashlib
    key = stable_hash({"ids": np.asarray(ids).tolist(), "source": source_split,
                       "feature": feature_name, "hog": HOG_SETTINGS,
                       "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                       "skimage": version("scikit-image"), "implementation": 1})
    directory = ROOT / "data/cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{feature_name}_{key[:20]}.npy"
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata["key"] != key or metadata["file_sha256"] != sha256(path):
            raise ValueError("Feature cache integrity mismatch")
        return np.load(path), key
    features = make_features(raw, feature_name)
    np.save(path, features)
    write_json(metadata_path, {"key": key, "ids": ids, "source_split": source_split,
                              "feature": feature_name, "shape": features.shape,
                              "file_sha256": sha256(path), "hog_settings": HOG_SETTINGS})
    return features, key

