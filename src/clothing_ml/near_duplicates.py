"""Near-duplicate detection; the overlap exact SHA-256 image hashing cannot see."""
from __future__ import annotations

import numpy as np

from .data import CLASS_NAMES

CHUNK = 512
EXACT_RMS_CUTOFF = 15.0
THRESHOLDS = [1.0, 2.0, 5.0, 10.0]
PIXELS = 784


def nearest_neighbour(query, reference, chunk=CHUNK):
    """Smallest Euclidean distance from every query row to any reference row.

    Returned distances are divided by sqrt(PIXELS), so they read as a
    root-mean-square pixel difference on the raw 0-255 scale.

    float64 is deliberate. The expanded form |a|^2 + |b|^2 - 2ab cancels
    catastrophically in float32 for 784-dimensional 0-255 rows: squared norms
    reach ~5e7, so an identical pair can report an RMS difference above 0.2
    and, worse, argmin can select the wrong neighbour among close candidates.
    That is the regime this audit exists to measure.
    """
    query = np.asarray(query).reshape(len(query), -1).astype(np.float64)
    reference = np.asarray(reference).reshape(len(reference), -1).astype(np.float64)
    if query.shape[1] != reference.shape[1]:
        raise ValueError("Query and reference rows must have the same length")
    reference_sq = (reference ** 2).sum(axis=1)
    best = np.empty(len(query), dtype=np.float64)
    index = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        squared = (block ** 2).sum(axis=1)[:, None] + reference_sq[None, :] - 2.0 * (block @ reference.T)
        np.maximum(squared, 0, out=squared)
        column = squared.argmin(axis=1)
        best[start:start + chunk] = np.sqrt(squared[np.arange(len(block)), column])
        index[start:start + chunk] = column
    return best / np.sqrt(query.shape[1]), index


def exact_rms(left, right):
    """Integer-exact RMS pixel difference, free of float32 accumulation error."""
    difference = np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64)
    return float(np.sqrt((difference ** 2).sum() / difference.size))


def describe(left, right):
    difference = np.abs(np.asarray(left, dtype=np.int32) - np.asarray(right, dtype=np.int32))
    return {"differing_pixels": int((difference > 0).sum()),
            "max_absolute_difference": int(difference.max()),
            "mean_absolute_difference": round(float(difference.mean()), 4)}


def audit(name, query_images, query_ids, query_labels, query_split,
          reference_images, reference_ids, reference_labels, reference_split, top=20):
    """Exhaustive nearest-neighbour comparison of two partitions."""
    approximate, column = nearest_neighbour(query_images, reference_images)

    # float32 is only a filter: recompute every candidate below the cutoff exactly.
    rms = np.asarray(approximate, dtype=np.float64)
    for row in np.flatnonzero(approximate < EXACT_RMS_CUTOFF):
        rms[row] = exact_rms(query_images[row], reference_images[column[row]])

    shares_label = np.asarray(query_labels) == np.asarray(reference_labels)[column]
    order = np.argsort(rms)
    pairs = [{
        "query": f"{query_split}:{int(query_ids[row])}",
        "query_class": CLASS_NAMES[int(query_labels[row])],
        "neighbour": f"{reference_split}:{int(reference_ids[column[row]])}",
        "neighbour_class": CLASS_NAMES[int(reference_labels[column[row]])],
        "same_class": bool(shares_label[row]),
        "rms_pixel_difference": round(float(rms[row]), 4),
        **describe(query_images[row], reference_images[column[row]]),
    } for row in order[:top]]

    return {
        "comparison": name,
        "query_split": query_split, "n_query": int(len(query_ids)),
        "reference_split": reference_split, "n_reference": int(len(reference_ids)),
        "exhaustive": True,
        "metric": "root-mean-square pixel difference on the raw 0-255 scale",
        "exact_below_rms": EXACT_RMS_CUTOFF,
        "distribution": {
            "min": round(float(rms.min()), 4),
            "p01": round(float(np.percentile(rms, 1)), 4),
            "p05": round(float(np.percentile(rms, 5)), 4),
            "median": round(float(np.median(rms)), 4),
            "mean": round(float(rms.mean()), 4),
            "max": round(float(rms.max()), 4),
        },
        "counts_below": {str(t): int((rms < t).sum()) for t in THRESHOLDS},
        "fraction_below_rms_5_percent": round(100 * float((rms < 5.0).mean()), 4),
        "nearest_neighbour_shares_label_percent": round(100 * float(shares_label.mean()), 2),
        "closest_pairs": pairs,
    }
