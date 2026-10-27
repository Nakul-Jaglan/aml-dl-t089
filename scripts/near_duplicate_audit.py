"""Exhaustive near-duplicate audit; the gap training_audit.json explicitly does not claim."""
from __future__ import annotations

import numpy as np

from clothing_ml.data import load_official
from clothing_ml.near_duplicates import audit
from clothing_ml.utils import ROOT, read_json, write_json

images, labels = load_official("train")
test_images, test_labels = load_official("test")
with np.load(ROOT / "data/full_indices.npz") as indices:
    fit_ids, val_ids = indices["train_idx"], indices["val_idx"]

report = {
    "purpose": "Quantify near-duplicate overlap that exact SHA-256 image hashing cannot detect.",
    "relates_to": "data/training_audit.json, which reports zero exact duplicates and claims no near-duplicate audit.",
    "exact_duplicate_groups_reference": read_json(ROOT / "data/training_audit.json")["exact_duplicate_groups"],
    "comparisons": [
        audit("validation versus fit pool",
              images[val_ids], val_ids, labels[val_ids], "official_train",
              images[fit_ids], fit_ids, labels[fit_ids], "official_train"),
        audit("official test versus official training",
              test_images, np.arange(len(test_images)), test_labels, "official_test",
              images, np.arange(len(images)), labels, "official_train"),
    ],
    "limits": [
        "Pixel-space distance only. It does not detect the same product photographed at another angle, scale or exposure, nor semantically repeated catalogue items.",
        "A small RMS difference is evidence of a near-identical frame, not proof of a reused source photograph.",
        "The test comparison is a post-freeze audit. It describes a limitation of the already-reported test numbers and must not be used to reselect a model or re-tune anything.",
    ],
}
write_json(ROOT / "data/near_duplicate_audit.json", report)
for comparison in report["comparisons"]:
    print(f"{comparison['comparison']}: min RMS {comparison['distribution']['min']}, "
          f"below 5 -> {comparison['counts_below']['5.0']} of {comparison['n_query']} "
          f"({comparison['fraction_below_rms_5_percent']}%)")
print("Wrote data/near_duplicate_audit.json")
