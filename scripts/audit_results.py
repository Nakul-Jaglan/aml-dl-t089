"""Regenerate metrics and verify the frozen evidence before submission."""
from pathlib import Path

import numpy as np
import pandas as pd

from clothing_ml.data import load_official
from clothing_ml.evaluate import label_metrics, validate_predictions
from clothing_ml.experiments import validate_manifest
from clothing_ml.utils import ROOT, read_json, sha256, write_json

result = ROOT / "results/full"
manifest = read_json(result / "freeze.json")
validate_manifest(manifest)
summary = pd.read_csv(result / "test_summary.csv")
if len(summary) != 8 or summary.train_n.nunique() != 1 or int(summary.train_n.iloc[0]) != 48000:
    raise ValueError("Expected eight comparable full-profile result rows")
training_images, training_labels = load_official("train")
test_images, test_labels = load_official("test")
with np.load(ROOT / "data/full_indices.npz") as indices:
    val_ids = indices["val_idx"]
checks = []
for row in manifest["finalists"]:
    for stage, ids, labels in [("val", val_ids, training_labels), ("test", np.arange(10000), test_labels)]:
        path = result / "predictions" / f"{row['run_id']}_{stage}.csv.gz"
        frame = validate_predictions(pd.read_csv(path), ids, labels, stage, row["run_id"])
        metrics = label_metrics(frame.y_true, frame.y_pred)
        expected = row["metrics"] if stage == "val" else read_json(result / f"{row['run_id']}_test_metrics.json")
        for key in ["accuracy", "macro_f1", "recall", "support", "confusion"]:
            if not np.allclose(metrics[key], expected[key], rtol=0, atol=1e-12):
                raise ValueError(f"Metric mismatch for {row['run_id']} {stage} {key}")
        checks.append({"run_id": row["run_id"], "stage": stage, "rows": len(frame), "predictions_sha256": sha256(path)})
    model = ROOT / row["artifact_path"]
    if model.exists() and sha256(model) != row["artifact_sha256"]:
        raise ValueError("Local trained model differs from frozen artifact")
receipt = read_json(result / "test_receipt.json")
if receipt["freeze_manifest_sha256"] != manifest["manifest_sha256"] or receipt["test_summary_sha256"] != sha256(result / "test_summary.csv"):
    raise ValueError("Test receipt does not match frozen manifest and saved summary")
recall = pd.read_csv(result / "per_class_recall.csv")
if len(recall) != 160:
    raise ValueError("Need all ten class recalls for eight models and two stages")
write_json(result / "evidence_audit.json", {"status": "passed", "checks": checks,
                                           "finalists": 8, "train_n_each": 48000,
                                           "validation_selected_run_id": manifest["selected_run_id"],
                                           "test_used_for_selection": False})
print("Verified 16 prediction files, all metrics, original IDs, available model hashes and test receipt.")
