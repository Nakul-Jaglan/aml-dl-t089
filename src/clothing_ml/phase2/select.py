"""Validation-only CNN selection, candidate table, freeze receipt and the separate test command."""
from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch

from ..data import CLASS_NAMES, load_official
from ..evaluate import label_metrics, prediction_frame, recall_rows, validate_predictions
from ..features import cached_features
from ..utils import ROOT, RESULTS_ROOT, read_json, sha256, stable_hash, write_json
from .runner import load_checkpoint, phase2_environment, phase2_paths

PRIMARY_SEED = 42
SELECTION_RULE = ("highest validation macro F1, then validation accuracy, then declared run_order "
                  "(cnn2=1, cnn2_p0=2, cnn2_aug=3, cnn3=4); primary seed 42 only")
PHASE1_REFERENCE = RESULTS_ROOT / "full" / "freeze.json"


def completed_runs(profile):
    result, _ = phase2_paths(profile)
    return [r for r in (read_json(p) for p in sorted((result / "runs").glob("*/record.json"))) if r["status"] == "completed"]


def candidate_table(runs):
    rows = []
    for r in runs:
        recall = r["val_metrics"]["recall"]
        rows.append({"run_id": r["run_id"], "name": r["name"], "run_order": r["config"]["run_order"],
                     "seed": r["config"]["seed"], "blocks": r["config"]["blocks"], "dropout": r["config"]["dropout"],
                     "augmentation": r["config"]["augmentation"], "parameters": r["parameters"],
                     "val_macro_f1": r["val_macro_f1"], "val_accuracy": r["val_accuracy"],
                     "best_epoch": r["best_epoch"], "stop_epoch": r["stop_epoch"],
                     "train_seconds": round(r["train_seconds"], 1),
                     "recall_tshirt": recall[0], "recall_pullover": recall[2], "recall_coat": recall[4],
                     "recall_shirt": recall[6], "config_sha256": r["config_sha256"][:12],
                     "checkpoint_sha256": r["checkpoint_sha256"][:12]})
    return pd.DataFrame(rows).sort_values(["seed", "run_order"]).reset_index(drop=True)


def ablation_deltas(table):
    """Validation-only contrasts against the cnn2 reference (same seed, same IDs)."""
    rows = []
    for seed, group in table.groupby("seed"):
        reference = group[group.name == "cnn2"]
        if reference.empty:
            continue
        ref = reference.iloc[0]
        for _, row in group[group.name != "cnn2"].iterrows():
            rows.append({"seed": seed, "comparison": f"{row['name']} vs cnn2",
                         "delta_val_macro_f1": row.val_macro_f1 - ref.val_macro_f1,
                         "delta_val_accuracy": row.val_accuracy - ref.val_accuracy,
                         "delta_recall_shirt": row.recall_shirt - ref.recall_shirt,
                         "delta_recall_tshirt": row.recall_tshirt - ref.recall_tshirt,
                         "delta_recall_pullover": row.recall_pullover - ref.recall_pullover,
                         "best_epoch": f"{ref.best_epoch} -> {row.best_epoch}",
                         "stop_epoch": f"{ref.stop_epoch} -> {row.stop_epoch}",
                         "parameters": f"{ref.parameters} -> {row.parameters}"})
    return pd.DataFrame(rows)


def select_cnn(profile="full"):
    if profile == "smoke":
        raise ValueError("Smoke runs are plumbing checks and cannot be selected")
    result, _ = phase2_paths(profile)
    runs = completed_runs(profile)
    primary = [r for r in runs if r["config"]["seed"] == PRIMARY_SEED and not r["config"].get("max_fit") and not r["config"].get("max_val")]
    if not primary:
        raise ValueError("No completed primary-seed runs")
    for key in ["split_sha256", "fit_ids_sha256", "val_ids_sha256"]:
        if len({r[key] for r in primary}) != 1:
            raise ValueError(f"Controlled rows disagree on {key}")
    for r in primary:
        if sha256(ROOT / r["checkpoint_path"]) != r["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after training: {r['run_id']}")
    table = candidate_table(runs)
    table.to_csv(result / "validation_candidates.csv", index=False)
    ablation_deltas(table).to_csv(result / "ablation_deltas.csv", index=False)
    chosen = sorted(primary, key=lambda r: (-r["val_macro_f1"], -r["val_accuracy"], r["config"]["run_order"]))[0]
    selection = {"selected_run_id": chosen["run_id"], "selection_rule": SELECTION_RULE,
                 "selection_stage": "validation_only", "test_used_for_selection": False,
                 "checkpoint_path": chosen["checkpoint_path"], "checkpoint_sha256": chosen["checkpoint_sha256"],
                 "arch": {"blocks": chosen["config"]["blocks"], "dropout": chosen["config"]["dropout"]},
                 "val_macro_f1": chosen["val_macro_f1"], "val_accuracy": chosen["val_accuracy"],
                 "split_sha256": chosen["split_sha256"], "fit_ids_sha256": chosen["fit_ids_sha256"],
                 "val_ids_sha256": chosen["val_ids_sha256"],
                 "candidates": [{"run_id": r["run_id"], "val_macro_f1": r["val_macro_f1"], "val_accuracy": r["val_accuracy"],
                                 "run_order": r["config"]["run_order"]} for r in primary],
                 "selected_utc": datetime.now(timezone.utc).isoformat()}
    write_json(result / "cnn_selection.json", selection)
    print(table.to_string(index=False))
    print(f"\nSelected CNN (validation only): {chosen['run_id']}  F1={chosen['val_macro_f1']:.4f}")
    return selection


def freeze(profile="full"):
    """Nominate the rows that may reach official test. Written once; never overwritten."""
    result, _ = phase2_paths(profile)
    target = result / "freeze.json"
    if target.exists():
        print(f"Already frozen: {target}")
        return read_json(target)
    selection = read_json(result / "cnn_selection.json")
    if sha256(ROOT / selection["checkpoint_path"]) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after selection")
    phase1 = read_json(PHASE1_REFERENCE)
    rows = [{"row": "selected_cnn", "kind": "cnn", "run_id": selection["selected_run_id"],
             "artifact_path": selection["checkpoint_path"], "artifact_sha256": selection["checkpoint_sha256"]}]
    fusion_path = result / "fusion" / "fusion_selection.json"
    if fusion_path.exists():
        fusion = read_json(fusion_path)
        if fusion["encoder_checkpoint_sha256"] != selection["checkpoint_sha256"]:
            raise ValueError("Fusion used a different encoder than the selected CNN")
        for row in fusion["selected_rows"]:
            if sha256(ROOT / row["artifact_path"]) != row["artifact_sha256"]:
                raise ValueError(f"Fusion artifact changed: {row['row']}")
            rows.append({"row": row["row"], "kind": "lr_head", "run_id": row["run_id"], "C": row["C"],
                         "artifact_path": row["artifact_path"], "artifact_sha256": row["artifact_sha256"]})
    runs = completed_runs(profile)
    manifest = {"profile": profile, "test_rows": rows,
                "phase1_reference": {"run_id": phase1["selected_run_id"], "freeze_manifest_sha256": phase1["manifest_sha256"],
                                     "note": "Phase 1 test predictions already exist; reuse them, do not refit"},
                "validation_only_rows": [r["run_id"] for r in runs if r["run_id"] != selection["selected_run_id"]],
                "selection_rule": selection["selection_rule"], "fusion_rule": read_json(fusion_path)["selection_rule"] if fusion_path.exists() else None,
                "split_sha256": selection["split_sha256"], "fit_ids_sha256": selection["fit_ids_sha256"],
                "val_ids_sha256": selection["val_ids_sha256"], "selection_stage": "validation_only",
                "test_used_for_selection": False, "environment": phase2_environment("cpu"),
                "frozen_utc": datetime.now(timezone.utc).isoformat()}
    manifest["manifest_sha256"] = stable_hash(manifest)
    write_json(target, manifest)
    print(f"Frozen {len(rows)} test rows: {[r['row'] for r in rows]}")
    return manifest


def verify_freeze(manifest):
    if stable_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Freeze manifest integrity mismatch")
    if manifest.get("test_used_for_selection") or manifest.get("selection_stage") != "validation_only":
        raise ValueError("Manifest does not establish validation-only selection")
    for row in manifest["test_rows"]:
        if sha256(ROOT / row["artifact_path"]) != row["artifact_sha256"]:
            raise ValueError(f"Frozen artifact hash mismatch: {row['row']}")
    return manifest["manifest_sha256"]


def final_test(profile="full", allow_test=False):
    """Separate, explicit test evaluation of frozen rows only."""
    if not allow_test:
        raise ValueError("Official test access requires --allow-test after the freeze")
    if profile == "smoke":
        raise ValueError("Smoke runs cannot produce test results")
    result, _ = phase2_paths(profile)
    manifest = read_json(result / "freeze.json")
    expected = verify_freeze(manifest)
    if (result / "test_summary.csv").exists():
        raise ValueError("Phase 2 test already recorded. Reuse saved predictions; do not retune or overwrite.")
    write_json(result / "test_access_receipt.json", {"freeze_manifest_sha256": expected, "rows": [r["row"] for r in manifest["test_rows"]],
                                                     "opened_utc": datetime.now(timezone.utc).isoformat()})
    from .fusion import extract_embeddings, row_matrix
    images, labels = load_official("test")
    ids = np.arange(len(labels))
    cnn_row = manifest["test_rows"][0]
    model, _ = load_checkpoint(ROOT / cnn_row["artifact_path"], "cpu", cnn_row["artifact_sha256"])
    embeddings = extract_embeddings(model, images, ids, labels)
    hog = cached_features(images, ids, "official_test", "hog")[0]
    (result / "predictions").mkdir(exist_ok=True)
    summary, recalls = [], []
    for row in manifest["test_rows"]:
        if row["kind"] == "cnn":
            with torch.inference_mode():
                logits = model.classifier(torch.from_numpy(embeddings))  # eval mode: dropout is identity
            predictions = logits.argmax(dim=1).numpy()
        else:
            predictions = joblib.load(ROOT / row["artifact_path"])["pipeline"].predict(row_matrix(row["row"], embeddings, hog))
        frame = prediction_frame(row["run_id"], "test", ids, labels, predictions)
        validate_predictions(frame, ids, labels, "test", row["run_id"])
        frame.to_csv(result / "predictions" / f"{row['run_id']}_test.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        metrics = label_metrics(labels, predictions)
        write_json(result / "predictions" / f"{row['run_id']}_test_metrics.json", metrics)
        summary.append({"row": row["row"], "run_id": row["run_id"], "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                        "recall_shirt": metrics["recall"][6]})
        recalls.extend(recall_rows(row["run_id"], "test", metrics))
        print(f"Test {row['row']}: accuracy={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} Shirt recall={metrics['recall'][6]:.3f}")
    pd.DataFrame(summary).to_csv(result / "test_summary.csv", index=False)
    pd.DataFrame(recalls).to_csv(result / "test_per_class_recall.csv", index=False)
    write_json(result / "test_receipt.json", {"freeze_manifest_sha256": expected, "test_n": len(ids),
                                              "test_summary_sha256": sha256(result / "test_summary.csv"),
                                              "class_names": CLASS_NAMES, "test_tuned": False})
    return summary
