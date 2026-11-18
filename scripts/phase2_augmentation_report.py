"""Augmentation ablation summary (Sudeshhna): validation only, same seed, split and settings.

    python scripts/phase2_augmentation_report.py

Compares the unaugmented CNN-2 reference with both translation policies from their
saved run records. Writes results/phase2/data_audit/augmentation_ablation.{csv,json}
and figures/phase2/data/augmentation_curves.png/.pdf. Nothing is retrained and
official test data is not loaded.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clothing_ml.data import CLASS_NAMES
from clothing_ml.plots import save_figure, setup_style
from clothing_ml.utils import ROOT, read_json, sha256, write_json

RUNS = ROOT / "results/phase2/full/runs"
OUT = ROOT / "results/phase2/data_audit"
ROWS = [("full_cnn2_s42_3617f59e", "no shift (reference)", "#243446"),
        ("full_cnn2_aug_s42_fdeb788e", "shift every image (translate2)", "#c0392b"),
        ("full_cnn2_aug_p50_s42_fe3c3cdd", "shift with p = 0.5 (translate2_p50)", "#1a7f72")]
CONTROLLED = ["blocks", "dropout", "seed", "batch_size", "max_epochs", "patience", "optimizer", "loss", "profile"]


def paired(reference, other):
    """Per-image comparison of two validation prediction files joined by original ID."""
    a = pd.read_csv(RUNS / reference / "val_predictions.csv.gz").set_index("sample_id")
    b = pd.read_csv(RUNS / other / "val_predictions.csv.gz").set_index("sample_id")
    assert a.index.equals(b.index) and (a.y_true == b.y_true).all()
    ra, rb = a.y_pred == a.y_true, b.y_pred == b.y_true
    shirt = a.y_true == 6
    return {"only_reference_correct": int((ra & ~rb).sum()), "only_variant_correct": int((~ra & rb).sum()),
            "both_wrong": int((~ra & ~rb).sum()),
            "shirt_only_reference_correct": int((ra & ~rb & shirt).sum()),
            "shirt_only_variant_correct": int((~ra & rb & shirt).sum()),
            "shirt_predicted_as": {CLASS_NAMES[k]: int(v) for k, v in b[shirt & ~rb].y_pred.value_counts().items()}}


def main():
    setup_style()
    records = [read_json(RUNS / run_id / "record.json") for run_id, *_ in ROWS]
    for r in records:
        assert r["status"] == "completed", r["run_id"]
        assert sha256(ROOT / r["checkpoint_path"]) == r["checkpoint_sha256"], r["run_id"]
    reference = records[0]
    for r in records[1:]:
        for key in ["split_sha256", "fit_ids_sha256", "val_ids_sha256"]:
            assert r[key] == reference[key], (r["run_id"], key)
        for key in CONTROLLED:
            assert r["config"][key] == reference["config"][key], (r["run_id"], key)

    rows = []
    for (run_id, label, _), r in zip(ROWS, records):
        recall = r["val_metrics"]["recall"]
        history = pd.DataFrame(r["history"])
        last = history.iloc[-1]
        rows.append({"run_id": run_id, "policy": r["config"]["augmentation"], "description": label,
                     "val_macro_f1": r["val_macro_f1"], "val_accuracy": r["val_accuracy"],
                     "recall_shirt": recall[6], "recall_tshirt": recall[0], "recall_pullover": recall[2],
                     "recall_coat": recall[4], "best_epoch": r["best_epoch"], "stop_epoch": r["stop_epoch"],
                     "stopped_early": r["stopped_early"], "train_seconds": round(r["train_seconds"], 1),
                     "final_fit_loss": last.fit_loss, "final_val_loss": last.val_loss,
                     "min_val_loss": history.val_loss.min(), "min_val_loss_epoch": int(history.val_loss.idxmin()) + 1})
    table = pd.DataFrame(rows)
    ref = table.iloc[0]
    for column in ["val_macro_f1", "val_accuracy", "recall_shirt", "recall_tshirt", "recall_pullover"]:
        table[f"delta_{column}"] = table[column] - ref[column]
    table.to_csv(OUT / "augmentation_ablation.csv", index=False)
    write_json(OUT / "augmentation_ablation.json", {
        "stage": "validation_only", "official_test_loaded": False,
        "controlled_settings_identical": CONTROLLED + ["split_sha256", "fit_ids_sha256", "val_ids_sha256"],
        "rows": table.to_dict(orient="records"),
        "paired_vs_reference": {run_id: paired(ROWS[0][0], run_id) for run_id, *_ in ROWS[1:]},
        "note": ("translate2_p50 was run after the selection and freeze were written; it is a validation-only "
                 "ablation row and is not a selection candidate.")})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for (run_id, label, color), r in zip(ROWS, records):
        h = pd.DataFrame(r["history"])
        axes[0].plot(h.epoch, h.fit_loss, color=color, linestyle="--", linewidth=1.2)
        axes[0].plot(h.epoch, h.val_loss, color=color, linewidth=1.6, label=label)
        axes[1].plot(h.epoch, h.val_macro_f1, color=color, linewidth=1.6, label=label)
        axes[1].scatter([r["best_epoch"]], [r["val_macro_f1"]], color=color, zorder=3, s=22)
        recall = np.array(r["val_metrics"]["recall"])
        axes[2].plot(range(10), recall, marker="o", markersize=3.5, color=color, linewidth=1.2, label=label)
    axes[0].set(title="Loss (dashed: fit, solid: validation)", xlabel="epoch", ylabel="cross-entropy")
    axes[1].set(title="Validation macro F1 (dot: kept checkpoint)", xlabel="epoch", ylabel="macro F1", ylim=(0.88, 0.935))
    axes[2].set(title="Validation recall at kept checkpoint", ylabel="recall", ylim=(0.6, 1.005))
    axes[2].set_xticks(range(10), CLASS_NAMES, rotation=55, ha="right", fontsize=8)
    axes[1].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("CNN-2 augmentation ablation, validation only (seed 42, 48,000 fit / 12,000 validation IDs)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, ROOT / "figures/phase2/data", "augmentation_curves")
    print(table[["policy", "val_macro_f1", "val_accuracy", "recall_shirt", "best_epoch", "stop_epoch",
                 "delta_val_macro_f1", "delta_recall_shirt"]].to_string(index=False))


if __name__ == "__main__":
    main()
