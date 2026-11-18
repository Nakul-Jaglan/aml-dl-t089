"""Read frozen Phase 1/2 evidence and build Nakul's integration artifacts.

The audit below independently derives metrics from confusion counts. It reads all
full-profile predictions, including the later validation-only augmentation run.
It does not fit models, rewrite freezes, or change the existing evaluation files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import CLASS_NAMES, load_official
from ..utils import ROOT, read_json, sha256, stable_hash, write_json

FULL = ROOT / "results/phase2/full"
INTEGRATION = ROOT / "results/phase2/integration"
FIGURES = ROOT / "figures/phase2/integration"
CNN_ID = "full_cnn3_s42_80a2edee"
SVM_ID = "full_hog_svm_3fd9d87a"
ROW_NAMES = {"selected_cnn": "CNN-3", "hog_svm": "HOG + SVM",
             "emb_lr": "Embedding + LR", "hog_lr": "HOG + LR", "fusion_lr": "Fusion + LR"}


def metrics_from_counts(true, predicted):
    """Derive ten-class metrics using integer confusion counts, independent of sklearn."""
    matrix = np.bincount(true * 10 + predicted, minlength=100).reshape(10, 10)
    true_n, predicted_n = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    recall = np.divide(diagonal, true_n, out=np.zeros(10), where=true_n != 0)
    precision = np.divide(diagonal, predicted_n, out=np.zeros(10), where=predicted_n != 0)
    f1 = np.divide(2 * diagonal, true_n + predicted_n, out=np.zeros(10), where=(true_n + predicted_n) != 0)
    return {"accuracy": float(diagonal.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "recall": recall, "precision": precision, "f1": f1, "support": true_n, "confusion": matrix}


def check_manifest(manifest):
    contents = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if stable_hash(contents) != manifest["manifest_sha256"]:
        raise ValueError("Freeze manifest self-hash mismatch")


def reviewed_evidence():
    """Return source-backed frames after checking labels, membership, metrics and hashes."""
    _, train_labels = load_official("train")
    _, test_labels = load_official("test")
    with np.load(ROOT / "data/full_indices.npz") as split:
        fit_ids, val_ids = split["train_idx"], split["val_idx"]
    assert len(fit_ids) == 48000 and len(val_ids) == 12000
    assert np.array_equal(np.sort(np.r_[fit_ids, val_ids]), np.arange(60000))
    assert not np.intersect1d(fit_ids, val_ids).size
    freeze = read_json(FULL / "freeze.json")
    phase1_freeze = read_json(ROOT / "results/full/freeze.json")
    check_manifest(freeze)
    check_manifest(phase1_freeze)
    assert freeze["phase1_reference"]["freeze_manifest_sha256"] == phase1_freeze["manifest_sha256"]
    assert freeze["split_sha256"] == sha256(ROOT / "data/full_indices.npz")
    for row in freeze["test_rows"]:
        if sha256(ROOT / row["artifact_path"]) != row["artifact_sha256"]:
            raise ValueError(f"Frozen artifact changed: {row['run_id']}")

    fusion_candidates = pd.read_csv(FULL / "fusion/fusion_candidates.csv").set_index("run_id")
    sources = []
    for path in sorted((ROOT / "results/full/predictions").glob("*.csv.gz")):
        run_id, stage = path.name.removesuffix(".csv.gz").rsplit("_", 1)
        reference = read_json(ROOT / "results/full/records" / f"{run_id}.json")["metrics"] if stage == "val" else read_json(ROOT / "results/full" / f"{run_id}_test_metrics.json")
        sources.append((path, run_id, stage, reference, None))
    cnn_records = []
    for record_path in sorted((FULL / "runs").glob("*/record.json")):
        record = read_json(record_path)
        if record["status"] != "completed":
            continue
        run_id = record["run_id"]
        assert record["fit_n"] == 48000 and record["val_n"] == 12000
        assert record["split_sha256"] == freeze["split_sha256"]
        path = record_path.parent / "val_predictions.csv.gz"
        sources.append((path, run_id, "val", record["val_metrics"], record["val_predictions_sha256"]))
        if sha256(ROOT / record["checkpoint_path"]) != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch: {run_id}")
        cnn_records.append(record)
    for path in sorted((FULL / "fusion/predictions").glob("*_val.csv.gz")):
        run_id = path.name.removesuffix("_val.csv.gz")
        row = fusion_candidates.loc[run_id]
        reference = {"accuracy": float(row.val_accuracy), "macro_f1": float(row.val_macro_f1)}
        sources.append((path, run_id, "val", reference, row.predictions_sha256))
    for row in freeze["test_rows"]:
        path = FULL / "predictions" / f"{row['run_id']}_test.csv.gz"
        reference = read_json(path.with_name(path.name.replace(".csv.gz", "_metrics.json")))
        sources.append((path, row["run_id"], "test", reference, None))

    audit_rows, per_class = [], []
    predictions = {}
    for path, run_id, stage, reference, expected_hash in sources:
        frame = pd.read_csv(path)
        ids = val_ids if stage == "val" else np.arange(10000)
        labels = train_labels if stage == "val" else test_labels
        source_split = "official_train" if stage == "val" else "official_test"
        required = {"run_id", "stage", "source_split", "sample_id", "y_true", "y_pred"}
        assert required.issubset(frame.columns), path
        assert len(frame) == len(ids) and not frame.sample_id.duplicated().any(), path
        assert set(frame.run_id) == {run_id} and set(frame.stage) == {stage}, path
        assert set(frame.source_split) == {source_split}, path
        assert np.array_equal(np.sort(frame.sample_id), np.sort(ids)), path
        aligned = frame.set_index("sample_id").loc[ids]
        true, predicted = aligned.y_true.to_numpy(), aligned.y_pred.to_numpy()
        assert np.array_equal(true, labels[ids]), path
        assert np.isin(predicted, np.arange(10)).all(), path
        values = metrics_from_counts(true, predicted)
        assert np.isclose(values["accuracy"], reference["accuracy"], atol=1e-12, rtol=0), path
        assert np.isclose(values["macro_f1"], reference["macro_f1"], atol=1e-12, rtol=0), path
        if "confusion" in reference:
            assert np.array_equal(values["confusion"], reference["confusion"]), path
        if expected_hash:
            assert sha256(path) == expected_hash, path
        audit_rows.append({"run_id": run_id, "stage": stage, "n": len(ids),
                           "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                           "shirt_recall": float(values["recall"][6]),
                           "source": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        for label, name in enumerate(CLASS_NAMES):
            per_class.append({"run_id": run_id, "stage": stage, "label": label, "category": name,
                              "recall": float(values["recall"][label]), "precision": float(values["precision"][label]),
                              "f1": float(values["f1"][label]), "support": int(values["support"][label])})
        predictions[run_id, stage] = aligned.reset_index()

    metrics = pd.DataFrame(audit_rows)
    test_scores = metrics[metrics.stage == "test"].set_index("run_id")
    val_scores = metrics[metrics.stage == "val"].set_index("run_id")
    cnn_table = pd.DataFrame([{
        "run_id": record["run_id"], "name": record["name"], "run_order": record["config"]["run_order"],
        "dropout": record["config"]["dropout"], "augmentation": record["config"]["augmentation"],
        "parameters": record["parameters"], "best_epoch": record["best_epoch"], "stop_epoch": record["stop_epoch"],
        "val_macro_f1": record["val_macro_f1"], "val_accuracy": record["val_accuracy"],
        "shirt_recall": float(record["val_metrics"]["recall"][6]),
        "selectable": record["config"]["run_order"] in [1, 2, 3, 4],
    } for record in cnn_records]).sort_values("run_order")
    selected = cnn_table[cnn_table.selectable].sort_values(
        ["val_macro_f1", "val_accuracy", "run_order"], ascending=[False, False, True]).iloc[0]
    assert selected.run_id == freeze["test_rows"][0]["run_id"] == CNN_ID
    assert not any("p50" in row["run_id"] for row in freeze["test_rows"])

    timing = read_json(FULL / "evaluation/timing.json")
    timing_by_row = timing["measurements"]
    comparison = []
    rows = [{"row": "hog_svm", "run_id": SVM_ID}] + freeze["test_rows"]
    for row in rows:
        run_id = row["run_id"]
        source = test_scores.loc[run_id]
        timed = timing_by_row[row["row"]]["batch128_amortized"]
        comparison.append({"run_id": run_id, "row": row["row"], "model": ROW_NAMES[row["row"]],
                           "val_macro_f1": float(val_scores.loc[run_id].macro_f1),
                           "test_accuracy": float(source.accuracy), "test_macro_f1": float(source.macro_f1),
                           "shirt_recall": float(source.shirt_recall),
                           "ms_per_image": timed["median_ms_per_image"]})
    return {"freeze": freeze, "metrics": metrics, "per_class": pd.DataFrame(per_class),
            "comparison": pd.DataFrame(comparison), "cnn_candidates": cnn_table,
            "fusion_candidates": fusion_candidates.reset_index(), "predictions": predictions,
            "cnn_records": {record["run_id"]: record for record in cnn_records},
            "timing": timing, "audit_count": len(audit_rows)}


def save_integration_tables(evidence):
    INTEGRATION.mkdir(parents=True, exist_ok=True)
    for key in ["metrics", "per_class", "comparison", "cnn_candidates"]:
        evidence[key].to_csv(INTEGRATION / f"{key}.csv", index=False)
    write_json(INTEGRATION / "audit.json", {
        "integration_date": "2026-11-19", "all_passed": True, "full_profile_prediction_files": evidence["audit_count"],
        "method": "independent integer confusion counts and classwise F1, authoritative IDX labels, exact IDs and saved artifact hashes",
        "freeze_sha256": evidence["freeze"]["manifest_sha256"],
        "scope": "31 files including all six fusion C candidates and the later validation-only p50 run; original results are read-only",
        "selected_cnn": CNN_ID, "fit_n": 48000, "val_n": 12000, "test_n": 10000})


def save_figure(figure, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=160, bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")


def comparison_plot(evidence):
    import matplotlib.pyplot as plt
    frame = evidence["comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#285F99", "#087F82", "#536671", "#9C824C", "#B86C16"]
    for axis, column, title in zip(axes, ["test_accuracy", "test_macro_f1"], ["Test accuracy", "Test macro F1"]):
        bars = axis.barh(frame.model, frame[column], color=colors)
        axis.bar_label(bars, fmt="%.4f", padding=5, fontsize=9)
        axis.invert_yaxis()
        axis.set_xlim(0, 1.06)
        axis.set_xlabel(title + " (fraction)")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen official-test comparison: 10,000 images, identical IDs")
    fig.tight_layout()
    return fig


def learning_plot(evidence):
    import matplotlib.pyplot as plt
    history = pd.DataFrame(evidence["cnn_records"][CNN_ID]["history"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(history.epoch, history.fit_loss, label="Fit", color="#285F99")
    axes[0].plot(history.epoch, history.val_loss, label="Validation", color="#B86C16", linestyle="--")
    axes[0].set_ylabel("Mean cross-entropy")
    axes[1].plot(history.epoch, history.val_macro_f1, color="#087F82", label="Validation macro F1")
    axes[1].set_ylabel("Macro F1 (fraction)")
    for axis in axes:
        axis.axvline(22, color="#536671", linestyle=":", label="Best epoch 22")
        axis.set_xlabel("Epoch")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Selected CNN-3: best validation F1 at epoch 22, cap at epoch 25")
    fig.tight_layout()
    return fig


def ablation_plot(evidence):
    import matplotlib.pyplot as plt
    frame = evidence["cnn_candidates"]
    names = ["CNN-2", "No dropout", "Always shift", "CNN-3", "50% shift\n(later ablation)"]
    reference = float(frame[frame.name == "cnn2"].val_macro_f1.iloc[0])
    delta = 100 * (frame.val_macro_f1.to_numpy() - reference)
    fig, axis = plt.subplots(figsize=(9, 3.9))
    bars = axis.barh(names, delta, color=["#536671", "#B86C16", "#B86C16", "#087F82", "#536671"])
    axis.bar_label(bars, labels=[f"{v:+.3f}" for v in delta], padding=5)
    axis.axvline(0, color="#233745", linewidth=.8)
    axis.set_xlim(-.82, .65)
    axis.invert_yaxis()
    axis.set_xlabel("Validation macro F1 difference from CNN-2 (percentage points)")
    axis.set_title("One-seed ablations, same 48,000 fit / 12,000 validation IDs")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def recall_plot(evidence):
    import matplotlib.pyplot as plt
    data = evidence["per_class"]
    cnn = data[(data.run_id == CNN_ID) & (data.stage == "test")].sort_values("label")
    svm = data[(data.run_id == SVM_ID) & (data.stage == "test")].sort_values("label")
    fig, axis = plt.subplots(figsize=(10, 4))
    pos = np.arange(10)
    axis.bar(pos-.18, 100*svm.recall, width=.36, label="HOG + SVM", color="#285F99")
    axis.bar(pos+.18, 100*cnn.recall, width=.36, label="CNN-3", color="#087F82", hatch="//")
    axis.set_xticks(pos, ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"], rotation=35, ha="right")
    axis.set_ylabel("Recall (%)")
    axis.set_ylim(0, 113)
    axis.legend(ncol=2, frameon=False, loc="upper center")
    axis.set_title("Official-test per-class recall: 1,000 images per true category")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def hybrid_plot(evidence):
    import matplotlib.pyplot as plt
    frame = evidence["comparison"].iloc[[1, 2, 3, 4]]
    fig, axis = plt.subplots(figsize=(9, 3.8))
    pos = np.arange(4)
    axis.barh(pos-.16, frame.val_macro_f1, height=.3, label="Validation", color="#536671")
    axis.barh(pos+.16, frame.test_macro_f1, height=.3, label="Official test", color="#087F82", hatch="//")
    axis.set_yticks(pos, frame.model)
    axis.set_xlim(0, 1.05)
    axis.invert_yaxis()
    axis.set_xlabel("Macro F1 (fraction)")
    fig.legend(frameon=False, ncol=2, loc="lower center")
    axis.set_title("Fusion controls: adding HOG lowers validation F1 vs embedding-only LR")
    fig.tight_layout(rect=(0, .12, 1, 1))
    return fig


def error_plot(evidence):
    import matplotlib.pyplot as plt
    images, _ = load_official("test")
    examples = pd.read_csv(FULL / "evaluation/figure_example_ids.csv")
    groups = ["Only CNN correct", "Only HOG + SVM correct", "Both wrong"]
    rows = [examples[(examples.panel == "upper_body") & (examples.group == group)].iloc[0] for group in groups]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    for axis, row in zip(axes, rows):
        axis.imshow(images[int(row.sample_id)], cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{row.group}\nID {row.sample_id}, true {row.true_category}", fontsize=10)
        axis.set_xlabel(f"CNN: {row.cnn_category}\nSVM: {row.svm_category}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("First sorted test ID per upper-body outcome group")
    fig.tight_layout(rect=(0, 0, 1, .87))
    return fig


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    evidence = reviewed_evidence()
    save_integration_tables(evidence)
    for function, stem in [(comparison_plot, "comparison"), (learning_plot, "learning"),
                           (ablation_plot, "ablations"), (recall_plot, "recall"),
                           (hybrid_plot, "hybrid"), (error_plot, "errors")]:
        figure = function(evidence)
        save_figure(figure, stem)
        plt.close(figure)
    print(f"Reviewed {evidence['audit_count']} full-profile prediction files; wrote integration tables and six figures.")


if __name__ == "__main__":
    build()
