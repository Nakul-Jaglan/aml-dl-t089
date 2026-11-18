"""Phase 2 evaluation from saved evidence: audit, comparison, paired errors, timing.

Owner: Neelanshu. Every number here is recomputed from committed prediction files,
joined by (source_split, sample_id) against the authoritative official labels. No
model is retuned; the frozen artifacts are only reloaded for timing.

  python -m clothing_ml.phase2.evaluation audit     # validate every prediction file, recompute metrics
  python -m clothing_ml.phase2.evaluation paired    # CNN vs HOG+SVM outcome groups, label pairs, figures
  python -m clothing_ml.phase2.evaluation timing    # same-machine CPU timing + complexity columns
  python -m clothing_ml.phase2.evaluation compare   # one-row-per-run table, per-class table, deltas, bootstrap
  python -m clothing_ml.phase2.evaluation all       # the four steps above, in that order
  python -m clothing_ml.phase2.evaluation lookup --id 42   # live demo: one test ID in both files
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data import CLASS_NAMES, load_official
from ..evaluate import label_metrics, validate_predictions
from ..utils import ROOT, RESULTS_ROOT, read_json, sha256, write_json

PHASE1 = RESULTS_ROOT / "full"
PHASE2 = RESULTS_ROOT / "phase2" / "full"
OUT = PHASE2 / "evaluation"
FIGURES = ROOT / "figures" / "phase2" / "full" / "evaluation"
SVM_ID = "full_hog_svm_3fd9d87a"
SHIRT, TSHIRT, PULLOVER, COAT = 6, 0, 2, 4
UPPER_BODY = [TSHIRT, PULLOVER, COAT, SHIRT]
GROUPS = ["Both correct", "Only CNN correct", "Only HOG + SVM correct", "Both wrong"]
BOOTSTRAP_SEED, BOOTSTRAP_REPEATS = 42, 2000
EXAMPLES_PER_GROUP = 4


# --------------------------------------------------------------------------- helpers

def per_class(metrics):
    """Precision, recall and F1 per class from a saved confusion matrix (rows true, columns predicted)."""
    cm = np.asarray(metrics["confusion"], dtype=float)
    tp, support, predicted = cm.diagonal(), cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(10), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=(precision + recall) != 0)
    return precision, recall, f1, support.astype(int)


def macro_f1_from_pairs(y_true, y_pred):
    cm = np.bincount(y_true * 10 + y_pred, minlength=100).reshape(10, 10).astype(float)
    tp = cm.diagonal()
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # 2TP + FP + FN
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(10), where=denom != 0)))


def outcome_groups(y_true, pred_cnn, pred_svm):
    """Four paired outcome groups on identical IDs; returns counts and a per-ID group label."""
    cnn_ok, svm_ok = pred_cnn == y_true, pred_svm == y_true
    label = np.select([cnn_ok & svm_ok, cnn_ok & ~svm_ok, ~cnn_ok & svm_ok], GROUPS[:3], default=GROUPS[3])
    counts = {g: int((label == g).sum()) for g in GROUPS}
    n = len(y_true)
    if sum(counts.values()) != n:
        raise ValueError("Outcome groups do not partition the IDs")
    # (only CNN - only SVM) / n must equal the accuracy difference exactly (integer identity).
    if counts["Only CNN correct"] - counts["Only HOG + SVM correct"] != int(cnn_ok.sum()) - int(svm_ok.sum()):
        raise ValueError("Discordant counts do not reproduce the accuracy delta")
    return counts, label


def label_pair_counts(y_true, pred, labels=UPPER_BODY):
    """Counts true->predicted among confusable upper-body classes, with true-row support."""
    rows = []
    for t in labels:
        support = int((y_true == t).sum())
        for p in labels:
            if p != t:
                rows.append({"true_label": t, "true_category": CLASS_NAMES[t], "pred_label": p,
                             "pred_category": CLASS_NAMES[p], "count": int(((y_true == t) & (pred == p)).sum()),
                             "true_support": support})
    return rows


def paired_bootstrap(y_true, pred_a, pred_b, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    """Resample aligned ID pairs with replacement; delta = a - b for accuracy and macro F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1 = np.empty(repeats), np.empty(repeats)
    for r in range(repeats):
        idx = rng.integers(0, n, n)
        t, a, b = y_true[idx], pred_a[idx], pred_b[idx]
        acc[r] = (a == t).mean() - (b == t).mean()
        f1[r] = macro_f1_from_pairs(t, a) - macro_f1_from_pairs(t, b)
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"seed": seed, "repeats": repeats, "resampled_unit": "aligned official-test ID pair",
            "accuracy_delta_ci95": q(acc), "macro_f1_delta_ci95": q(f1),
            "scope": "sampling variability of this 10,000-image benchmark only; excludes training-seed "
                     "sensitivity and transfer to real catalogue photos"}


def mcnemar_exact(only_a, only_b):
    from scipy.stats import binomtest
    n = only_a + only_b
    return {"discordant_pairs": n, "only_a": only_a, "only_b": only_b,
            "two_sided_p": float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0,
            "test": "exact McNemar (binomial on discordant pairs)"}


def prediction_files():
    """Every prediction file the comparison uses, with its role, stage and recorded metrics."""
    freeze = read_json(PHASE2 / "freeze.json")
    files = []
    for path in sorted((PHASE1 / "predictions").glob("*.csv.gz")):
        run_id, stage = path.name.rsplit("_", 1)[0], path.name.rsplit("_", 1)[1].split(".")[0]
        record = read_json(PHASE1 / "records" / f"{run_id}.json")
        recorded = record["metrics"] if stage == "val" else read_json(PHASE1 / f"{run_id}_test_metrics.json")
        role = "Phase 1 reference (validation-selected)" if run_id == SVM_ID else "Phase 1 context"
        files.append({"run_id": run_id, "stage": stage, "path": path, "role": role, "recorded": recorded})
    test_rows = {r["run_id"]: r for r in freeze["test_rows"]}
    fusion = pd.read_csv(PHASE2 / "fusion" / "fusion_candidates.csv").set_index("run_id")
    for run_id in list(test_rows) + freeze["validation_only_rows"]:
        if run_id in test_rows and test_rows[run_id]["kind"] == "lr_head":
            val_path = PHASE2 / "fusion" / "predictions" / f"{run_id}_val.csv.gz"
            recorded_val = {"accuracy": float(fusion.loc[run_id, "val_accuracy"]), "macro_f1": float(fusion.loc[run_id, "val_macro_f1"])}
            role = {"emb_lr": "LR control (CNN embedding)", "hog_lr": "LR control (HOG)", "fusion_lr": "Hybrid (embedding + HOG)"}[test_rows[run_id]["row"]]
        else:
            val_path = PHASE2 / "runs" / run_id / "val_predictions.csv.gz"
            recorded_val = read_json(PHASE2 / "runs" / run_id / "record.json")["val_metrics"]
            role = "Selected CNN" if run_id in test_rows else "CNN ablation (validation only)"
        files.append({"run_id": run_id, "stage": "val", "path": val_path, "role": role, "recorded": recorded_val})
        if run_id in test_rows:
            files.append({"run_id": run_id, "stage": "test", "path": PHASE2 / "predictions" / f"{run_id}_test.csv.gz", "role": role,
                          "recorded": read_json(PHASE2 / "predictions" / f"{run_id}_test_metrics.json")})
    return files, freeze


def load_aligned(path, stage, run_id, labels, ids):
    return validate_predictions(pd.read_csv(path), ids, labels, stage, run_id)


def official_labels():
    _, train_labels = load_official("train")
    test_images, test_labels = load_official("test")
    with np.load(ROOT / "data" / "full_indices.npz") as split:
        val_ids = split["val_idx"]
    return train_labels, test_images, test_labels, val_ids


# --------------------------------------------------------------------------- audit

def audit(log=print):
    """Validate schema/identity of every prediction file and recompute its metrics from rows."""
    train_labels, _, test_labels, val_ids = official_labels()
    test_ids = np.arange(len(test_labels))
    files, freeze = prediction_files()
    rows, per_class_rows = [], []
    for f in files:
        labels, ids = (train_labels, val_ids) if f["stage"] == "val" else (test_labels, test_ids)
        aligned = load_aligned(f["path"], f["stage"], f["run_id"], labels, ids)
        m = label_metrics(aligned.y_true, aligned.y_pred)
        cm = np.asarray(m["confusion"])
        checks = {
            "columns_and_identity": True,  # validate_predictions raised otherwise
            "n_rows": len(aligned), "unique_ids": int(aligned.sample_id.nunique()),
            "support_sum_equals_n": int(sum(m["support"])) == len(ids),
            "every_support_equal": len(set(m["support"])) == 1,
            "diagonal_over_n_equals_accuracy": bool(np.isclose(cm.trace() / len(ids), m["accuracy"], rtol=0, atol=1e-12)),
            "accuracy_matches_record": bool(np.isclose(m["accuracy"], f["recorded"]["accuracy"], rtol=0, atol=1e-12)),
            "macro_f1_matches_record": bool(np.isclose(m["macro_f1"], f["recorded"]["macro_f1"], rtol=0, atol=1e-12)),
        }
        if "confusion" in f["recorded"]:
            checks["confusion_matches_record"] = bool(np.array_equal(cm, np.asarray(f["recorded"]["confusion"])))
        if f["stage"] == "test":
            checks["test_support_1000_each"] = m["support"] == [1000] * 10
        failed = [k for k, v in checks.items() if v is False]
        if failed:
            raise ValueError(f"{f['run_id']} {f['stage']}: failed checks {failed}")
        precision, recall, f1, support = per_class(m)
        rows.append({"run_id": f["run_id"], "role": f["role"], "stage": f["stage"], "n": len(aligned),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "recall_shirt": m["recall"][SHIRT],
                     "file": str(f["path"].relative_to(ROOT)), "file_sha256": sha256(f["path"]),
                     "checks_passed": len(checks)})
        for k in range(10):
            per_class_rows.append({"run_id": f["run_id"], "stage": f["stage"], "label": k, "category": CLASS_NAMES[k],
                                   "support": int(support[k]), "precision": precision[k], "recall": recall[k], "f1": f1[k]})
        log(f"OK {f['stage']:>4} {f['run_id']:<32} n={len(aligned):>5} acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics_export.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT / "per_class_metrics.csv", index=False)
    write_json(OUT / "prediction_audit.json", {
        "files_checked": len(rows), "all_passed": True,
        "schema": ["run_id", "stage", "source_split", "sample_id", "y_true", "y_pred"],
        "checks": ["required columns", "single run_id", "stage and source_split match the partition",
                   "exact ID membership, no duplicates", "category IDs 0-9", "y_true equals official label by sample_id",
                   "support sums to n", "diagonal / n equals accuracy", "accuracy, macro F1 (and confusion) equal the saved record",
                   "official test: support 1,000 for each of ten labels"],
        "validation_ids": "data/full_indices.npz val_idx (12,000 official_train IDs)",
        "test_ids": "official_test 0..9,999", "freeze_manifest_sha256": freeze["manifest_sha256"],
        "label_source": "load_official (checksum-verified IDX files)",
        "created_utc": datetime.now(timezone.utc).isoformat()})
    log(f"Audited {len(rows)} prediction files; all checks passed -> {OUT.relative_to(ROOT)}")
    return rows


# --------------------------------------------------------------------------- paired errors

def paired(log=print):
    freeze = read_json(PHASE2 / "freeze.json")
    cnn_id = freeze["test_rows"][0]["run_id"]
    _, _, test_labels, _ = official_labels()
    test_images = load_official("test")[0]
    ids = np.arange(len(test_labels))
    cnn = load_aligned(PHASE2 / "predictions" / f"{cnn_id}_test.csv.gz", "test", cnn_id, test_labels, ids)
    svm = load_aligned(PHASE1 / "predictions" / f"{SVM_ID}_test.csv.gz", "test", SVM_ID, test_labels, ids)
    y, pc, ps = test_labels, cnn.y_pred.to_numpy(), svm.y_pred.to_numpy()
    counts, group = outcome_groups(y, pc, ps)
    acc_delta = float((pc == y).mean() - (ps == y).mean())
    OUT.mkdir(parents=True, exist_ok=True)

    # Group table, overall and restricted to the four upper-body classes.
    table = []
    for scope, mask in [("all ten classes", np.ones_like(y, bool)), ("true T-shirt/Pullover/Coat/Shirt", np.isin(y, UPPER_BODY)),
                        ("true Shirt", y == SHIRT)]:
        for g in GROUPS:
            table.append({"scope": scope, "group": g, "count": int((group[mask] == g).sum()), "scope_n": int(mask.sum())})
    pd.DataFrame(table).to_csv(OUT / "paired_groups.csv", index=False)

    pairs = [r | {"model": "selected_cnn", "run_id": cnn_id} for r in label_pair_counts(y, pc)] + \
            [r | {"model": "hog_svm", "run_id": SVM_ID} for r in label_pair_counts(y, ps)]
    pd.DataFrame(pairs).to_csv(OUT / "label_pair_counts.csv", index=False)
    for name, pred in [("selected_cnn", pc), ("hog_svm", ps)]:
        m = label_metrics(y, pred)
        pd.DataFrame(m["confusion"], index=[f"true {c}" for c in CLASS_NAMES], columns=CLASS_NAMES).to_csv(OUT / f"confusion_{name}_test.csv")

    # Example IDs: first EXAMPLES_PER_GROUP sorted original IDs per group (documented rule, no cherry-picking).
    examples = []
    for panel, mask in [("all", np.ones_like(y, bool)), ("upper_body", np.isin(y, UPPER_BODY))]:
        for g in GROUPS:
            for sid in np.sort(ids[mask & (group == g)])[:EXAMPLES_PER_GROUP]:
                examples.append({"panel": panel, "group": g, "sample_id": int(sid), "y_true": int(y[sid]), "true_category": CLASS_NAMES[y[sid]],
                                 "cnn_pred": int(pc[sid]), "cnn_category": CLASS_NAMES[pc[sid]],
                                 "svm_pred": int(ps[sid]), "svm_category": CLASS_NAMES[ps[sid]]})
    pd.DataFrame(examples).to_csv(OUT / "figure_example_ids.csv", index=False)

    discordant = mcnemar_exact(counts["Only CNN correct"], counts["Only HOG + SVM correct"])
    shirt = {f"{m}_shirt_to_{CLASS_NAMES[p]}": int(((y == SHIRT) & (pred == p)).sum())
             for m, pred in [("cnn", pc), ("svm", ps)] for p in (TSHIRT, PULLOVER, COAT)}
    write_json(OUT / "paired_summary.json", {
        "cnn_run_id": cnn_id, "svm_run_id": SVM_ID, "n": len(y), "groups": counts,
        "groups_sum": sum(counts.values()), "accuracy_delta_cnn_minus_svm": acc_delta,
        "discordant_identity": f"({counts['Only CNN correct']} - {counts['Only HOG + SVM correct']}) / {len(y)} = {acc_delta:.4f}",
        "mcnemar": discordant, "true_shirt_support": int((y == SHIRT).sum()), "shirt_confusions": shirt,
        "example_rule": f"first {EXAMPLES_PER_GROUP} sorted original test IDs per group; panel 'upper_body' restricts to true "
                        "T-shirt/top, Pullover, Coat, Shirt",
        "created_utc": datetime.now(timezone.utc).isoformat()})
    _paired_figures(test_images, y, pc, ps, examples)
    log(f"Groups {counts}; sum={sum(counts.values())}; accuracy delta {acc_delta * 100:+.2f} pp; McNemar p={discordant['two_sided_p']:.3g}")
    return counts


def _paired_figures(images, y, pc, ps, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    short = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    ex = pd.DataFrame(examples)
    for panel in ["all", "upper_body"]:
        sub = ex[ex.panel == panel]
        fig, axes = plt.subplots(len(GROUPS), EXAMPLES_PER_GROUP, figsize=(2.1 * EXAMPLES_PER_GROUP, 2.45 * len(GROUPS)))
        for r, g in enumerate(GROUPS):
            rows = sub[sub.group == g].reset_index(drop=True)
            for c in range(EXAMPLES_PER_GROUP):
                ax = axes[r, c]
                ax.axis("off")
                if c >= len(rows):
                    continue
                e = rows.loc[c]
                ax.imshow(images[e.sample_id], cmap="gray_r", vmin=0, vmax=255)
                ax.set_title(f"ID {e.sample_id}\ntrue {short[e.y_true]}\nCNN {short[e.cnn_pred]} | SVM {short[e.svm_pred]}", fontsize=7.5)
            axes[r, 0].text(-0.35, 0.5, g, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="center", fontsize=9, weight="bold")
        fig.suptitle("Paired official-test outcomes: selected CNN vs HOG + RBF SVM\n"
                     f"first {EXAMPLES_PER_GROUP} sorted IDs per group" + (" (true T-shirt/Pullover/Coat/Shirt)" if panel == "upper_body" else ""), fontsize=9.5)
        fig.tight_layout()
        fig.savefig(FIGURES / f"paired_examples_{panel}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    mats = []
    for pred in (pc, ps):
        cm = np.asarray(label_metrics(y, pred)["confusion"], dtype=float)
        mats.append(cm / cm.sum(axis=1, keepdims=True))
    for ax, mat, title in [(axes[0], mats[0], "Selected CNN (CNN-3)"), (axes[1], mats[1], "HOG + RBF SVM (Phase 1)")]:
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if mat[i, j] >= 0.005:
                    ax.text(j, i, f"{mat[i, j] * 100:.0f}" if i != j else f"{mat[i, j] * 100:.1f}", ha="center", va="center",
                            fontsize=6.5, color="white" if mat[i, j] > 0.6 else "black")
        ax.set_title(f"{title}\nrow-normalised %, official test (n=1,000 per row)", fontsize=9)
    diff = (mats[0] - mats[1]) * 100
    lim = max(3.0, float(np.abs(diff).max()))
    axes[2].imshow(diff, cmap="RdBu", vmin=-lim, vmax=lim)
    for i in range(10):
        for j in range(10):
            if abs(diff[i, j]) >= 0.25:
                axes[2].text(j, i, f"{diff[i, j]:+.1f}", ha="center", va="center", fontsize=6.5)
    axes[2].set_title("CNN minus SVM (percentage points)\ndiagonal: + means CNN recall higher; off-diagonal: + means more CNN errors", fontsize=9)
    for ax in axes:
        ax.set_xticks(range(10), short, rotation=60, fontsize=7.5)
        ax.set_yticks(range(10), short, fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_cnn_vs_svm_test.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- timing

def _cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def _refit_phase1_svm(train_images, train_labels, test_images, test_labels, log):
    """Rebuild the frozen Phase 1 HOG + RBF SVM for a same-machine timing run.

    The Phase 1 joblib artifact is git-ignored and not in the checkout. SVC with these
    settings is deterministic, so a refit on the identical 48,000 fit IDs should give
    identical predictions; we check every official-test prediction against the saved file
    and record the result instead of assuming it.
    """
    import joblib
    from threadpoolctl import threadpool_limits
    from ..features import cached_features
    from ..models import make_model

    record = read_json(PHASE1 / "records" / f"{SVM_ID}.json")
    work = ROOT / "work" / "phase2_timing"
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{SVM_ID}_refit.joblib"
    original = ROOT / record["artifact_path"]
    if original.exists() and sha256(original) == record["artifact_sha256"]:
        return joblib.load(original)["pipeline"], {"source": "original Phase 1 artifact", "artifact_sha256": record["artifact_sha256"]}
    with np.load(ROOT / "data" / "full_indices.npz") as split:
        fit_ids = split["train_idx"]
    if path.exists():
        pipe = joblib.load(path)
    else:
        log("Phase 1 SVM artifact not in checkout; refitting HOG + RBF SVM (C=10) on the 48,000 fit IDs for timing ...")
        X, _ = cached_features(train_images[fit_ids], fit_ids, "official_train", "hog")
        pipe = make_model("svm", "hog", record["parameters"])
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            pipe.fit(X, train_labels[fit_ids])
            log(f"  refit in {time.perf_counter() - start:.0f}s")
        joblib.dump(pipe, path, compress=3)
    saved = pd.read_csv(PHASE1 / "predictions" / f"{SVM_ID}_test.csv.gz").set_index("sample_id").loc[np.arange(len(test_labels))].y_pred.to_numpy()
    Xt, _ = cached_features(test_images, np.arange(len(test_labels)), "official_test", "hog")
    with threadpool_limits(limits=1):
        refit_pred = pipe.predict(Xt)
    agree = int((refit_pred == saved).sum())
    model = pipe.named_steps["model"]
    same_settings = all(model.get_params()[k] == v for k, v in record["estimator_settings"].items() if k != "probability")
    info = {"source": "refit for timing only (Phase 1 artifact is git-ignored)", "refit_artifact": str(path.relative_to(ROOT)),
            "refit_artifact_sha256": sha256(path), "refit_artifact_bytes": path.stat().st_size,
            "estimator_settings_match_record": bool(same_settings),
            "test_predictions_identical_to_saved": f"{agree}/{len(saved)}",
            "support_vectors_refit": int(model.support_vectors_.shape[0]),
            "support_vectors_record": record["complexity"]["support_vectors"]}
    if agree != len(saved):
        log(f"WARNING: refit SVM agrees with saved predictions on {agree}/{len(saved)} IDs")
    return pipe, info


def timing(log=print):
    import joblib
    import torch
    from threadpoolctl import threadpool_limits
    from ..evaluate import benchmark
    from ..features import make_features, normalized_images
    from .models import parameter_count
    from .runner import load_checkpoint

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    freeze = read_json(PHASE2 / "freeze.json")
    train_images, train_labels = load_official("train")
    test_images, test_labels = load_official("test")
    bench_ids = np.arange(1024)  # same raw IDs 0-1023 as the Phase 1 timing policy
    raw = test_images[bench_ids]

    rows = {r["row"]: r for r in freeze["test_rows"]}
    cnn_row = rows["selected_cnn"]
    encoder, _ = load_checkpoint(ROOT / cnn_row["artifact_path"], "cpu", cnn_row["artifact_sha256"])
    heads = {name: joblib.load(ROOT / rows[name]["artifact_path"])["pipeline"] for name in ["emb_lr", "hog_lr", "fusion_lr"]}
    svm, svm_info = _refit_phase1_svm(train_images, train_labels, test_images, test_labels, log)

    def tensor(batch):
        return torch.from_numpy(normalized_images(batch)).unsqueeze(1)

    def cnn(batch):
        with torch.inference_mode():
            return encoder(tensor(batch)).argmax(1).numpy()

    def embed(batch):
        with torch.inference_mode():
            return encoder.embed(tensor(batch)).numpy()

    pipelines = {
        "hog_svm": (SVM_ID, lambda b: svm.predict(make_features(b, "hog")), "raw uint8 -> /255 -> HOG -> StandardScaler -> RBF SVC"),
        "selected_cnn": (cnn_row["run_id"], cnn, "raw uint8 -> /255 tensor [B,1,28,28] -> CNN-3 eval forward -> argmax"),
        "emb_lr": (rows["emb_lr"]["run_id"], lambda b: heads["emb_lr"].predict(embed(b)), "raw -> CNN-3 embedding (64) -> scaler -> LR"),
        "hog_lr": (rows["hog_lr"]["run_id"], lambda b: heads["hog_lr"].predict(make_features(b, "hog")), "raw -> HOG (1,296) -> scaler -> LR"),
        "fusion_lr": (rows["fusion_lr"]["run_id"], lambda b: heads["fusion_lr"].predict(np.hstack([embed(b), make_features(b, "hog")])),
                      "raw -> [CNN-3 embedding (64) | HOG (1,296)] -> scaler -> LR"),
    }
    measurements = {}
    try:
        for name, (run_id, fn, includes) in pipelines.items():
            # Correctness guard: the timed function must reproduce the saved test predictions on the timed IDs.
            path = (PHASE1 / "predictions" / f"{run_id}_test.csv.gz") if name == "hog_svm" else (PHASE2 / "predictions" / f"{run_id}_test.csv.gz")
            saved = pd.read_csv(path).set_index("sample_id").loc[bench_ids].y_pred.to_numpy()
            with threadpool_limits(limits=1):
                produced = np.concatenate([fn(raw[i:i + 128]) for i in range(0, len(raw), 128)])
            batch = benchmark(fn, raw, batch_size=128, repeats=5)
            single = benchmark(fn, raw[:128], batch_size=1, repeats=5)
            measurements[name] = {"run_id": run_id, "includes": includes,
                                  "timed_predictions_match_saved": f"{int((produced == saved).sum())}/{len(bench_ids)}",
                                  "batch128_amortized": batch, "single_image_latency": single,
                                  "torch_threads_during_timing": torch.get_num_threads()}
            log(f"{name:<13} median {batch['median_ms_per_image']:.4f} ms/img (range {batch['min_ms_per_image']:.4f}-"
                f"{batch['max_ms_per_image']:.4f}); single-image {single['median_ms_per_image']:.3f} ms")
    finally:
        torch.set_num_threads(previous_threads)

    # Complexity columns, with units.
    cnn_params = parameter_count(encoder)
    head_params = int(sum(p.numel() for p in encoder.classifier.parameters()))
    svm_record = read_json(PHASE1 / "records" / f"{SVM_ID}.json")
    complexity = [
        {"row": "selected_cnn", "run_id": cnn_row["run_id"], "value": cnn_params, "unit": "trainable float32 scalars (all CNN-3 parameters)",
         "scaler_scalars": 0, "artifact_bytes": (ROOT / cnn_row["artifact_path"]).stat().st_size,
         "artifact_format": "torch.save zip (state_dict + arch), float32, no extra compression"},
        {"row": "hog_svm", "run_id": SVM_ID, "value": svm_record["complexity"]["complexity_value"],
         "unit": "stored SVC decision-array scalars (support vectors + dual coefficients + intercepts); storage proxy, not learned weights",
         "support_vectors": svm_record["complexity"]["support_vectors"], "scaler_scalars": svm_record["complexity"]["scaler_stored_mean_scale_scalars"],
         "artifact_bytes": svm_record["serialized_bytes"], "artifact_format": "joblib bundle, compress=3 (Phase 1 record)"},
    ]
    for name in ["emb_lr", "hog_lr", "fusion_lr"]:
        model, scaler = heads[name].named_steps["model"], heads[name].named_steps["scale"]
        frozen = cnn_params - head_params if name != "hog_lr" else 0
        complexity.append({"row": name, "run_id": rows[name]["run_id"], "value": int(model.coef_.size + model.intercept_.size),
                           "unit": "LR coefficients + intercepts", "frozen_encoder_scalars_shipped": frozen,
                           "frozen_encoder_note": "CNN-3 conv blocks + dense layer up to the 64-value embedding (its own 650-value head unused)" if frozen else "none",
                           "scaler_scalars": int(scaler.mean_.size + scaler.scale_.size),
                           "artifact_bytes": (ROOT / rows[name]["artifact_path"]).stat().st_size,
                           "artifact_format": "joblib bundle (scikit-learn Pipeline)" + (" + CNN-3 checkpoint" if frozen else "")})
    env = {"python": platform.python_version(), "platform": platform.platform(), "cpu": _cpu_name(),
           "logical_cpus": os.cpu_count(), "torch": torch.__version__, "torch_threads": 1, "torch_interop_threads": torch.get_num_interop_threads(),
           "native_threads": "threadpoolctl limits=1 for NumPy/BLAS/OpenMP", "device": "cpu (no accelerator numbers reported)"}
    for package in ["numpy", "scikit-learn", "scikit-image", "threadpoolctl", "joblib"]:
        from importlib.metadata import version
        env[package] = version(package)
    OUT.mkdir(parents=True, exist_ok=True)
    phase1_timing = read_json(PHASE1 / "timing.json")
    historical = next(m for m in phase1_timing["measurements"] if m["run_id"] == SVM_ID)["end_to_end"]
    write_json(OUT / "timing.json", {
        "ids": "official_test 0..1023 (raw uint8)", "batch_size": 128, "warmup_batches": 1, "repeats": 5,
        "timer": "time.perf_counter wall time per 1,024 images per repeat, converted to ms/image; loading and disk I/O excluded",
        "environment": env, "phase1_svm_artifact": svm_info, "measurements": measurements,
        "phase1_historical_hog_svm": {"median_ms_per_image": historical["median_ms_per_image"],
                                      "environment": phase1_timing["environment"],
                                      "note": "Phase 1 end-to-end timing from results/full/timing.json; different machine and session"},
        "created_utc": datetime.now(timezone.utc).isoformat()})
    write_json(OUT / "complexity.json", complexity)
    return measurements


# --------------------------------------------------------------------------- comparison table

def compare(log=print):
    metrics = pd.read_csv(OUT / "metrics_export.csv")
    timing_info = read_json(OUT / "timing.json")["measurements"] if (OUT / "timing.json").exists() else {}
    by_run = {m["run_id"]: m for m in timing_info.values()}
    complexity = {c["run_id"]: c for c in read_json(OUT / "complexity.json")} if (OUT / "complexity.json").exists() else {}
    order = ["Selected CNN", "Phase 1 reference (validation-selected)", "LR control (CNN embedding)", "LR control (HOG)",
             "Hybrid (embedding + HOG)", "CNN ablation (validation only)", "Phase 1 context"]
    rows = []
    for run_id, g in metrics.groupby("run_id", sort=False):
        val, test = g[g.stage == "val"], g[g.stage == "test"]
        t = by_run.get(run_id)
        c = complexity.get(run_id, {})
        rows.append({"run_id": run_id, "role": g.role.iloc[0],
                     "val_macro_f1": float(val.macro_f1.iloc[0]) if len(val) else "pending",
                     "test_accuracy": float(test.accuracy.iloc[0]) if len(test) else "validation only",
                     "test_macro_f1": float(test.macro_f1.iloc[0]) if len(test) else "validation only",
                     "test_shirt_recall": float(test.recall_shirt.iloc[0]) if len(test) else "validation only",
                     "median_ms_per_image": t["batch128_amortized"]["median_ms_per_image"] if t else "not timed",
                     "ms_range": f"{t['batch128_amortized']['min_ms_per_image']:.4f}-{t['batch128_amortized']['max_ms_per_image']:.4f}" if t else "",
                     "complexity_value": c.get("value", ""), "complexity_unit": c.get("unit", "")})
    table = pd.DataFrame(rows)
    table["_o"] = table.role.map(order.index)
    table = table.sort_values(["_o", "run_id"]).drop(columns="_o")
    table.to_csv(OUT / "results_table.csv", index=False)

    # Deltas (unrounded, CNN minus SVM) and paired bootstrap on aligned test IDs.
    freeze = read_json(PHASE2 / "freeze.json")
    cnn_id = freeze["test_rows"][0]["run_id"]
    pc = pd.read_csv(PHASE2 / "predictions" / f"{cnn_id}_test.csv.gz").sort_values("sample_id")
    ps = pd.read_csv(PHASE1 / "predictions" / f"{SVM_ID}_test.csv.gz").sort_values("sample_id")
    if not (np.array_equal(pc.sample_id, ps.sample_id) and np.array_equal(pc.y_true, ps.y_true)):
        raise ValueError("CNN and SVM test files are not aligned on IDs and labels")
    y = pc.y_true.to_numpy()
    pcl = pd.read_csv(OUT / "per_class_metrics.csv")
    rc = pcl[(pcl.run_id == cnn_id) & (pcl.stage == "test")].set_index("label")
    rs = pcl[(pcl.run_id == SVM_ID) & (pcl.stage == "test")].set_index("label")
    mc, ms = metrics[(metrics.run_id == cnn_id) & (metrics.stage == "test")].iloc[0], metrics[(metrics.run_id == SVM_ID) & (metrics.stage == "test")].iloc[0]
    deltas = {"accuracy_pp": (mc.accuracy - ms.accuracy) * 100, "macro_f1_points_x100": (mc.macro_f1 - ms.macro_f1) * 100,
              "recall_pp_by_class": {CLASS_NAMES[k]: (rc.recall[k] - rs.recall[k]) * 100 for k in range(10)}}
    boot = paired_bootstrap(y, pc.y_pred.to_numpy(), ps.y_pred.to_numpy())
    t_cnn, t_svm = by_run.get(cnn_id), by_run.get(SVM_ID)
    if t_cnn and t_svm:
        deltas["speed_ratio_svm_over_cnn"] = t_svm["batch128_amortized"]["median_ms_per_image"] / t_cnn["batch128_amortized"]["median_ms_per_image"]
    write_json(OUT / "cnn_vs_svm_deltas.json", {"cnn_run_id": cnn_id, "svm_run_id": SVM_ID, "deltas_cnn_minus_svm": deltas,
                                                "paired_bootstrap": boot, "rounding": "deltas from unrounded values; round once when reporting"})
    recall_wide = pd.DataFrame({"label": range(10), "category": CLASS_NAMES, "support": rc.support.values,
                                "cnn_recall": rc.recall.values, "svm_recall": rs.recall.values,
                                "delta_pp": (rc.recall.values - rs.recall.values) * 100,
                                "cnn_precision": rc.precision.values, "svm_precision": rs.precision.values})
    recall_wide.to_csv(OUT / "per_class_cnn_vs_svm_test.csv", index=False)
    log(table.to_string(index=False))
    log(f"CNN - SVM: accuracy {deltas['accuracy_pp']:+.2f} pp, macro F1 {deltas['macro_f1_points_x100']:+.2f} points; "
        f"bootstrap 95% acc {np.round(np.array(boot['accuracy_delta_ci95']) * 100, 2)} pp")
    return table


# --------------------------------------------------------------------------- live demo

def lookup(sample_id, log=print):
    """Show one official-test ID in both saved files, its confusion cells, then recompute Shirt recall."""
    freeze = read_json(PHASE2 / "freeze.json")
    cnn_id = freeze["test_rows"][0]["run_id"]
    files = {"selected CNN": PHASE2 / "predictions" / f"{cnn_id}_test.csv.gz", "HOG + SVM": PHASE1 / "predictions" / f"{SVM_ID}_test.csv.gz"}
    for name, path in files.items():
        frame = pd.read_csv(path)
        row = frame[frame.sample_id == sample_id].iloc[0]
        cell = int(((frame.y_true == row.y_true) & (frame.y_pred == row.y_pred)).sum())
        shirts = frame[frame.y_true == SHIRT]
        log(f"{name:<13} {path.relative_to(ROOT)}\n  ID {sample_id}: true {CLASS_NAMES[row.y_true]} ({row.y_true}), "
            f"predicted {CLASS_NAMES[row.y_pred]} ({row.y_pred}) -> confusion cell [{row.y_true},{row.y_pred}] holds {cell} images\n"
            f"  Shirt recall recomputed = {int((shirts.y_pred == SHIRT).sum())} / {len(shirts)} = {(shirts.y_pred == SHIRT).mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["audit", "paired", "timing", "compare", "all", "lookup"])
    parser.add_argument("--id", type=int, default=0, help="official test ID for lookup")
    args = parser.parse_args()
    if args.command == "lookup":
        lookup(args.id)
        return
    steps = {"audit": [audit], "paired": [paired], "timing": [timing], "compare": [compare],
             "all": [audit, paired, timing, compare]}[args.command]
    for step in steps:
        step()


if __name__ == "__main__":
    main()
