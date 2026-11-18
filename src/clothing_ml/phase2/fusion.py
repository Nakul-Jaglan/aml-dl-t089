"""Optional hybrid: frozen CNN embedding (64) + fixed HOG (1,296) -> Logistic Regression.

Hypothesis: HOG adds edge cues the selected CNN embedding misses. Three LR heads
share one recipe so the fusion row can be read against two controls:
  emb_lr     64 learned features      (effect of swapping the neural head for LR)
  hog_lr     1,296 fixed features     (the HOG branch alone)
  fusion_lr  1,360 joined features    (actual ML + DL fusion)
The encoder is the validation-selected checkpoint, frozen before any head is fitted.
Official test data is never loaded here.
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data import load_official
from ..evaluate import benchmark, label_metrics, prediction_frame, validate_predictions
from ..features import HOG_SETTINGS, cached_features, make_features, normalized_images
from ..utils import ROOT, read_json, sha256, stable_hash, write_json
from .data import FashionIDDataset, make_loader, split_ids
from .runner import load_checkpoint, phase2_paths

C_GRID = [0.1, 1.0]  # same grid and order for every head; earlier C wins a full tie
ROWS = ["emb_lr", "hog_lr", "fusion_lr"]
SELECTION_RULE = "per head: highest validation macro F1, then accuracy, then C order [0.1, 1.0]; converged fits only"


def extract_embeddings(model, images, ids, labels):
    """Canonical images, eval mode, inference_mode: deterministic 64-value features in ID order."""
    model.eval()
    loader = make_loader(FashionIDDataset(images, labels, ids, "none"), 512, shuffle=False)
    out, seen = [], []
    with torch.inference_mode():
        for x, _y, original_ids in loader:
            out.append(model.embed(x).numpy())
            seen.append(original_ids.numpy())
    if not np.array_equal(np.concatenate(seen), ids):
        raise ValueError("Embedding order does not match requested IDs")
    features = np.concatenate(out).astype(np.float32)
    if features.shape != (len(ids), 64) or not np.isfinite(features).all():
        raise ValueError("Invalid embedding matrix")
    return features


def keyed(ids, labels, source_split):
    return pd.DataFrame({"source_split": source_split, "sample_id": ids, "y_true": labels[ids]})


def joined_features(emb, emb_keys, hog, hog_keys):
    """Join by original partition + ID and assert identical labels and rows before concatenation."""
    if not emb_keys.equals(hog_keys):
        raise ValueError("Embedding and HOG rows differ in partition, ID or label")
    if len(emb) != len(hog):
        raise ValueError("Row-count mismatch")
    return np.hstack([emb, hog]).astype(np.float32)


def row_matrix(row, emb, hog):
    return {"emb_lr": emb, "hog_lr": hog, "fusion_lr": np.hstack([emb, hog]).astype(np.float32)}[row]


def make_head(C):
    # Scaler is fitted on fit rows only, inside the pipeline (no preprocessing leakage).
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42, C=C))])


def run_fusion(profile="full", log=print):
    if profile == "smoke":
        log("Note: smoke fusion is a plumbing check only")
    result, artifact = phase2_paths(profile)
    out_dir, model_dir = result / "fusion", artifact / "fusion"
    for d in [out_dir / "predictions", model_dir]:
        d.mkdir(parents=True, exist_ok=True)
    selection = read_json(result / "cnn_selection.json")
    encoder, bundle = load_checkpoint(ROOT / selection["checkpoint_path"], "cpu", selection["checkpoint_sha256"])
    images, labels = load_official("train")
    fit_ids, val_ids, split_sha = split_ids(profile)
    if split_sha != selection["split_sha256"]:
        raise ValueError("Split file changed since CNN selection")

    tick = time.perf_counter()
    emb_fit, emb_val = (extract_embeddings(encoder, images, ids, labels) for ids in (fit_ids, val_ids))
    embed_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    hog_fit, fit_key = cached_features(images[fit_ids], fit_ids, "official_train", "hog")
    hog_val, val_key = cached_features(images[val_ids], val_ids, "official_train", "hog")
    hog_seconds = time.perf_counter() - tick
    fit_keys, val_keys = keyed(fit_ids, labels, "official_train"), keyed(val_ids, labels, "official_train")
    matrices = {"fit": {"emb_lr": emb_fit, "hog_lr": hog_fit, "fusion_lr": joined_features(emb_fit, fit_keys, hog_fit, fit_keys.copy())},
                "val": {"emb_lr": emb_val, "hog_lr": hog_val, "fusion_lr": joined_features(emb_val, val_keys, hog_val, val_keys.copy())}}
    y_fit, y_val = labels[fit_ids], labels[val_ids]
    log(f"Features ready: embedding {emb_fit.shape} ({embed_seconds:.1f}s), HOG {hog_fit.shape} ({hog_seconds:.1f}s, may use cache)")

    extraction = {"encoder_run_id": selection["selected_run_id"], "encoder_checkpoint_sha256": selection["checkpoint_sha256"],
                  "encoder_epoch": bundle["epoch"], "embedding": "dense ReLU output before dropout, eval mode, canonical images",
                  "hog_settings": HOG_SETTINGS, "hog_cache_keys": [fit_key, val_key], "split_sha256": split_sha}
    candidates = []
    for row in ROWS:
        for C in C_GRID:
            run_id = f"{profile}_{row}_C{C:g}_{stable_hash({'extraction': extraction, 'C': C, 'row': row})[:8]}"
            head = make_head(C)
            with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                start = time.perf_counter()
                head.fit(matrices["fit"][row], y_fit)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                pred = head.predict(matrices["val"][row])
                predict_seconds = time.perf_counter() - start
            messages = [str(w.message) for w in captured]
            converged = not any("converg" in m.lower() or "iteration" in m.lower() for m in messages)
            frame = prediction_frame(run_id, "val", val_ids, y_val, pred)
            validate_predictions(frame, val_ids, labels, "val", run_id)
            pred_path = out_dir / "predictions" / f"{run_id}_val.csv.gz"
            frame.to_csv(pred_path, index=False, compression={"method": "gzip", "mtime": 0})
            model_path = model_dir / f"{run_id}.joblib"
            joblib.dump({"pipeline": head, "row": row, "C": C, "extraction": extraction}, model_path, compress=3)
            metrics = label_metrics(y_val, pred)
            candidates.append({"row": row, "C": C, "run_id": run_id, "feature_dim": matrices["fit"][row].shape[1],
                               "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
                               "recall_tshirt": metrics["recall"][0], "recall_pullover": metrics["recall"][2],
                               "recall_shirt": metrics["recall"][6], "fit_seconds": fit_seconds,
                               "val_predict_seconds": predict_seconds, "converged": converged, "warnings": messages,
                               "artifact_path": str(model_path.relative_to(ROOT)), "artifact_sha256": sha256(model_path),
                               "predictions_sha256": sha256(pred_path), "metrics": metrics, "pred": pred})
            log(f"  {row:9s} C={C:<4g} dim={matrices['fit'][row].shape[1]:5d} val F1={metrics['macro_f1']:.4f} "
                f"acc={metrics['accuracy']:.4f} fit={fit_seconds:.1f}s converged={converged}")

    selected = []
    for row in ROWS:
        pool = [c for c in candidates if c["row"] == row and c["converged"]]
        if not pool:
            raise ValueError(f"No converged candidate for {row}")
        selected.append(sorted(pool, key=lambda c: (-c["val_macro_f1"], -c["val_accuracy"], C_GRID.index(c["C"])))[0])
    by_row = {c["row"]: c for c in selected}

    # Paired validation errors: does fusion fix examples each control gets wrong?
    cnn_preds = pd.read_csv(result / "runs" / selection["selected_run_id"] / "val_predictions.csv.gz").set_index("sample_id").loc[val_ids].y_pred.to_numpy()
    paired = {}
    for other_name, other in [("emb_lr", by_row["emb_lr"]["pred"]), ("hog_lr", by_row["hog_lr"]["pred"]), ("selected_cnn", cnn_preds)]:
        fusion_ok, other_ok = by_row["fusion_lr"]["pred"] == y_val, other == y_val
        paired[f"fusion_vs_{other_name}"] = {"both_correct": int((fusion_ok & other_ok).sum()),
                                             "only_fusion_correct": int((fusion_ok & ~other_ok).sum()),
                                             "only_other_correct": int((~fusion_ok & other_ok).sum()),
                                             "both_wrong": int((~fusion_ok & ~other_ok).sum())}

    # End-to-end development timing on 1,024 fixed validation IDs (label-independent choice).
    bench_ids = val_ids[:1024]
    raw = images[bench_ids]
    heads = {row: joblib.load(ROOT / by_row[row]["artifact_path"])["pipeline"] for row in ROWS}

    def embed_raw(batch):
        with torch.inference_mode():
            return encoder.embed(torch.from_numpy(normalized_images(batch)).unsqueeze(1)).numpy()

    def cnn_raw(batch):
        with torch.inference_mode():
            return encoder(torch.from_numpy(normalized_images(batch)).unsqueeze(1)).argmax(1).numpy()

    pipelines = {"selected_cnn": cnn_raw,
                 "emb_lr": lambda b: heads["emb_lr"].predict(embed_raw(b)),
                 "hog_lr": lambda b: heads["hog_lr"].predict(make_features(b, "hog")),
                 "fusion_lr": lambda b: heads["fusion_lr"].predict(np.hstack([embed_raw(b), make_features(b, "hog")]))}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        timing = {name: benchmark(fn, raw) for name, fn in pipelines.items()}
    finally:
        torch.set_num_threads(threads)

    cnn_record = read_json(result / "runs" / selection["selected_run_id"] / "record.json")
    cm = cnn_record["val_metrics"]
    summary = [{"row": "selected_cnn", "run_id": selection["selected_run_id"], "C": None, "feature_dim": 64,
                "val_macro_f1": cm["macro_f1"], "val_accuracy": cm["accuracy"], "recall_tshirt": cm["recall"][0],
                "recall_pullover": cm["recall"][2], "recall_shirt": cm["recall"][6],
                "end_to_end_ms_per_image": timing["selected_cnn"]["median_ms_per_image"]}]
    for c in selected:
        summary.append({k: c[k] for k in ["row", "run_id", "C", "feature_dim", "val_macro_f1", "val_accuracy",
                                          "recall_tshirt", "recall_pullover", "recall_shirt"]}
                       | {"end_to_end_ms_per_image": timing[c["row"]]["median_ms_per_image"]})
    pd.DataFrame(summary).to_csv(out_dir / "fusion_summary.csv", index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k not in {"metrics", "pred", "warnings"}} for c in candidates]).to_csv(out_dir / "fusion_candidates.csv", index=False)
    write_json(out_dir / "fusion_timing.json", {"ids": "first 1,024 validation IDs in saved order", "measurements": timing,
                                                "torch_threads": 1, "includes": "raw uint8 -> normalization/HOG/CNN features -> scaler -> LR"})
    write_json(out_dir / "fusion_selection.json", {
        "selection_rule": SELECTION_RULE, "selection_stage": "validation_only", "test_used_for_selection": False,
        "encoder_checkpoint_sha256": selection["checkpoint_sha256"], "extraction": extraction, "C_grid": C_GRID,
        "selected_rows": [{k: c[k] for k in ["row", "run_id", "C", "artifact_path", "artifact_sha256", "val_macro_f1", "val_accuracy"]} for c in selected],
        "paired_validation_errors": paired, "embed_seconds_fit_plus_val": embed_seconds,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    log(pd.DataFrame(summary).to_string(index=False))
    log(paired)
    return summary
