"""Metrics, sample-ID validation, consistent timing, and model storage proxies."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from .data import CLASS_NAMES

LABELS = np.arange(10)


def label_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or not len(y_true) or y_true.shape != y_pred.shape:
        raise ValueError("Require nonempty, aligned 1D label arrays")
    if not np.isin(y_true, LABELS).all() or not np.isin(y_pred, LABELS).all():
        raise ValueError("Unexpected category ID")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    support = cm.sum(axis=1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(10, dtype=float), where=support != 0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "support": support.tolist(), "recall": recall.tolist(), "confusion": cm.tolist()}


def prediction_frame(run_id, stage, ids, labels, predictions):
    if stage not in {"val", "test"}:
        raise ValueError("stage must be val or test")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate original sample IDs")
    return pd.DataFrame({"run_id": run_id, "stage": stage,
                         "source_split": "official_train" if stage == "val" else "official_test",
                         "sample_id": ids, "y_true": labels, "y_pred": predictions})


def validate_predictions(frame, ids, authoritative_labels, stage, run_id=None):
    required = {"run_id", "stage", "source_split", "sample_id", "y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing prediction columns")
    if len(frame) != len(ids) or frame.sample_id.duplicated().any():
        raise ValueError("Missing or duplicate prediction IDs")
    if not np.array_equal(np.sort(frame.sample_id.to_numpy()), np.sort(ids)):
        raise ValueError("Wrong prediction ID membership")
    source = "official_train" if stage == "val" else "official_test"
    if set(frame.stage) != {stage} or set(frame.source_split) != {source}:
        raise ValueError("Wrong stage or original partition")
    if frame.run_id.nunique() != 1 or (run_id is not None and set(frame.run_id) != {run_id}):
        raise ValueError("Mixed or incorrect run IDs")
    ordered = frame.set_index("sample_id").loc[ids].reset_index()
    if not np.array_equal(ordered.y_true.to_numpy(), authoritative_labels[ids]):
        raise ValueError("Exported labels do not match authoritative labels")
    label_metrics(ordered.y_true, ordered.y_pred)
    return ordered


def predict_batches(pipe, features, batch_size=128):
    return np.concatenate([pipe.predict(features[start:start + batch_size]) for start in range(0, len(features), batch_size)])


def benchmark(predict_fn, inputs, batch_size=128, repeats=5):
    if len(inputs) == 0 or batch_size <= 0 or repeats <= 0:
        raise ValueError("Positive batch size, repetitions and nonempty inputs required")
    durations = []
    with threadpool_limits(limits=1):
        predict_fn(inputs[:min(batch_size, len(inputs))])
        for _ in range(repeats):
            start = time.perf_counter()
            for index in range(0, len(inputs), batch_size):
                predict_fn(inputs[index:index + batch_size])
            durations.append(time.perf_counter() - start)
    ms = np.asarray(durations) * 1000 / len(inputs)
    return {"median_ms_per_image": float(np.median(ms)), "min_ms_per_image": float(ms.min()),
            "max_ms_per_image": float(ms.max()), "repeat_seconds": durations,
            "images": len(inputs), "batch_size": batch_size, "repeats": repeats,
            "warmup_batches": 1, "native_threads": 1,
            "interpretation": "amortized batch cost; excludes disk reads and model loading"}


def complexity(pipe):
    model = pipe.named_steps["model"]
    name = type(model).__name__
    if name in {"LogisticRegression", "LinearSVC"}:
        value = int(model.coef_.size + model.intercept_.size)
        result = {"complexity_type": "learned classifier coefficient and intercept scalars", "complexity_value": value}
    elif name == "KNeighborsClassifier":
        result = {"complexity_type": "stored training feature scalars (not learned weights)",
                  "complexity_value": int(model._fit_X.size), "stored_rows": int(model._fit_X.shape[0]),
                  "stored_feature_bytes": int(model._fit_X.nbytes),
                  "learned_weight_count": None, "learned_weight_note": "not applicable to instance-based kNN"}
    elif name == "RandomForestClassifier":
        result = {"complexity_type": "total tree nodes (not neural-equivalent parameters)",
                  "complexity_value": sum(tree.tree_.node_count for tree in model.estimators_),
                  "trees": len(model.estimators_), "max_tree_depth": max(t.tree_.max_depth for t in model.estimators_)}
    elif name == "SVC":
        result = {"complexity_type": "stored SVC decision-array scalars (storage proxy)",
                  "complexity_value": int(model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size),
                  "support_vectors": int(model.support_vectors_.shape[0])}
    else:
        raise ValueError(f"Unsupported complexity summary {name}")
    scaler = pipe.named_steps["scale"]
    result["scaler_stored_mean_scale_scalars"] = 0 if isinstance(scaler, str) else int(scaler.mean_.size + scaler.scale_.size)
    return result


def recall_rows(run_id, stage, metrics):
    return [{"run_id": run_id, "stage": stage, "label": i, "category": CLASS_NAMES[i],
             "support": metrics["support"][i], "recall": metrics["recall"][i]} for i in range(10)]

