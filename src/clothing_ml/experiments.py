"""Run -> validation selection -> immutable freeze -> final held-out evaluation."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from .data import load_official, prepare
from .evaluate import benchmark, complexity, label_metrics, predict_batches, prediction_frame, recall_rows, validate_predictions
from .features import HOG_SETTINGS, cached_features, make_features
from .models import MODEL_NAMES, candidates, make_model
from .utils import ROOT, RESULTS_ROOT, environment, read_json, sha256, stable_hash, write_json


def profile_paths(profile):
    if profile not in {"smoke", "budget", "full"}:
        raise ValueError("Unknown profile")
    result = RESULTS_ROOT / profile
    artifact_root = ROOT / "artifacts" if RESULTS_ROOT == ROOT / "results" else ROOT / "artifacts" / stable_hash(str(RESULTS_ROOT))[:12]
    artifact = artifact_root / profile
    for directory in [result, result / "records", result / "predictions", artifact]:
        directory.mkdir(parents=True, exist_ok=True)
    return result, artifact


def train(profile="budget", search="compact", selection_from=None):
    result, artifact = profile_paths(profile)
    if (result / "freeze.json").exists():
        raise ValueError("This profile is frozen; use a fresh profile/config rather than overwrite")
    images, labels = prepare()
    split_path = ROOT / f"data/{profile}_indices.npz"
    with np.load(split_path) as saved:
        tr, va = saved["train_idx"], saved["val_idx"]
    params_by_row = None
    if selection_from:
        previous = read_json(RESULTS_ROOT / selection_from / "freeze.json")
        validate_manifest(previous)
        params_by_row = {(r["feature"], r["model"]): r["parameters"] for r in previous["finalists"]}
    settings = {"profile": profile, "search": search, "selection_from": selection_from,
                "train_n": len(tr), "validation_n": len(va), "seed": 42,
                "split_sha256": sha256(split_path), "hog": HOG_SETTINGS,
                "metric": "validation_macro_f1", "refit_on_train_plus_val": False,
                "tie_break": "accuracy descending, validation prediction time ascending, run ID ascending",
                "source_files_sha256": {p.name: sha256(p) for p in sorted((ROOT / "src/clothing_ml").glob("*.py"))}}
    write_json(result / "protocol.json", settings)
    write_json(result / "environment.json", environment())
    pd.DataFrame({"sample_id": tr, "source_split": "official_train"}).to_csv(result / "fit_ids.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    rows = []
    for feature in ["pixels", "hog"]:
        tick = time.perf_counter()
        Xtr, train_key = cached_features(images[tr], tr, "official_train", feature)
        Xva, val_key = cached_features(images[va], va, "official_train", feature)
        feature_seconds = time.perf_counter() - tick
        print(f"{profile}: {feature} features ready {Xtr.shape}; {feature_seconds:.1f}s (may use cache)", flush=True)
        for model_name in MODEL_NAMES:
            choices = [params_by_row[(feature, model_name)]] if params_by_row else candidates(model_name, "baseline" if profile == "smoke" else search)
            for candidate_no, parameters in enumerate(choices):
                run_id = f"{profile}_{feature}_{model_name}_{stable_hash(parameters)[:8]}"
                record_path = result / "records" / f"{run_id}.json"
                model_path = artifact / f"{run_id}.joblib"
                fingerprint = stable_hash({"protocol": settings, "parameters": parameters, "train_cache": train_key, "val_cache": val_key})
                if record_path.exists():
                    previous_record = read_json(record_path)
                    if previous_record.get("fingerprint") != fingerprint:
                        raise ValueError("Run record belongs to another configuration")
                    if previous_record.get("status") == "completed" and model_path.exists() and sha256(model_path) == previous_record["artifact_sha256"]:
                        rows.append(previous_record)
                        print(f"Reuse completed run: {run_id}", flush=True)
                        continue
                print(f"Training {run_id}: {parameters}", flush=True)
                pipe = make_model(model_name, feature, parameters, smoke=profile == "smoke")
                record = dict(run_id=run_id, profile=profile, feature=feature, model=model_name,
                              parameters=parameters, train_n=len(tr), val_n=len(va), feature_dim=Xtr.shape[1],
                              fingerprint=fingerprint, split_sha256=sha256(split_path), candidate_number=candidate_no,
                              status="running", feature_cache_seconds=feature_seconds,
                              estimator_settings=pipe.named_steps["model"].get_params(deep=False),
                              scaler=type(pipe.named_steps["scale"]).__name__)
                write_json(record_path, record)
                try:
                    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
                        warnings.simplefilter("always")
                        start = time.perf_counter()
                        pipe.fit(Xtr, labels[tr])
                        record["fit_seconds"] = time.perf_counter() - start
                        start = time.perf_counter()
                        pred = predict_batches(pipe, Xva)
                        record["validation_predict_seconds"] = time.perf_counter() - start
                    record["warnings"] = [str(w.message) for w in captured]
                    record["converged"] = not any("converg" in message.lower() or "iteration" in message.lower() for message in record["warnings"])
                    metrics = label_metrics(labels[va], pred)
                    frame = prediction_frame(run_id, "val", va, labels[va], pred)
                    validate_predictions(frame, va, labels, "val", run_id)
                    pred_path = result / "predictions" / f"{run_id}_val.csv.gz"
                    frame.to_csv(pred_path, index=False, compression={"method": "gzip", "mtime": 0})
                    bundle = {"pipeline": pipe, "feature": feature, "hog": HOG_SETTINGS, "run_id": run_id,
                              "raw_input": "uint8_N_28_28", "protocol": settings, "parameters": parameters}
                    joblib.dump(bundle, model_path, compress=3)
                    reloaded = joblib.load(model_path)
                    if not np.array_equal(pipe.predict(Xva[:8]), reloaded["pipeline"].predict(Xva[:8])):
                        raise ValueError("Reloaded artifact changed predictions")
                    record.update(accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                                  metrics=metrics, complexity=complexity(pipe),
                                  artifact_path=str(model_path.relative_to(ROOT)), artifact_sha256=sha256(model_path),
                                  prediction_sha256=sha256(pred_path), serialized_bytes=model_path.stat().st_size,
                                  status="completed")
                    write_json(record_path, record)
                    rows.append(record)
                    print(f"  validation F1={record['macro_f1']:.4f}, accuracy={record['accuracy']:.4f}, fit={record['fit_seconds']:.1f}s", flush=True)
                except Exception as error:
                    record.update(status="failed", exception=repr(error))
                    write_json(record_path, record)
                    raise
        del Xtr, Xva
    summary_keys = ["run_id", "feature", "model", "train_n", "val_n", "feature_dim", "accuracy", "macro_f1", "fit_seconds", "validation_predict_seconds", "serialized_bytes", "converged"]
    pd.DataFrame([{key: row[key] for key in summary_keys} for row in rows]).to_csv(result / "validation_candidates.csv", index=False)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(tr), 1)), labels[tr])
    write_json(result / "dummy_validation.json", label_metrics(labels[va], dummy.predict(np.zeros((len(va), 1)))))
    return rows


def freeze(profile="budget"):
    result, _ = profile_paths(profile)
    target = result / "freeze.json"
    if target.exists():
        return read_json(target)
    rows = [read_json(path) for path in sorted((result / "records").glob("*.json"))]
    protocol = read_json(result / "protocol.json")
    finalists = []
    for feature in ["pixels", "hog"]:
        for model in MODEL_NAMES:
            matching = [r for r in rows if r["feature"] == feature and r["model"] == model and r["status"] == "completed" and r["converged"]]
            if not matching:
                raise ValueError(f"Missing completed converged row: {feature}/{model}")
            matching.sort(key=lambda r: (-r["macro_f1"], -r["accuracy"], r["validation_predict_seconds"], r["run_id"]))
            chosen = matching[0]
            if sha256(ROOT / chosen["artifact_path"]) != chosen["artifact_sha256"]:
                raise ValueError("Finalist model changed before freeze")
            finalists.append(chosen)
    if len({r["split_sha256"] for r in finalists}) != 1 or len({r["train_n"] for r in finalists}) != 1:
        raise ValueError("Finalists use inconsistent splits or training sizes")
    selected = sorted(finalists, key=lambda r: (-r["macro_f1"], -r["accuracy"], r["validation_predict_seconds"], r["run_id"]))[0]
    manifest = {"protocol": protocol, "finalists": finalists, "selected_run_id": selected["run_id"],
                "selection_stage": "validation_only", "test_used_for_selection": False,
                "environment": environment()}
    manifest["manifest_sha256"] = stable_hash(manifest)
    write_json(target, manifest)
    pd.DataFrame([{key: r[key] for key in ["run_id", "feature", "model", "train_n", "feature_dim", "accuracy", "macro_f1", "fit_seconds", "serialized_bytes"]} for r in finalists]).to_csv(result / "validation_summary.csv", index=False)
    print(f"Frozen {profile}: selected {selected['run_id']} using validation only", flush=True)
    return manifest


def validate_manifest(manifest):
    expected = manifest["manifest_sha256"]
    if stable_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != expected:
        raise ValueError("Freeze manifest integrity mismatch")
    if manifest.get("test_used_for_selection") or manifest.get("selection_stage") != "validation_only":
        raise ValueError("Manifest does not establish validation-only selection")
    return expected


def final_test(profile="budget", allow_test=False):
    if not allow_test:
        raise ValueError("Final test access requires --allow-test after freezing validation choices")
    if profile == "smoke":
        raise ValueError("Smoke runs are plumbing checks and cannot produce final test results")
    result, _ = profile_paths(profile)
    manifest = read_json(result / "freeze.json")
    expected = validate_manifest(manifest)
    if (result / "test_summary.csv").exists():
        raise ValueError("Final test already recorded. Reuse saved predictions; do not retune or overwrite.")
    images, labels = load_official("test")  # first test-content access in normal workflow
    ids = np.arange(len(labels))
    test_rows, recalls, timings, complexity_rows = [], [], [], []
    bench_ids = ids[:1024]  # fixed independent of labels or model outcomes
    for row in manifest["finalists"]:
        if sha256(ROOT / row["artifact_path"]) != row["artifact_sha256"]:
            raise ValueError("Frozen model hash mismatch")
        bundle = joblib.load(ROOT / row["artifact_path"])
        pipe, feature, run_id = bundle["pipeline"], row["feature"], row["run_id"]
        Xtest, _ = cached_features(images, ids, "official_test", feature)
        with threadpool_limits(limits=1):
            predictions = predict_batches(pipe, Xtest)
        frame = prediction_frame(run_id, "test", ids, labels, predictions)
        validate_predictions(frame, ids, labels, "test", run_id)
        frame.to_csv(result / "predictions" / f"{run_id}_test.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        metrics = label_metrics(labels, predictions)
        write_json(result / f"{run_id}_test_metrics.json", metrics)
        prediction_timing = benchmark(pipe.predict, Xtest[bench_ids])
        e2e = benchmark(lambda raw: pipe.predict(make_features(raw, feature)), images[bench_ids])
        timing = {"run_id": run_id, "pipeline_prediction": prediction_timing, "end_to_end": e2e}
        timings.append(timing)
        test_rows.append({"run_id": run_id, "feature": feature, "model": row["model"],
                          "train_n": row["train_n"], "feature_dim": row["feature_dim"],
                          "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                          "fit_seconds": row["fit_seconds"], "serialized_bytes": row["serialized_bytes"],
                          "predict_ms_per_image": prediction_timing["median_ms_per_image"],
                          "end_to_end_ms_per_image": e2e["median_ms_per_image"],
                          "validation_selected": run_id == manifest["selected_run_id"]})
        recalls.extend(recall_rows(run_id, "test", metrics))
        recalls.extend(recall_rows(run_id, "val", row["metrics"]))
        complexity_rows.append({"run_id": run_id, **row["complexity"], "serialized_bytes": row["serialized_bytes"]})
        print(f"Final test {run_id}: accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}", flush=True)
    pd.DataFrame(test_rows).to_csv(result / "test_summary.csv", index=False)
    pd.DataFrame(recalls).to_csv(result / "per_class_recall.csv", index=False)
    write_json(result / "timing.json", {"environment": environment(), "ids": bench_ids, "measurements": timings})
    write_json(result / "complexity.json", complexity_rows)
    write_json(result / "test_receipt.json", {"freeze_manifest_sha256": expected,
                                              "test_summary_sha256": sha256(result / "test_summary.csv"),
                                              "test_n": len(ids), "test_tuned": False})
    return test_rows

