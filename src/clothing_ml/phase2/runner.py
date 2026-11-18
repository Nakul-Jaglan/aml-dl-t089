"""Config-driven CNN training runner with a validation-macro-F1 checkpoint rule.

Training boundary: this module reads only official *training* data. Fit and
validation IDs come from the saved Phase 1 split files. Official test access is a
separate command (select.final_test) that requires an explicit freeze.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import load_official
from ..evaluate import label_metrics, prediction_frame, validate_predictions
from ..utils import ROOT, RESULTS_ROOT, read_json, sha256, stable_hash, write_json
from .data import AUGMENTATIONS, FashionIDDataset, make_loader, split_ids
from .models import build_model, parameter_count, shape_table

DEFAULTS = {
    "profile": "full", "blocks": 2, "dropout": 0.30, "augmentation": "none",
    "seed": 42, "device": "cpu", "batch_size": 128, "max_epochs": 25, "patience": 5,
    "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
    "loss": "CrossEntropyLoss", "selection_metric": "val_macro_f1", "run_order": 99,
}
# Keys that define the experiment. Device is recorded but is not part of the
# fingerprint, so the same config on another device is visible as an environment change.
FINGERPRINT_KEYS = ["profile", "blocks", "dropout", "augmentation", "seed", "batch_size",
                    "max_epochs", "patience", "optimizer", "loss", "selection_metric", "max_fit", "max_val"]


def load_config(path_or_dict, **overrides):
    config = dict(DEFAULTS)
    config.update(read_json(path_or_dict) if isinstance(path_or_dict, (str, Path)) else path_or_dict)
    config.update({k: v for k, v in overrides.items() if v is not None})
    if "name" not in config:
        raise ValueError("Config needs a 'name'")
    if config["augmentation"] not in AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation {config['augmentation']!r}")
    if config["optimizer"]["name"] != "AdamW" or config["loss"] != "CrossEntropyLoss":
        raise ValueError("The shared protocol fixes AdamW and cross-entropy")
    config["config_sha256"] = stable_hash({k: config.get(k) for k in FINGERPRINT_KEYS})
    config["run_id"] = f"{config['profile']}_{config['name']}_s{config['seed']}_{config['config_sha256'][:8]}"
    return config


def phase2_paths(profile):
    result = RESULTS_ROOT / "phase2" / profile
    artifact_root = ROOT / "artifacts" if RESULTS_ROOT == ROOT / "results" else ROOT / "artifacts" / stable_hash(str(RESULTS_ROOT))[:12]
    artifact = artifact_root / "phase2" / profile
    for directory in [result / "runs", artifact]:
        directory.mkdir(parents=True, exist_ok=True)
    return result, artifact


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def phase2_environment(device):
    packages = ["torch", "numpy", "pandas", "scikit-learn", "scikit-image"]
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "device": str(device), "torch_threads": torch.get_num_threads(),
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "git_commit": commit, "src_dirty": dirty,
            "phase2_source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "dataloader": "num_workers=0, seeded torch.Generator for shuffling and augmentation"}


def save_checkpoint_atomically(path, model, config, epoch, metrics):
    """Write to a temporary file, then atomically replace: no half-written best.pt."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "run_id": config["run_id"],
                "config_sha256": config["config_sha256"], "epoch": epoch,
                "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"]}, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu", expected_sha256=None):
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    bundle = torch.load(path, map_location=device, weights_only=True)
    model = build_model(bundle["arch"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n, preds, targets = 0.0, 0, [], []
    for x, y, _ids in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)  # batch mean -> example-weighted sum
        n += len(y)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(y.cpu())
    metrics = label_metrics(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    return {"loss": total_loss / n, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}


def evaluate(model, loader, criterion, device):
    """Aggregate over the whole partition before computing macro F1; keeps ID order."""
    model.eval()
    total_loss, n, ids, targets, preds = 0.0, 0, [], [], []
    with torch.inference_mode():
        for x, y, original_ids in loader:
            logits = model(x.to(device))
            total_loss += criterion(logits, y.to(device)).item() * len(y)
            n += len(y)
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y)
            ids.append(original_ids)
    ids, targets, preds = (torch.cat(t).numpy() for t in (ids, targets, preds))
    metrics = label_metrics(targets, preds)
    metrics["loss"] = total_loss / n
    return metrics, ids, targets, preds


def _subset(ids, labels, limit, seed):
    """Optional stratified cap for quick diagnostics; recorded in the fingerprint."""
    if not limit or limit >= len(ids):
        return ids
    from ..data import stratified_subset
    return np.sort(stratified_subset(ids, labels, int(limit), seed=seed))


def train_run(config, images=None, labels=None, log=print):
    """Train one config, keep the best validation-macro-F1 checkpoint, export predictions."""
    result, artifact = phase2_paths(config["profile"])
    run_id = config["run_id"]
    run_dir = result / "runs" / run_id
    record_path = run_dir / "record.json"
    checkpoint = artifact / f"{run_id}.pt"
    fit_ids, val_ids, split_sha = split_ids(config["profile"])
    fingerprint = stable_hash({"config_sha256": config["config_sha256"], "split_sha256": split_sha})
    if record_path.exists():
        previous = read_json(record_path)
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"{run_id}: existing record has a different config/split; refusing to overwrite")
        if previous.get("status") == "completed" and checkpoint.exists() and sha256(checkpoint) == previous["checkpoint_sha256"]:
            log(f"Reuse completed run {run_id}")
            return previous
        log(f"{run_id}: previous attempt was {previous.get('status')}; restarting from epoch 1 (same config and split)")
    if images is None:
        images, labels = load_official("train")
    fit_ids = _subset(fit_ids, labels, config.get("max_fit"), config["seed"])
    val_ids = _subset(val_ids, labels, config.get("max_val"), config["seed"])

    device = resolve_device(config["device"])
    seed_everything(config["seed"])
    model = build_model({"blocks": config["blocks"], "dropout": config["dropout"]}).to(device)
    fit_set = FashionIDDataset(images, labels, fit_ids, config["augmentation"], seed=config["seed"])
    val_set = FashionIDDataset(images, labels, val_ids, "none")
    fit_loader = make_loader(fit_set, config["batch_size"], shuffle=True, seed=config["seed"])
    val_loader = make_loader(val_set, 512, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    opt = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"])

    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    record = {"run_id": run_id, "name": config["name"], "status": "running", "fingerprint": fingerprint,
              "config": config, "config_sha256": config["config_sha256"], "split_file": f"data/{config['profile']}_indices.npz",
              "split_sha256": split_sha, "fit_n": len(fit_ids), "val_n": len(val_ids),
              "fit_ids_sha256": stable_hash(fit_ids.tolist()), "val_ids_sha256": stable_hash(val_ids.tolist()),
              "first_ids_to_labels": {"fit": [[int(i), int(labels[i])] for i in fit_ids[:5]],
                                      "val": [[int(i), int(labels[i])] for i in val_ids[:5]]},
              "parameters": parameter_count(model), "shape_table": shape_table(model.cpu()),
              "environment": phase2_environment(device), "started_utc": started, "warnings": []}
    model.to(device)
    write_json(run_dir / "config.json", config)
    write_json(record_path, record)
    log(f"Training {run_id}: {len(fit_ids)} fit / {len(val_ids)} val, {record['parameters']:,} params, device={device}")

    history, best_f1, best_epoch, stale, stop_epoch = [], float("-inf"), None, 0, None
    clock = time.perf_counter()
    try:
        for epoch in range(1, config["max_epochs"] + 1):
            fit_metrics = train_one_epoch(model, fit_loader, optimizer, criterion, device)
            val_metrics, *_ = evaluate(model, val_loader, criterion, device)
            current = val_metrics["macro_f1"]
            improved = current > best_f1  # strict: a tie keeps the earlier epoch
            if improved:
                best_f1, best_epoch, stale = current, epoch, 0
                save_checkpoint_atomically(checkpoint, model, config, epoch, val_metrics)
            else:
                stale += 1
            row = {"epoch": epoch, "fit_loss": fit_metrics["loss"], "fit_accuracy": fit_metrics["accuracy"],
                   "fit_macro_f1": fit_metrics["macro_f1"], "val_loss": val_metrics["loss"],
                   "val_accuracy": val_metrics["accuracy"], "val_macro_f1": current,
                   "elapsed_seconds": time.perf_counter() - clock, "best_epoch": best_epoch, "improved": improved}
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            log(f"  epoch {epoch:2d} fit_loss={row['fit_loss']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.4f} val_F1={current:.4f} best={best_epoch} {row['elapsed_seconds']:.0f}s")
            stop_epoch = epoch
            if stale >= config["patience"]:
                break
        stopped_early = stop_epoch < config["max_epochs"]

        # Reload the best file (not the in-memory last epoch) before exporting predictions.
        best_model, bundle = load_checkpoint(checkpoint, device)
        val_metrics, ids, targets, preds = evaluate(best_model, val_loader, criterion, device)
        if bundle["epoch"] != best_epoch or abs(val_metrics["macro_f1"] - best_f1) > 1e-12:
            record["warnings"].append("Reloaded best checkpoint did not reproduce the logged best validation macro F1")
        frame = prediction_frame(run_id, "val", ids, targets, preds)
        validate_predictions(frame, val_ids, labels, "val", run_id)
        pred_path = run_dir / "val_predictions.csv.gz"
        frame.to_csv(pred_path, index=False, compression={"method": "gzip", "mtime": 0})
        record.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                      train_seconds=time.perf_counter() - clock, best_epoch=best_epoch, stop_epoch=stop_epoch,
                      stopped_early=stopped_early, val_macro_f1=val_metrics["macro_f1"], val_accuracy=val_metrics["accuracy"],
                      val_metrics={k: v for k, v in val_metrics.items() if k != "loss"}, val_loss=val_metrics["loss"],
                      checkpoint_path=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha256(checkpoint),
                      checkpoint_bytes=checkpoint.stat().st_size, val_predictions_sha256=sha256(pred_path),
                      history=history)
        write_json(record_path, record)
        log(f"Done {run_id}: best epoch {best_epoch}, stopped {stop_epoch}, val F1={record['val_macro_f1']:.4f}, acc={record['val_accuracy']:.4f}")
        return record
    except BaseException as error:
        record.update(status="failed", exception=repr(error), history=history,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(record_path, record)
        raise


def verify_run(profile, run_id, device="cpu", log=print):
    """Fresh-process check: reload checkpoint, recompute validation predictions, compare to export."""
    result, _ = phase2_paths(profile)
    record = read_json(result / "runs" / run_id / "record.json")
    if record["status"] != "completed":
        raise ValueError(f"{run_id} is not completed")
    model, bundle = load_checkpoint(ROOT / record["checkpoint_path"], device, record["checkpoint_sha256"])
    images, labels = load_official("train")
    saved = pd.read_csv(result / "runs" / run_id / "val_predictions.csv.gz")
    ids = saved.sample_id.to_numpy()
    loader = make_loader(FashionIDDataset(images, labels, ids, "none"), 512, shuffle=False)
    metrics, out_ids, targets, preds = evaluate(model, loader, nn.CrossEntropyLoss(), torch.device(device))
    same = bool(np.array_equal(out_ids, ids) and np.array_equal(preds, saved.y_pred.to_numpy()))
    report = {"run_id": run_id, "checkpoint_sha256_verified": True, "checkpoint_epoch": bundle["epoch"],
              "predictions_identical": same, "n": int(len(ids)),
              "mismatches": int((preds != saved.y_pred.to_numpy()).sum()),
              "val_macro_f1": metrics["macro_f1"], "record_val_macro_f1": record["val_macro_f1"],
              "first_ids_to_labels": [[int(i), int(labels[i]), int(p)] for i, p in zip(ids[:5], preds[:5])],
              "verified_utc": datetime.now(timezone.utc).isoformat(), "device": device}
    write_json(result / "runs" / run_id / "reload_check.json", report)
    log(json.dumps(report, indent=2))
    if not same:
        raise ValueError(f"{run_id}: reloaded checkpoint changed {report['mismatches']} validation predictions")
    return report
