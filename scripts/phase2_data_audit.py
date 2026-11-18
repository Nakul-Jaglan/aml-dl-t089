"""Phase 2 data audit and augmentation evidence (Sudeshhna). Reads official_train only.

    python scripts/phase2_data_audit.py

Writes results/phase2/data_audit/{data_audit.json, class_counts.csv, first_batch.csv,
transform_examples.csv} and figures/phase2/data/{example_grid, shift_comparison}.png/.pdf.
Official test images are never loaded.
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.phase2.data import (
    POLICIES, SPLIT_FILES, FashionIDDataset, RandomShift, check_batch, make_loader, make_loaders, split_ids, translate,
)
from clothing_ml.plots import save_figure, setup_style
from clothing_ml.utils import ROOT, read_json, write_json

SEED = 42
OUT = ROOT / "results/phase2/data_audit"
FIGURES = ROOT / "figures/phase2/data"
UPPER_BODY = [0, 2, 4, 6]  # T-shirt/top, Pullover, Coat, Shirt
FOOTWEAR = [5, 7, 9]  # Sandal, Sneaker, Ankle boot


def split_audit(labels):
    """Guide p.4 assertions plus per-class counts and digests for both saved profiles."""
    report = {}
    for profile in SPLIT_FILES:
        fit, val, digest = split_ids(profile)
        assert len(fit) == len(set(fit.tolist())) and len(val) == len(set(val.tolist()))
        assert not np.intersect1d(fit, val).size
        entry = {"split_file": SPLIT_FILES[profile], "split_sha256": digest, "fit_n": len(fit), "val_n": len(val),
                 "fit_per_class": np.bincount(labels[fit], minlength=10).tolist(),
                 "val_per_class": np.bincount(labels[val], minlength=10).tolist(),
                 "covers_all_official_train_rows": bool(np.array_equal(np.sort(np.r_[fit, val]), np.arange(len(labels))))}
        report[profile] = entry
    full_fit, full_val, _ = split_ids("full")
    assert len(full_fit) == 48_000 and len(full_val) == 12_000
    assert report["full"]["covers_all_official_train_rows"]
    assert Counter(labels[full_fit].tolist()) == {c: 4800 for c in range(10)}
    assert Counter(labels[full_val].tolist()) == {c: 1200 for c in range(10)}
    smoke_fit, smoke_val, _ = split_ids("smoke")
    assert set(smoke_fit.tolist()) <= set(full_fit.tolist()) and set(smoke_val.tolist()) <= set(full_val.tolist())
    report["smoke"]["subset_of_full"] = True
    return report


def provenance_check(split_report):
    """Compare the actual split digest with Phase 1 provenance and every Phase 2 run record."""
    full_sha = split_report["full"]["split_sha256"]
    smoke_sha = split_report["smoke"]["split_sha256"]
    provenance = read_json(ROOT / "data/provenance.json")
    phase1 = read_json(ROOT / "results/full/protocol.json")
    runs = {}
    for record_path in sorted((ROOT / "results/phase2").glob("*/runs/*/record.json")):
        record = read_json(record_path)
        expected = full_sha if record["config"]["profile"] == "full" else smoke_sha
        runs[record["run_id"]] = {"split_sha256": record["split_sha256"], "matches": record["split_sha256"] == expected}
    result = {"data_provenance_split_sha256": provenance["split_sha256"],
              "phase1_protocol_split_sha256": phase1["split_sha256"],
              "provenance_matches": provenance["split_sha256"] == full_sha == phase1["split_sha256"],
              "phase2_run_records": runs, "all_runs_match": all(r["matches"] for r in runs.values()),
              "published_md5_verified_by_loader": provenance["published_md5"]}
    assert result["provenance_matches"] and result["all_runs_match"], result
    return result


def first_batches(images, labels):
    """First fit and validation batch of the smoke and full loaders: shape, dtype, range, IDs."""
    rows, summaries = [], {}
    for profile in SPLIT_FILES:
        loaders, meta = make_loaders(profile, "none", SEED, images=images, labels=labels)
        for stage in ["fit", "val"]:
            batch = next(iter(loaders[stage]))
            summary = check_batch(batch, images, labels, canonical=True)
            summaries[f"{profile}_{stage}"] = summary
            x, y, ids = batch
            for position in range(5):
                original = int(ids[position])
                rows.append({"profile": profile, "stage": stage, "position": position, "source_split": "official_train",
                             "original_id": original, "label": int(y[position]), "class": CLASS_NAMES[int(y[position])],
                             "shape": "x".join(map(str, x[position].shape)), "dtype": str(x.dtype).replace("torch.", ""),
                             "min": round(float(x[position].min()), 6), "max": round(float(x[position].max()), 6),
                             "raw_min": int(images[original].min()), "raw_max": int(images[original].max()),
                             "matches_raw_over_255": bool(torch.equal(x[position, 0], torch.from_numpy(images[original]).float() / 255.0)),
                             "label_matches_raw": int(y[position]) == int(labels[original])})
        summaries[f"{profile}_meta"] = meta
    return pd.DataFrame(rows), summaries


def full_partition_check(images, labels):
    """Every validation batch passes the canonical checks, and two passes are identical."""
    loaders, _ = make_loaders("full", "none", SEED, images=images, labels=labels)
    passes = []
    for _ in range(2):
        xs, ys, ids = [], [], []
        for batch in loaders["val"]:
            check_batch(batch, images, labels, canonical=True)
            xs.append(batch[0]); ys.append(batch[1]); ids.append(batch[2])
        passes.append((torch.cat(xs), torch.cat(ys), torch.cat(ids)))
    (x1, y1, i1), (x2, y2, i2) = passes
    _, val_ids, _ = split_ids("full")
    fit_ids_seen = torch.cat([b[2] for b in loaders["fit"]]).numpy()
    fit_ids, _, _ = split_ids("full")
    result = {"val_batches": len(loaders["val"]), "val_n": int(len(i1)),
              "two_passes_identical_images": bool(torch.equal(x1, x2)),
              "two_passes_identical_labels": bool(torch.equal(y1, y2)),
              "two_passes_identical_ids": bool(torch.equal(i1, i2)),
              "val_order_equals_saved_order": bool(np.array_equal(i1.numpy(), val_ids)),
              "fit_epoch_covers_each_fit_id_once": bool(np.array_equal(np.sort(fit_ids_seen), np.sort(fit_ids))),
              "fit_epoch_is_shuffled": bool(not np.array_equal(fit_ids_seen, fit_ids)),
              "val_tensor_sha256": __import__("hashlib").sha256(x1.numpy().tobytes()).hexdigest()}
    assert all(v for k, v in result.items() if isinstance(v, bool)), result
    return result


def wraparound_check():
    """A white pixel near each edge must leave the frame, never reappear on the opposite side."""
    findings = []
    for name, (row, col) in {"top_left": (0, 0), "top_right": (0, 27), "bottom_left": (27, 0),
                             "bottom_right": (27, 27), "right_edge_mid": (14, 26)}.items():
        image = torch.zeros(1, 28, 28)
        image[0, row, col] = 1.0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                out = translate(image, dx, dy)
                new_row, new_col = row + dy, col + dx
                inside = 0 <= new_row < 28 and 0 <= new_col < 28
                ok = out.shape == (1, 28, 28) and float(out.min()) >= 0 and float(out.max()) <= 1
                ok &= float(out.sum()) == (1.0 if inside else 0.0)
                ok &= (not inside) or float(out[0, new_row, new_col]) == 1.0
                findings.append(ok)
    full = translate(torch.ones(1, 28, 28), 2, 2)
    border_zero = float(full[:, :2, :].sum()) == 0 and float(full[:, :, :2].sum()) == 0
    result = {"cases": len(findings), "all_passed": all(findings),
              "uncovered_border_is_zero": border_zero, "shape_after_shift": list(full.shape)}
    assert result["all_passed"] and border_zero
    return result


def policy_statistics(images, labels):
    """Empirical behaviour of the fit transform over one full fit epoch for each policy."""
    fit_ids, _, _ = split_ids("full")
    stats = {}
    for name, policy in POLICIES.items():
        if policy is None:
            continue
        dataset = FashionIDDataset(images, labels, fit_ids, name, seed=SEED)
        offsets = Counter()
        unchanged = 0
        for index in range(len(dataset)):
            dataset[index]
            offset = dataset.transform.last_offset
            if offset is None or offset == (0, 0):
                unchanged += 1
            offsets[str(offset)] += 1
        stats[name] = {"policy": policy, "fit_items": len(dataset),
                       "fraction_shift_drawn": 1 - offsets["None"] / len(dataset),
                       "fraction_left_unchanged": unchanged / len(dataset),
                       "offset_counts": dict(sorted(offsets.items()))}
    return stats


def seeded_examples(labels, ids, per_class, seed=SEED):
    rng = np.random.default_rng(seed)
    return {c: np.sort(rng.choice(ids[labels[ids] == c], per_class, replace=False)) for c in range(10)}


def example_grid(images, labels):
    fit_ids, _, _ = split_ids("full")
    chosen = seeded_examples(labels, fit_ids, 4)
    fig, axes = plt.subplots(4, 10, figsize=(13.5, 6.4))
    for c in range(10):
        for r, original in enumerate(chosen[c]):
            ax = axes[r, c]
            ax.imshow(images[original], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"train:{original}", fontsize=7)
            if r == 0:
                ax.set_title(f"{c} {CLASS_NAMES[c]}", fontsize=9)
    fig.suptitle("Seeded fit examples, four per category (seed 42, data/full_indices.npz)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, FIGURES, "example_grid")
    return {str(c): v.tolist() for c, v in chosen.items()}


def shift_comparison(images, labels):
    """Original vs shifted for upper-body classes and footwear; captions keep label and ID."""
    fit_ids, _, _ = split_ids("full")
    chosen = seeded_examples(labels, fit_ids, 1)
    classes = UPPER_BODY + FOOTWEAR
    shift = RandomShift(p=1.0, seed=SEED)  # forced draw for illustration; training uses p=0.5
    rows = []
    fig, axes = plt.subplots(3, len(classes), figsize=(13, 5.6))
    for column, c in enumerate(classes):
        original = int(chosen[c][0])
        canonical = torch.from_numpy(images[original]).float().unsqueeze(0) / 255.0
        shifted = shift(canonical)
        dx, dy = shift.last_offset
        if (dx, dy) == (0, 0):
            dx, dy = 2, 0
            shifted = translate(canonical, dx, dy)
        extreme = translate(canonical, 2 if dx >= 0 else -2, 2 if dy >= 0 else -2)
        lost = float(canonical.sum() - extreme.sum()) / max(float(canonical.sum()), 1e-9)
        panels = [(canonical, "original"), (shifted, f"dx={dx:+d}, dy={dy:+d}"),
                  (extreme, f"corner ({2 if dx >= 0 else -2:+d},{2 if dy >= 0 else -2:+d})")]
        for row, (tensor, caption) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(tensor[0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(caption, fontsize=7)
            if row == 0:
                ax.set_title(f"{CLASS_NAMES[c]}\ntrain:{original}", fontsize=8)
        rows.append({"original_id": original, "label": c, "class": CLASS_NAMES[c], "dx": dx, "dy": dy,
                     "corner_shift_mass_lost_fraction": round(lost, 4),
                     "shifted_min": float(shifted.min()), "shifted_max": float(shifted.max())})
    axes[0, 0].set_ylabel("canonical", fontsize=9)
    axes[1, 0].set_ylabel("sampled shift", fontsize=9)
    axes[2, 0].set_ylabel("largest shift", fontsize=9)
    fig.suptitle("Fit-only translation policy: zero fill, no wraparound, |dx|,|dy| <= 2 (applied with p = 0.5)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, FIGURES, "shift_comparison")
    return pd.DataFrame(rows)


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    images, labels = load_official("train")
    assert images.shape == (60_000, 28, 28) and images.dtype == np.uint8
    split_report = split_audit(labels)
    counts = pd.DataFrame({"label": range(10), "class": CLASS_NAMES,
                           "official_train": np.bincount(labels, minlength=10),
                           "fit": split_report["full"]["fit_per_class"], "val": split_report["full"]["val_per_class"],
                           "smoke_fit": split_report["smoke"]["fit_per_class"], "smoke_val": split_report["smoke"]["val_per_class"]})
    counts.to_csv(OUT / "class_counts.csv", index=False)
    batch_rows, batch_summaries = first_batches(images, labels)
    batch_rows.to_csv(OUT / "first_batch.csv", index=False)
    transform_rows = shift_comparison(images, labels)
    transform_rows.to_csv(OUT / "transform_examples.csv", index=False)
    report = {
        "purpose": "Phase 2 CNN data adapter audit and augmentation-policy evidence (Sudeshhna)",
        "official_test_loaded": False,
        "raw": {"shape": list(images.shape), "dtype": str(images.dtype), "pixel_min": int(images.min()),
                "pixel_max": int(images.max()), "class_names": CLASS_NAMES},
        "splits": split_report,
        "provenance": provenance_check(split_report),
        "first_batches": batch_summaries,
        "stable_evaluation": full_partition_check(images, labels),
        "wraparound": wraparound_check(),
        "policy_statistics": policy_statistics(images, labels),
        "example_grid_ids": example_grid(images, labels),
        "figures": ["figures/phase2/data/example_grid.png", "figures/phase2/data/shift_comparison.png"],
        "environment": {"torch": torch.__version__, "numpy": np.__version__},
    }
    write_json(OUT / "data_audit.json", report)
    print(batch_rows[batch_rows.profile == "smoke"].to_string(index=False))
    print(json.dumps({k: report[k] for k in ["raw", "stable_evaluation", "wraparound"]}, indent=2))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
