"""Portable figures generated from saved data and ID-aligned predictions."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog

from .data import CLASS_NAMES, load_official, saved_split
from .evaluate import validate_predictions
from .features import HOG_SETTINGS, normalized_images
from .utils import ROOT, RESULTS_ROOT, read_json, sha256, write_json

COLORS = {"pixels": "#285F99", "hog": "#B86C16"}
DISPLAY = {"lr": "Logistic Regression", "knn": "kNN", "rf": "Random Forest", "svm": "RBF SVM"}


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.labelcolor": "#243446"})


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_eda():
    setup_style()
    directory = ROOT / "figures/eda"
    images, labels = load_official("train")
    tr, va, _ = saved_split(labels)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    positions = np.arange(10)
    ax.bar(positions - .18, np.bincount(labels[tr], minlength=10), width=.36, color=COLORS["pixels"], label="Fit pool (48,000)")
    ax.bar(positions + .18, np.bincount(labels[va], minlength=10), width=.36, color=COLORS["hog"], label="Validation (12,000)", hatch="//")
    ax.set_xticks(positions, CLASS_NAMES, rotation=50, ha="right")
    ax.set_ylabel("Images per true category")
    ax.set_title("Category counts in the saved official-training split")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, directory, "class_counts")
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(5, 6, figsize=(10, 9))
    selected = []
    for cls in range(10):
        chosen = rng.choice(tr[labels[tr] == cls], 3, replace=False)
        for j, idx in enumerate(chosen):
            axis = axes[cls // 2, (cls % 2) * 3 + j]
            axis.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
            axis.set_title(f"{CLASS_NAMES[cls]}\nID {idx}", fontsize=9)
            axis.axis("off")
            selected.append({"sample_id": int(idx), "label": cls})
    fig.suptitle("Seeded examples from the fit pool (official training only)")
    fig.tight_layout()
    save_figure(fig, directory, "training_gallery")
    write_json(directory / "gallery_ids.json", selected)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for cls, axis in enumerate(axes.flat):
        axis.imshow(images[tr[labels[tr] == cls]].mean(axis=0), cmap="gray", vmin=0, vmax=255)
        axis.set_title(CLASS_NAMES[cls], fontsize=10)
        axis.axis("off")
    fig.suptitle("Mean images per category: fit pool, not real individual examples")
    fig.tight_layout()
    save_figure(fig, directory, "mean_images")
    one_id = int(tr[labels[tr] == 6][0])
    image = normalized_images(images[one_id:one_id + 1])[0]
    _, picture = hog(image, visualize=True, **HOG_SETTINGS)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Fit shirt, original ID {one_id}")
    axes[1].imshow(picture, cmap="gray")
    axes[1].set_title("Fixed HOG edge visualization")
    for axis in axes: axis.axis("off")
    fig.tight_layout()
    save_figure(fig, directory, "hog_example")
    return images, labels


def generate_figures(profile="budget"):
    generate_eda()
    result = RESULTS_ROOT / profile
    directory = ROOT / "figures" / profile if RESULTS_ROOT == ROOT / "results" else RESULTS_ROOT / profile / "figures"
    manifest = read_json(result / "freeze.json")
    summary = pd.read_csv(result / "test_summary.csv")
    train_n = int(summary.train_n.iloc[0])
    subtitle = f"Official test: 10,000 images; {train_n:,} fit images per model"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro F1"]):
        for feature, offset in [("pixels", -.18), ("hog", .18)]:
            values = [float(summary[(summary.feature == feature) & (summary.model == model)][metric].iloc[0]) for model in DISPLAY]
            bars = axis.barh(np.arange(4) + offset, values, height=.34, color=COLORS[feature], label=feature,
                             hatch="//" if feature == "hog" else None)
            for bar, value in zip(bars, values):
                axis.text(value + .015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
        axis.set_yticks(np.arange(4), list(DISPLAY.values()))
        axis.set_xlim(0, 1.08)
        axis.set_xlabel(title + " (fraction)")
        axis.set_title(title)
        axis.invert_yaxis()
    fig.suptitle("Pixels versus HOG across four classical classifiers\n" + subtitle)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, .88))
    save_figure(fig, directory, "model_comparison")
    recalls = pd.read_csv(result / "per_class_recall.csv")
    order = [r["run_id"] for r in manifest["finalists"]]
    matrix = recalls[recalls.stage == "test"].pivot(index="run_id", columns="label", values="recall").loc[order, range(10)]
    fig, axis = plt.subplots(figsize=(11, 5))
    im = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axis.set_xticks(range(10), CLASS_NAMES, rotation=50, ha="right")
    axis.set_yticks(range(8), [f"{r['feature']} + {DISPLAY[r['model']]}" for r in manifest["finalists"]])
    for i in range(8):
        for j in range(10):
            value = matrix.iloc[i, j]
            axis.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                      color="white" if value > .65 else "#17293D")
    axis.set_title("Per-class recall on official test data\n" + subtitle)
    fig.colorbar(im, ax=axis, label="Recall (fraction)")
    fig.tight_layout()
    save_figure(fig, directory, "per_class_recall")
    fig, axis = plt.subplots(figsize=(9, 5))
    display_rows = summary.set_index("run_id").loc[order]
    names = [f"{r['feature']} + {DISPLAY[r['model']]}" for r in manifest["finalists"]]
    bars = axis.barh(names, display_rows.end_to_end_ms_per_image, color=[COLORS[r["feature"]] for r in manifest["finalists"]])
    for bar, value in zip(bars, display_rows.end_to_end_ms_per_image):
        axis.text(value + display_rows.end_to_end_ms_per_image.max() * .015, bar.get_y() + .4, f"{value:.3f}", va="center", fontsize=9)
    axis.set_xlim(0, display_rows.end_to_end_ms_per_image.max() * 1.23)
    axis.set_xlabel("Median end-to-end milliseconds / image (amortized batch cost)")
    axis.set_title("Inference including fixed features, learned scaler and classifier\n1,024 fixed test images; batches of 128; 5 repeats; 1 native thread")
    axis.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, directory, "inference_time")
    images, labels = load_official("test")
    ids = np.arange(len(labels))
    predictions = {}
    for row in manifest["finalists"]:
        run_id = row["run_id"]
        frame = pd.read_csv(result / "predictions" / f"{run_id}_test.csv.gz")
        frame = validate_predictions(frame, ids, labels, "test", run_id)
        predictions[(row["feature"], row["model"])] = frame.y_pred.to_numpy()
        metrics = read_json(result / f"{run_id}_test_metrics.json")
        cm = np.asarray(metrics["confusion"])
        normalized = cm / cm.sum(axis=1, keepdims=True)
        for suffix, values, upper in [("counts", cm, cm.max()), ("recall", normalized, 1)]:
            fig, axis = plt.subplots(figsize=(8, 7))
            image = axis.imshow(values, cmap="Blues", vmin=0, vmax=upper)
            axis.set_xticks(range(10), CLASS_NAMES, rotation=50, ha="right")
            axis.set_yticks(range(10), CLASS_NAMES)
            axis.set_xlabel("Predicted category")
            axis.set_ylabel("True category")
            axis.set_title(f"{row['feature']} + {DISPLAY[row['model']]} | official test\n{train_n:,} fit images; " + ("raw counts" if suffix == "counts" else "row-normalized (diagonal = recall)"))
            for i in range(10):
                for j in range(10):
                    text = str(int(values[i, j])) if suffix == "counts" else f"{values[i,j]:.2f}"
                    axis.text(j, i, text, ha="center", va="center", fontsize=7,
                              color="white" if values[i, j] > .6 * upper else "#17293D")
            fig.colorbar(image, ax=axis, label="Count" if suffix == "counts" else "Fraction of true category")
            fig.tight_layout()
            save_figure(fig, directory, f"confusion_{row['feature']}_{row['model']}_{suffix}")
    selection_rows = []
    for model in DISPLAY:
        pixels, hog_pred = predictions[("pixels", model)], predictions[("hog", model)]
        pc, hc = pixels == labels, hog_pred == labels
        conditions = [("Both correct", pc & hc), ("Only pixels correct", pc & ~hc),
                      ("Only HOG correct", ~pc & hc), ("Both wrong", ~pc & ~hc)]
        fig, axes = plt.subplots(4, 4, figsize=(9.5, 10.5))
        for group_no, (group_name, mask) in enumerate(conditions):
            chosen = ids[mask][:4]  # smallest original IDs, deterministic and symmetric
            for col, axis in enumerate(axes[group_no]):
                axis.axis("off")
                if col < len(chosen):
                    idx = int(chosen[col])
                    axis.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
                    axis.set_title(f"ID {idx}; true: {CLASS_NAMES[labels[idx]]}\nP: {CLASS_NAMES[pixels[idx]]}\nH: {CLASS_NAMES[hog_pred[idx]]}", fontsize=8)
                    selection_rows.append({"model": model, "group": group_name, "group_count": int(mask.sum()),
                                           "sample_id": idx, "y_true": int(labels[idx]),
                                           "pixel_prediction": int(pixels[idx]), "hog_prediction": int(hog_pred[idx])})
                if col == 0:
                    axis.text(-.1, .5, f"{group_name}\nn={mask.sum():,}", transform=axis.transAxes,
                              rotation=90, ha="right", va="center", fontsize=9)
        fig.suptitle(f"Paired representation errors: {DISPLAY[model]}\nOfficial test; first four IDs per outcome group; descriptive after freeze")
        fig.tight_layout()
        save_figure(fig, directory, f"paired_errors_{model}")
    pd.DataFrame(selection_rows).to_csv(result / "error_example_ids.csv", index=False)
    write_json(directory / "figure_sources.json", {"test_summary": str((result / "test_summary.csv").relative_to(ROOT)),
                                                   "summary_sha256": sha256(result / "test_summary.csv"),
                                                   "freeze_sha256": manifest["manifest_sha256"],
                                                   "error_selection": "smallest original test IDs per paired outcome group"})
    print(f"Saved figures to {directory}", flush=True)

