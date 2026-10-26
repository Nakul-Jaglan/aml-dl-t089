"""Create the portable, evidence-checking master notebook using nbformat."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Clothing Category Recognition: Phase 1 master notebook

**Team:** Nakul, Sudeshhna, Puneet, Samarth and Neelanshu.

This executed notebook examines **pixels versus HOG with four classical classifiers**.
It reads saved experiment evidence and independently checks the reported metrics.
It does not retrain models or silently repeat final test selection when opened.

**AI assistance:** Code and initial analysis were prepared with Codex at Nakul's request.
Each student still needs to review the work, record real contributions and prepare for the individual viva.
The CNN and any hybrid approach belong to Phase 2.
""")
md("""
## Setup

Install the project with `python -m pip install -e '.[dev]'` from the repository root.
Use its virtual environment as your notebook kernel. The original data files are downloaded
from the official source if missing and checked against published checksums.
Saved test evidence must already exist before running this read-only notebook.
""")
code("""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from skimage.feature import hog
from clothing_ml.utils import ROOT, read_json, sha256
from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.features import HOG_SETTINGS, make_features, normalized_images
from clothing_ml.evaluate import label_metrics, validate_predictions
from clothing_ml.experiments import validate_manifest

PROFILE = "full"
RESULTS = ROOT / "results" / PROFILE
manifest = read_json(RESULTS / "freeze.json")
validate_manifest(manifest)
validation = pd.read_csv(RESULTS / "validation_summary.csv")
test = pd.read_csv(RESULTS / "test_summary.csv")
selected_id = manifest["selected_run_id"]
selected = test.set_index("run_id").loc[selected_id]
assert test.train_n.nunique() == 1 and len(test) == 8
plt.rcParams.update({"font.size": 10, "figure.figsize": (10, 4),
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.precision", 4)
print("Profile:", PROFILE, "| Fit rows per model:", int(test.train_n.iloc[0]))
print("Recorded Python/packages:", manifest["environment"]["python"], manifest["environment"]["packages"])
""")
md("## tl;dr")
code("""
display(Markdown(
    f"The model selected using **validation only** was **{selected.feature} + {selected.model.upper()}**. "
    f"On the official 10,000-image test partition, it achieved **{selected.accuracy:.2%} accuracy** "
    f"and **{selected.macro_f1:.4f} macro F1**. All eight final rows used "
    f"**{int(selected.train_n):,} identical fit examples** and the same validation/test partitions. "
    "The detailed comparison below includes weaker categories and inference costs; "
    "it does not establish performance on arbitrary catalogue photos or a future CNN."))
""")
md("""
## Context & Methods

The question is whether an explicit local-edge description changes classification
compared with raw intensities, and whether that effect depends on the classifier.

1. Official training data -> fixed stratified 48,000/12,000 fit/validation split, seed 42.
2. Development: two candidates per row on a saved 12,000-image subset of the fit pool;
   unchanged validation set. This is a modest search, not exhaustive optimization.
3. Final fitting: the eight development-selected configurations are fitted afresh on all
   48,000 fit images, including each learned scaler. Validation labels are never fitted.
4. Choose the overall final model using final validation macro F1 (then declared tie-breaks).
5. Freeze artifact hashes, then evaluate all eight on the official 10,000-image test partition.

### Key Assumptions

- The one fixed split is representative enough for this course comparison; no repeated-seed
  or cross-validation uncertainty analysis is claimed.
- Pixels and HOG are fixed transforms; learned standardization stays inside Pipelines.
- Preprocessing is intentionally model-specific. Representation comparisons are comparisons
  of the declared pipelines, not a pure HOG ablation under every possible scaler.
- Test error panels describe frozen models and do not authorize further test-driven tuning.
- Full details and primary sources: `docs/protocol.md`, `docs/literature_review.md`,
  `report/references.bib`. Original dataset: https://github.com/zalandoresearch/fashion-mnist.
""")
md("""
## Data

### 1. Verify the training audit and original IDs

An original image ID stays attached to its true label. The `source_split` disambiguates
the same numeric ID in official training versus official test. The audit below concerns
official training only; it did not inspect test content during development.
""")
code("""
train_images, train_labels = load_official("train")
audit = read_json(ROOT / "data/training_audit.json")
with np.load(ROOT / "data/full_indices.npz") as split:
    fit_ids, val_ids = split["train_idx"], split["val_idx"]
assert len(fit_ids) == 48000 and len(val_ids) == 12000
assert not np.intersect1d(fit_ids, val_ids).size
assert np.array_equal(np.sort(np.r_[fit_ids, val_ids]), np.arange(60000))
display(pd.DataFrame({"check": ["raw shape", "dtype", "pixel range", "finite", "all-zero images",
                               "exact duplicate groups", "conflicting duplicate groups", "fit/validation duplicate groups"],
                      "observed": [str(audit["shape"]), audit["dtype"],
                                   f"{audit['pixel_min']}..{audit['pixel_max']}", str(audit["finite"]),
                                   str(audit["all_zero_images"]), str(audit["exact_duplicate_groups"]),
                                   str(audit["conflicting_label_duplicate_groups"]), str(audit["fit_val_duplicate_groups"])]}))
""")
md("""
The exact-image hash audit does not detect all near-duplicates or prove every semantic label
is correct. No examples were removed or relabeled. Original sample membership is saved in `.npz` files.

### 2. Inspect seeded fit examples
""")
code("""
rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for category, ax in enumerate(axes.flat):
    image_id = int(rng.choice(fit_ids[train_labels[fit_ids] == category]))
    ax.imshow(train_images[image_id], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{CLASS_NAMES[category]}\\nID {image_id}", fontsize=10)
    ax.axis("off")
fig.suptitle("Seeded examples from the fit pool; no test exploration")
fig.tight_layout()
plt.show()
""")
md("### 3. Check class balance")
code("""
counts = pd.DataFrame({"category": CLASS_NAMES,
                       "fit_pool": np.bincount(train_labels[fit_ids], minlength=10),
                       "validation": np.bincount(train_labels[val_ids], minlength=10)})
display(counts)
ax = counts.set_index("category").plot.bar(color=["#285F99", "#B86C16"], rot=50, figsize=(10, 4))
ax.set_ylabel("Images per true category")
ax.set_title("Fixed official-training split: 48,000 fit / 12,000 validation")
ax.legend(["Fit pool", "Validation"], frameon=False)
plt.tight_layout()
plt.show()
""")
md("""
## Results

### 4. Inspect the two representations

Flattening preserves intensity values and their recoverable ordering but supplies no built-in
local-neighborhood bias. HOG summarizes local gradients using a fixed rule. For this setup:
7 cells per axis -> 6 overlapping blocks per axis -> 6 x 6 x 2 x 2 x 9 = **1,296** values.
HOG is not dimensionality reduction here.
""")
code("""
image_id = int(fit_ids[train_labels[fit_ids] == 6][0])
raw = train_images[image_id:image_id + 1]
pixels, hog_vector = make_features(raw, "pixels"), make_features(raw, "hog")
print("Pixel shape:", pixels.shape, "| HOG shape:", hog_vector.shape)
_, visualization = hog(normalized_images(raw)[0], visualize=True, **HOG_SETTINGS)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
axes[0].imshow(raw[0], cmap="gray", vmin=0, vmax=255)
axes[0].set_title(f"Fit shirt: ID {image_id}")
axes[1].imshow(visualization, cmap="gray")
axes[1].set_title("Fixed HOG visualization")
for ax in axes: ax.axis("off")
fig.tight_layout()
plt.show()
""")
md("""
### 5. Connect theory to the model factories

| Model | Decision rule | Development setting | Scaling |
|---|---|---|---|
| Logistic Regression | Learned linear class scores; softmax classification | C=0.1 or 1 | StandardScaler |
| kNN | Distance-weighted vote among stored neighbors | k=3 or 5 | Pixel passthrough; HOG standardized |
| Random Forest | Combine randomized tree predictions | depth=20 or unrestricted; 200 trees | Passthrough |
| RBF SVM | Nonlinear margin-based decisions | C=1 or 10; gamma='scale' | StandardScaler |

Smaller LR C means stronger regularization. kNN can fit quickly while retaining expensive
prediction/storage costs. RF samples training examples/features. SVC trains pairwise classifiers
for the ten classes. See the actual code in `src/clothing_ml/models.py`.
""")
code("""
development = pd.read_csv(ROOT / "results/budget/validation_candidates.csv")
display(development[["feature", "model", "train_n", "macro_f1", "accuracy", "fit_seconds", "converged"]])
display(Markdown("**Development only:** 12,000 fit examples per candidate. These scores are not mixed with full-profile or test results."))
display(pd.DataFrame([{"feature": r["feature"], "model": r["model"], "selected_parameters": r["parameters"]}
                      for r in manifest["finalists"]]))
""")
md("### 6. Read the full-profile validation decision")
code("""
display(validation.sort_values("macro_f1", ascending=False))
rule_order = sorted(manifest["finalists"], key=lambda r: (-r["macro_f1"], -r["accuracy"], r["validation_predict_seconds"], r["run_id"]))
assert rule_order[0]["run_id"] == selected_id
assert manifest["test_used_for_selection"] is False
display(Markdown(f"**Frozen overall model:** `{selected_id}`. This identifier was saved before loading test content."))
dummy = read_json(ROOT / "results/full/dummy_validation.json")
print("Most-frequent dummy validation accuracy/F1:", dummy["accuracy"], dummy["macro_f1"])
""")
md("""
### 7. Independently regenerate test metrics from predictions

Accuracy is total correct / N. Per-class recall is TP/(TP+FN). Macro F1 averages ten
separately calculated class F1 values equally; it is not F1 of averaged precision/recall.
Missing/duplicate sample IDs or incorrect true labels cause validation to fail.
""")
code("""
test_images, test_labels = load_official("test")
test_ids = np.arange(len(test_labels))
predictions = {}
regenerated = []
for row in manifest["finalists"]:
    run_id = row["run_id"]
    frame = pd.read_csv(RESULTS / "predictions" / f"{run_id}_test.csv.gz")
    frame = validate_predictions(frame, test_ids, test_labels, "test", run_id)
    metrics = label_metrics(frame.y_true, frame.y_pred)
    recorded = test.set_index("run_id").loc[run_id]
    assert np.isclose(metrics["accuracy"], recorded.accuracy, atol=1e-12)
    assert np.isclose(metrics["macro_f1"], recorded.macro_f1, atol=1e-12)
    predictions[(row["feature"], row["model"])] = frame.y_pred.to_numpy()
    regenerated.append({"feature": row["feature"], "model": row["model"],
                        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
display(pd.DataFrame(regenerated))
print("All eight test rows independently regenerated and ID-checked.")
""")
md("### 8. Compare representations within each classifier")
code("""
models = ["lr", "knn", "rf", "svm"]
names = ["Logistic Regression", "kNN", "Random Forest", "RBF SVM"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
    for feature, offset, color in [("pixels", -.18, "#285F99"), ("hog", .18, "#B86C16")]:
        values = [test[(test.feature == feature) & (test.model == m)][metric].iloc[0] for m in models]
        bars = ax.barh(np.arange(4) + offset, values, height=.34, color=color, label=feature,
                       hatch="//" if feature == "hog" else None)
        for bar, value in zip(bars, values):
            ax.text(value + .012, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metric + " (fraction)")
fig.suptitle(f"Official test: 10,000 images; {int(selected.train_n):,} fit examples per model")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, .12, 1, .90))
plt.show()
""")
code("""
deltas = []
for model in models:
    subset = test[test.model == model].set_index("feature")
    deltas.append({"model": model,
                   "accuracy_change_pp": 100*(subset.loc["hog", "accuracy"]-subset.loc["pixels", "accuracy"]),
                   "macro_f1_change_pp": 100*(subset.loc["hog", "macro_f1"]-subset.loc["pixels", "macro_f1"])})
display(pd.DataFrame(deltas))
display(Markdown("Positive values favor HOG. These are observed differences on one split, not statistical-significance or causal claims. The declared scaler choices are part of the pipelines."))
""")
md("### 9. Inspect every category's recall")
code("""
recall = pd.read_csv(RESULTS / "per_class_recall.csv")
order = [r["run_id"] for r in manifest["finalists"]]
recall_matrix = recall[recall.stage == "test"].pivot(index="run_id", columns="label", values="recall").loc[order, range(10)]
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(recall_matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(10), CLASS_NAMES, rotation=50, ha="right")
ax.set_yticks(range(8), [f"{r['feature']} + {r['model']}" for r in manifest["finalists"]])
for i in range(8):
    for j in range(10):
        value = recall_matrix.iloc[i,j]
        ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                color="white" if value > .65 else "#17293D")
ax.set_title("Per-class recall | official test | 1,000 actual images per category")
fig.colorbar(im, ax=ax, label="Recall (fraction)")
fig.tight_layout()
plt.show()
selected_recalls = recall[(recall.stage == "test") & (recall.run_id == selected_id)]
display(selected_recalls[["category", "support", "recall"]].sort_values("recall"))
""")
md("### 10. Explain the validation-selected model's confusion")
code("""
selected_predictions = predictions[(selected.feature, selected.model)]
selected_metrics = label_metrics(test_labels, selected_predictions)
cm = np.asarray(selected_metrics["confusion"])
row_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(row_norm, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(10), CLASS_NAMES, rotation=50, ha="right")
ax.set_yticks(range(10), CLASS_NAMES)
ax.set_xlabel("Predicted category")
ax.set_ylabel("True category")
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{row_norm[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if row_norm[i,j] > .65 else "#17293D")
ax.set_title(f"Validation-selected {selected.feature} + {selected.model}: official test\\nRow-normalized; diagonal equals per-class recall")
fig.colorbar(im, ax=ax, label="Fraction of actual category")
fig.tight_layout()
plt.show()
np.fill_diagonal(cm, 0)
top_cells = np.argsort(cm.ravel())[::-1][:5]
display(pd.DataFrame([{"true_category": CLASS_NAMES[k//10], "predicted_category": CLASS_NAMES[k%10],
                       "errors": int(cm.ravel()[k])} for k in top_cells]))
""")
md("""
### 11. See paired errors instead of selecting only favorable pictures

The following panel compares both representations **within SVM**, using the smallest
original IDs from each outcome group. The rule is symmetric and deterministic. All
four classifiers' complete example panels and selected IDs are saved under `figures/full/`
and `results/full/error_example_ids.csv`. Visual observations suggest possible explanations;
they do not prove a mechanism or establish that ambiguous labels are wrong.
""")
code("""
pixel_pred, hog_pred = predictions[("pixels", "svm")], predictions[("hog", "svm")]
pc, hc = pixel_pred == test_labels, hog_pred == test_labels
groups = [("Both correct", pc & hc), ("Only pixels correct", pc & ~hc),
           ("Only HOG correct", ~pc & hc), ("Both wrong", ~pc & ~hc)]
fig, axes = plt.subplots(4, 3, figsize=(9, 10))
for i, (label, mask) in enumerate(groups):
    chosen = test_ids[mask][:3]
    for j, ax in enumerate(axes[i]):
        ax.axis("off")
        if j < len(chosen):
            k = chosen[j]
            ax.imshow(test_images[k], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{label}; ID {k}\\nTrue: {CLASS_NAMES[test_labels[k]]}\\nPixels: {CLASS_NAMES[pixel_pred[k]]}; HOG: {CLASS_NAMES[hog_pred[k]]}", fontsize=8)
fig.suptitle("Paired RBF SVM examples: descriptive after model freeze")
fig.tight_layout()
plt.show()
display(pd.DataFrame({"group": [g[0] for g in groups], "test_images": [int(g[1].sum()) for g in groups]}))
""")
md("""
### 12. Read fit time, inference time and complexity honestly

The benchmark starts with loaded models and in-memory data. End-to-end includes normalization,
fixed feature extraction, scaler and classification; Pipeline-only begins at cached fixed features.
Both exclude disk/model-loading time. Five repeats of 1,024 fixed test images in batches of 128
produce an **amortized batch cost**, not single-request latency.

Counts below describe different structures. kNN stored scalars, RF nodes and SVM support-vector
storage are not interchangeable with neural-network-style trainable weights.
""")
code("""
display(test[["feature", "model", "fit_seconds", "predict_ms_per_image", "end_to_end_ms_per_image", "serialized_bytes"]])
complexities = read_json(RESULTS / "complexity.json")
display(pd.DataFrame([{k:r[k] for k in ["run_id", "complexity_type", "complexity_value", "serialized_bytes"]} for r in complexities]))
timing = read_json(RESULTS / "timing.json")
print("Timing machine:", timing["environment"]["platform"], timing["environment"]["machine"])
print("Protocol:", timing["measurements"][0]["end_to_end"])
""")
md("""
### 13. A reproducible label demonstration

This demonstration uses saved frozen predictions, so it works even when the large model
bundles have not been downloaded or regenerated. After local training, a new raw-image
prediction is `bundle['pipeline'].predict(make_features(raw_batch, bundle['feature']))`.
Only load trusted joblib bundles. An arbitrary uploaded photo also needs an evaluated
preprocessing/domain strategy; this benchmark does not make it production-ready.
""")
code("""
sample_id = 0
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(test_images[sample_id], cmap="gray", vmin=0, vmax=255)
ax.axis("off")
ax.set_title(f"Test ID {sample_id}\\nTrue: {CLASS_NAMES[test_labels[sample_id]]}\\nFrozen prediction: {CLASS_NAMES[selected_predictions[sample_id]]}")
plt.show()
""")
md("## Takeaways")
code("""
weak = selected_recalls.sort_values("recall").iloc[0]
gains = pd.DataFrame(deltas)
display(Markdown(
    f"1. The predeclared validation-selected model achieved **{selected.accuracy:.2%} test accuracy** "
    f"and **{selected.macro_f1:.4f} macro F1**.\\n"
    f"2. Its weakest category recall was **{weak['category']}: {weak['recall']:.1%}**. "
    "Aggregate accuracy therefore does not describe every category equally.\\n"
    "3. HOG's effect must be assessed within each classifier using the delta table, "
    "together with preprocessing, runtime and storage.\\n"
    "4. This is a one-split benchmark comparison. Repeated seeds, wider tuning and a "
    "true catalogue-image holdout are limitations/future work.\\n"
    "5. A CNN is a motivated **Phase 2 hypothesis**, not a measured result in this notebook."))
""")
md("""
## Reproduction and individual review

See README for commands to reproduce training in an isolated `CLOTHING_RESULTS_ROOT`.
The default notebook examines checked-in evidence; it makes no new selection decision.
The model artifacts and raw data are excluded from Git, while split IDs, predictions,
configuration hashes, figures and recorded results are retained.

Nakul should be able to trace each module, reproduce one metric from its CSV, explain
the frozen selection and identify a failure. All five members must add real review or
implementation contributions to `docs/contributions.md` and prepare for individual viva.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"display_name": "Python 3 (Phase 1 environment)", "language": "python", "name": "clothing-phase1"}
notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
target = ROOT / "notebooks/00_phase1_master.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(notebook)
nbf.write(notebook, target)
print(target)
