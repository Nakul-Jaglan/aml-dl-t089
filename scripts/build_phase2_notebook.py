"""Build Nakul's Phase 2 master notebook from frozen, reviewed evidence."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''
# Clothing Category Recognition - Phase 2 master notebook

**Nakul, Sudeshhna, Puneet, Samarth and Neelanshu, Section A.**

The completed Phase 2 comparison examines a small CNN, the Phase 1 HOG + SVM
reference, and HOG/CNN-feature fusion with matched Logistic Regression controls.
This notebook reads saved predictions, checks their original IDs and labels,
and derives the main reported scores from integer confusion counts.

Install the repository with `python -m pip install -e '.[dev,phase2]'` and use its
Python environment as the notebook kernel. Downloaded official IDX files are
checksum verified. Read the results with the shared methodology in
`docs/phase2_training_protocol.md` and `docs/phase2_integration_nakul.md`.
''')
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.utils import ROOT, read_json, sha256
from clothing_ml.phase2 import reporting as review

plt.rcParams.update({"font.size": 10, "figure.figsize": (10, 4),
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.precision", 4)
pd.set_option("display.max_colwidth", 62)
evidence = review.reviewed_evidence()
comparison = evidence["comparison"]
cnn = comparison.set_index("row").loc["selected_cnn"]
svm = comparison.set_index("row").loc["hog_svm"]
print("Prediction files independently checked:", evidence["audit_count"])
print("Saved split SHA-256:", evidence["freeze"]["split_sha256"])
''')
md('## tl;dr')
code('''
display(Markdown(
    f"The validation-selected **CNN-3** achieved **{cnn.test_accuracy:.2%} official-test accuracy** "
    f"and **{cnn.test_macro_f1:.4f} macro F1**, compared with **{svm.test_accuracy:.2%}** and "
    f"**{svm.test_macro_f1:.4f}** for HOG + SVM. The accuracy gain is "
    f"**{100*(cnn.test_accuracy-svm.test_accuracy):+.2f} percentage points**. "
    f"**Shirt recall fell from {svm.shirt_recall:.1%} to {cnn.shirt_recall:.1%}.** "
    "Fusion scored below embedding-only LR on validation. "
    f"Recorded same-machine CPU batch cost was **{cnn.ms_per_image:.3f} ms/image** for the CNN "
    f"and **{svm.ms_per_image:.3f} ms/image** for HOG + SVM."))
''')
md('''
## Context & Methods

Four seed-42 configurations trained on identical 48,000 fit IDs. Validation on
12,000 original training images selected the CNN by macro F1, then accuracy,
then declared run order. Each run kept its highest validation-F1 checkpoint.
AdamW used LR 0.001 and weight decay 0.0001. Cross-entropy accepted logits.
Batch size was 128, the epoch cap was 25, and patience was 5.

The selected CNN and three LR feature heads entered a hashed freeze before
Phase 2 official-test evaluation. Each LR head compared C=0.1 and 1 on the
same fit and validation IDs. The later 50%-translation run stayed a
validation-only ablation outside the frozen selection set.

### Key assumptions and limits

- The official test was already examined in Phase 1. These scores are a
  comparison on the same public benchmark. Phase 2 configurations used validation.
- One training seed and one split limit evidence about training variability.
- A paired bootstrap over test-ID pairs describes benchmark-sample variability.
- Original timing used one Apple M3 CPU and includes input conversion and features.
- The frozen encoder was supervised by fit labels and selected on validation.
  LR-head validation uses those same validation IDs, so it is a development
  comparison with selection reuse. The held-out test provides the final comparison.
- See `results/phase2/full/test_access_log.md` for the fixed-row test attempts and
  the artifact-loading repair. Source revisions and freeze history remain recorded.
''')
md('## Data\n\n### Original IDs and class balance')
code('''
train_images, train_labels = load_official("train")
with np.load(ROOT / "data/full_indices.npz") as split:
    fit_ids, val_ids = split["train_idx"], split["val_idx"]
assert not np.intersect1d(fit_ids, val_ids).size
assert len(fit_ids) == 48000 and len(val_ids) == 12000
counts = pd.DataFrame({"Category": CLASS_NAMES,
                      "Fit": np.bincount(train_labels[fit_ids], minlength=10),
                      "Validation": np.bincount(train_labels[val_ids], minlength=10)})
display(counts)
print("Official training shape:", train_images.shape, train_images.dtype)
''')
code('''
fig, axes = plt.subplots(2, 5, figsize=(10, 5.5))
for label, axis in enumerate(axes.flat):
    sample_id = int(fit_ids[train_labels[fit_ids] == label][0])
    axis.imshow(train_images[sample_id], cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"{CLASS_NAMES[label]}\\ntraining ID {sample_id}", fontsize=10)
    axis.axis("off")
fig.suptitle("Fit-pool examples with original training IDs")
fig.subplots_adjust(hspace=.55, wspace=.12, top=.85, bottom=.02)
plt.show()
''')
md('''
The recorded near-duplicate audit found five validation images and six official
test images within pixel RMS 5 of an image in the respective fit/training pool.
The headline results use the full official partitions. The pixel-distance audit
has limited coverage of semantically related garments under different views.

### CNN architecture

The **trained** three-block model uses channels **32, 64, 128**. Its third pool
reduces a 7x7 map to 3x3. The resulting smaller dense input yields **167,114**
trainable parameters. CNN-2 has **220,234**. The earlier work guide also proposed
a 64-channel third-block variant with 93,322 parameters. The results below refer
to the actual 128-channel model stored with the checkpoints.
''')
code('''
record = evidence["cnn_records"][review.CNN_ID]
display(pd.DataFrame(record["shape_table"]))
print("CNN-3 parameter count:", record["parameters"])
print("Best epoch:", record["best_epoch"], "| Final epoch:", record["stop_epoch"])
print("Checkpoint SHA-256:", record["checkpoint_sha256"])
''')
md('## Results\n\n### Learning curves and checkpoint selection')
code('''
review.learning_plot(evidence)
plt.show()
''')
md('''
Epoch 22 had the highest validation macro F1. Training continued to the cap at
25 and then reloaded that checkpoint. The loss-minimizing epoch differs because
checkpoint selection uses macro F1. The checkpoint reload reproduced every one
of the 12,000 saved validation predictions after the data-adapter integration.

### Controlled validation comparisons
''')
code('''
candidates = evidence["cnn_candidates"]
display(candidates[["name", "parameters", "val_accuracy", "val_macro_f1",
                    "shirt_recall", "best_epoch", "stop_epoch", "selectable"]])
review.ablation_plot(evidence)
plt.show()
''')
md('''
Against CNN-2, removing dropout reduced validation F1 by 0.0036. Always shifting
reduced it by 0.0065. The three-block architecture increased it by 0.0041 while
also changing the final spatial map and dense-head size. The later 50%-shift
ablation increased F1 by 0.0012 but reduced Shirt recall by about 4.8 percentage
points. These are one-seed observations under a shared checkpoint rule.

### Frozen official-test comparison
''')
code('''
display(comparison.rename(columns={"test_accuracy": "Test accuracy", "test_macro_f1": "Test macro F1",
                                  "shirt_recall": "Shirt recall", "ms_per_image": "CPU ms/image"}))
review.comparison_plot(evidence)
plt.show()
''')
md('''
CNN-3 reaches 92.14% accuracy and 0.9207 macro F1. The embedding-only LR control
scores 92.21% and 0.9218. It entered testing as a predeclared control and has the
highest aggregate score among these frozen Phase 2 rows. CNN-3 remains the
CNN selected by the original validation rule.

### Fusion and its single-feature controls
''')
code('''
display(evidence["fusion_candidates"][["row", "C", "feature_dim", "val_macro_f1", "converged"]])
review.hybrid_plot(evidence)
plt.show()
''')
md('''
Fusion joins 64 CNN embedding values with 1,296 HOG values and fits LR after
training-only standardization. Its validation macro F1 is 0.9261 versus 0.9319
for embedding-only LR. Official-test F1 is 0.9206 versus 0.9218. The controls
provide no support for an aggregate fusion improvement in this experiment.
Fusion's 76.0% test Shirt recall is an exploratory class trade-off.

### Category recall and paired mistakes
''')
code('''
review.recall_plot(evidence)
plt.show()
per_class = evidence["per_class"]
shirt = per_class[(per_class.label == 6) & (per_class.stage == "test") &
                  per_class.run_id.isin([review.CNN_ID, review.SVM_ID])]
display(shirt[["run_id", "precision", "recall", "f1", "support"]])
''')
code('''
cnn_pred = evidence["predictions"][review.CNN_ID, "test"].sort_values("sample_id")
svm_pred = evidence["predictions"][review.SVM_ID, "test"].sort_values("sample_id")
assert np.array_equal(cnn_pred.sample_id, svm_pred.sample_id)
truth = cnn_pred.y_true.to_numpy()
cnn_ok = cnn_pred.y_pred.to_numpy() == truth
svm_ok = svm_pred.y_pred.to_numpy() == truth
paired = pd.DataFrame({"Outcome": ["Both correct", "Only CNN correct", "Only SVM correct", "Both wrong"],
                       "Count": [(cnn_ok & svm_ok).sum(), (cnn_ok & ~svm_ok).sum(),
                                 (~cnn_ok & svm_ok).sum(), (~cnn_ok & ~svm_ok).sum()]})
assert paired.Count.sum() == 10000
assert int(paired.Count.iloc[1]-paired.Count.iloc[2]) == 123
display(paired)
review.error_plot(evidence)
plt.show()
''')
md('''
The CNN corrects 425 images the SVM misses and loses 302 the SVM gets right.
The difference is 123 additional correct labels. For true Shirt, the CNN sends
134 images to T-shirt/top versus the SVM's 103. In the reverse direction, the
CNN sends 64 T-shirts to Shirt versus 104 for SVM. Shirt precision rises while
recall falls. A visible neckline or fastening suggests an explanation to
investigate, with the recorded labels and predictions as direct observations.

### Uncertainty and inference cost
''')
code('''
uncertainty = read_json(review.FULL / "evaluation/cnn_vs_svm_deltas.json")
bootstrap = uncertainty["paired_bootstrap"]
display(pd.DataFrame({"Metric": ["Accuracy gain (pp)", "Macro F1 gain"],
                      "Estimate": [100*(cnn.test_accuracy-svm.test_accuracy), cnn.test_macro_f1-svm.test_macro_f1],
                      "95% lower": [100*bootstrap["accuracy_delta_ci95"][0], bootstrap["macro_f1_delta_ci95"][0]],
                      "95% upper": [100*bootstrap["accuracy_delta_ci95"][1], bootstrap["macro_f1_delta_ci95"][1]]}))
timing = evidence["timing"]
display(pd.DataFrame([{"Pipeline": name, **row["batch128_amortized"]}
                      for name,row in timing["measurements"].items()])[
    ["Pipeline", "median_ms_per_image", "min_ms_per_image", "max_ms_per_image", "images", "batch_size", "repeats"]])
print("Recorded CPU/environment:", timing["environment"])
''')
md('''
The bootstrap used 2,000 aligned-pair resamples and seed 42. Its accuracy-gain
interval is +0.73 to +1.75 percentage points. This interval covers sampling
variability of the benchmark images. Training-seed variability remains unmeasured.

The recorded CPU timer used 1,024 original test IDs, batches of 128, one warm-up
and five repeats under one PyTorch/native thread. It includes input conversion,
feature extraction and prediction, with loaded models. CNN-3 costs 0.312 ms/image
versus 19.748 for HOG + SVM, about 63.3 times lower on that machine. The SVM used
a recorded deterministic refit that matched all 10,000 original predictions.
''')
md('''
## Takeaways

- Learned features improve aggregate metrics under the recorded benchmark protocol.
- Shirt remains difficult. The CNN's precision/recall change shifts the class boundary.
- Embedding-only LR is a strong frozen control. Adding HOG reduces validation F1.
- Dropout, depth and augmentation need measured controls and careful scope.
- One seed, the examined public test set, near-copy findings and one-machine timing
  limit claims about another training run or production catalogue photographs.

### Saved evidence and handoff

`results/phase2/integration/audit.json` records the expanded 31-file audit.
`docs/phase2_integration_nakul.md` links the integration protocol and final artifacts.
`report/phase2/` contains the IEEE/LaTeX sources and compiled report.
The PowerPoint uses short Hinglish presenter notes, matching the team's Phase 1 preference.
''')

for index, description in {7: 'Ten fit-pool clothing images, one per category, with original training IDs.', 11: 'Fit and validation loss over 25 epochs; validation macro F1 peaks at epoch 22.', 13: 'Validation macro F1 differences from CNN-2: no dropout and always-shift are lower; CNN-3 is higher.', 15: 'Official-test accuracy and macro F1 for the SVM, CNN, and three LR feature controls.', 17: 'Fusion validation macro F1 is below the embedding-only LR control.', 19: 'Class recall for SVM and CNN; Shirt recall falls from 73.3 percent to 71.7 percent.', 20: 'Paired examples at test IDs 40, 17 and 25: CNN-only correct, SVM-only correct, and both wrong.'}.items():
    cells[index].metadata["alt"] = description

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name":"Python 3 (Phase 2)", "language":"python", "name":"python3"}
nbf.validate(notebook)
path = ROOT / "notebooks/01_phase2_master.ipynb"
nbf.write(notebook, path)
print(f"Wrote {path.name} ({len(cells)} cells)")
