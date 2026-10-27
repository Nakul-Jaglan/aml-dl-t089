# Project 11: Automatic Clothing Category Recognition

**Inventory Image Tagging via Handcrafted Features and a Small CNN**

This repository currently implements **Phase 1: classical machine learning only**.
It compares flattened pixels and Histogram of Oriented Gradients (HOG) with
Logistic Regression, k nearest neighbors, Random Forest and an RBF SVM.
The CNN and classical-versus-CNN comparison are reserved for Phase 2.

Team: **Nakul, Sudeshhna, Puneet, Samarth and Neelanshu**.

The implementation is AI-assisted work prepared at Nakul's request. Planned
ownership does not establish anyone's completed individual contribution.
See `docs/contributions.md` for the distinction and `docs/START_HERE_NAKUL.md`
for the review and viva path.

## Executed Phase 1 results

All eight final models used the same **48,000 fit images**, 12,000 validation
images and official 10,000-image test partition. The model selected on validation
before test access was **HOG + RBF SVM**: **90.91% test accuracy** and **0.9090
macro F1**. Its lowest category recall was Shirt, at 73.3%.

| Model | Pixels accuracy | HOG accuracy | Pixels macro F1 | HOG macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 84.01% | 87.60% | 0.8394 | 0.8757 |
| kNN | 85.04% | 87.34% | 0.8505 | 0.8730 |
| Random Forest | 87.67% | 87.29% | 0.8753 | 0.8718 |
| RBF SVM | 89.28% | 90.91% | 0.8924 | 0.9090 |

HOG helps three families under the declared preprocessing policy and slightly
weakens RF here. These one-split differences do not establish statistical
significance or a universal winner. Inference and storage trade-offs are in the
notebook and report.

Start with the [executed master notebook](notebooks/00_phase1_master.ipynb), its
[HTML companion](notebooks/00_phase1_master.html), the [public report](report/main.pdf),
and [Nakul's review guide](docs/START_HERE_NAKUL.md). The report is an IEEE-format
draft for human review; final faculty formatting requirements remain unconfirmed.
The slide content is a [12-slide outline with speaker notes](docs/presentation_outline.md),
rather than an exported PowerPoint deck.

Section A and the five roll numbers were supplied by the team. Roll numbers live
only in git-ignored private author files and `report/submission.pdf`. Public
GitHub files and `report/main.pdf` omit those identifiers. See [report build notes](report/README.md).

## Quick start

Use Python 3.12 for the exact tested lock file. The package also supports compatible
Python 3.11+ environments when dependencies are resolved rather than locked.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ipykernel install --prefix .venv --name clothing-phase1 --display-name "Python 3 (Phase 1 environment)"
python -m clothing_ml prepare
python -m pytest -q
```

Windows PowerShell: activate with `.venv\Scripts\Activate.ps1` instead.
For the exact tested package versions, install `requirements-lock.txt` before
installing this package with `python -m pip install -e . --no-deps`.

## Reproduce experiments

All profiles preserve the official 10,000-example test partition. Validation
comes only from the official 60,000-example training partition. Original IDs,
stratification, seed 42 and file hashes are saved.

The checkout includes frozen evidence. **Do not delete or overwrite it to rerun.**
For a fresh reproduction, set an isolated result location before the commands below:

```bash
export CLOTHING_RESULTS_ROOT="$PWD/work/reproduction"
# Windows PowerShell: $env:CLOTHING_RESULTS_ROOT="$PWD/work/reproduction"
```

Leave this variable unset to inspect the checked-in results. Fitted artifacts for
isolated reproductions are stored in a separate local subdirectory automatically.

| Profile | Fit images per model | Validation images | Purpose |
|---|---:|---:|---|
| `smoke` | 2,000 | 1,000 | Interface checks; final testing is blocked |
| `budget` | 12,000 | 12,000 | Shared compute-bounded development and tuning |
| `full` | 48,000 | 12,000 | Full fit-partition comparison |

The remaining 36,000 fit-pool images are unused by the budget profile. They
are not part of validation or test. All models within a profile use identical IDs.

```bash
python -m clothing_ml train --profile smoke --search baseline
python -m clothing_ml freeze --profile smoke

python -m clothing_ml train --profile budget --search compact
python -m clothing_ml freeze --profile budget

# Transfer validation-selected hyperparameters, then fit all eight rows on 48k.
python -m clothing_ml train --profile full --selection-from budget
python -m clothing_ml freeze --profile full

# Run only after the intended final profile and overall model have been frozen.
python -m clothing_ml test --profile full --allow-test
python -m clothing_ml figures --profile full
```

Freezing records one model per representation/classifier and the overall
validation-selected model. Existing frozen runs and final test summaries are
protected from accidental overwrites. To inspect results again, reuse the
saved predictions. Do not retune against final test results.

## Experiment design

- Fixed divide-by-255 normalization; 784 pixel features or 1,296 HOG features.
- HOG: 9 orientation bins, 4x4-pixel cells, 2x2-cell blocks, L2-Hys normalization.
- Learned scalers are fitted only through training Pipelines. RF and pixel kNN
  do not use learned standardization; LR, SVM and HOG kNN do.
- Compact validation grids: LR C in {0.1, 1}; kNN k in {3, 5} with distance
  weighting; RF depth in {20, unrestricted} with 200 trees; RBF SVM C in {1, 10}
  with gamma='scale'. These are modest chosen grids, not exhaustive optimization.
- Select on validation macro F1, then accuracy, then validation prediction time,
  then run ID. A most-frequent dummy classifier provides a separate reference.
- Accuracy, macro F1, all ten class recalls, fit time, inference time and
  model-specific storage/complexity measures are saved.
- Final inference benchmark: 1,024 fixed test images, batch size 128, one warmup,
  five repeats, one native thread. Both Pipeline-only and raw-image-to-label
  time are measured. Values are amortized batch cost, not single-request latency.
- No fitting on validation/test rows, no test-based tuning, and no silent refit
  on the union of fit and validation.

## Repository map

```text
src/clothing_ml/       data, features, models, experiments, evaluation and plots
tests/                meaningful data/ID/leakage/final-test guard checks
data/                 saved split IDs, source checksums and training audit
notebooks/            executed master notebook and HTML companion
results/<profile>/    protocol, candidates, freeze, predictions and metrics
figures/              reproducible EDA, comparisons and error-analysis figures
data/near_duplicate_audit.json  exhaustive near-duplicate measurement (data review)
report/               LaTeX/IEEE-format report source and compiled PDF
docs/                 protocol, research notes, Nakul guide and contribution log
artifacts/            locally trained models (git-ignored; regenerate by command)
data/raw/             checksum-verified official downloads (git-ignored)
data/cache/           content-addressed fixed features (git-ignored)
```

Model artifacts, raw data, virtual environments and feature caches are deliberately
not uploaded to Git. Saved predictions, small metadata, figures and split IDs
allow the conclusions to be inspected without downloading large models.
Load Python model bundles only from a trusted source.

## Sources and limitations

Dataset: [official Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
See `docs/literature_review.md` and `report/references.bib` for primary references.
The course's Phase 1 rubric and individual-viva requirement are recorded in
`docs/course_instructions.md`.

This is a low-resolution grayscale benchmark. Its results do not establish
performance on arbitrary real catalogue photos. Exact duplicate checks do not
verify semantic labels, and they do not detect near-duplicates; an exhaustive
pixel-space near-duplicate audit ([`docs/data_quality_review.md`](docs/data_quality_review.md))
finds six test images with a training near-twin, which shifts test accuracy by
0.0055 percentage points. A single fixed
validation split and a small search do not establish statistically significant
superiority or globally optimal settings. See the executed notebook for observed
results and the final report for the supported conclusions.
