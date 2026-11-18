# Team responsibilities and review record

| Member | Responsibility |
|---|---|
| Nakul | Integration, literature review, repository, master notebook and report assembly |
| Sudeshhna | Dataset quality, exploratory analysis and shared splits |
| Puneet | Pixel and HOG features, Logistic Regression and kNN |
| Samarth | Random Forest, SVM and experiment management |
| Neelanshu | Evaluation metrics, inference timing and error analysis |

These assignments describe responsibility. Add completed individual reviews and
changes to the log below with the relevant evidence.

## Preparation record

The initial implementation, experiment execution and document drafts were
prepared with Codex at Nakul's request. This record does not establish each
member's individual review or contribution.

| Member | Planned responsibility | Individual review/contribution status |
|---|---|---|
| Nakul | Integration, literature, repository, master notebook, report assembly | AI-assisted materials prepared; human review/viva practice pending |
| Sudeshhna | Data quality, EDA, shared splits; Phase 2 CNN data adapter and augmentation | Near-duplicate audit and Phase 2 data adapter/augmentation ablation contributed (AI-assisted, see log below); personal review and viva preparation pending |
| Puneet | Pixels/HOG, LR and kNN | Feature tests, kNN neighbour analysis and LR/kNN notebook contributed (AI-assisted, see log below); personal review and viva preparation pending |
| Samarth | RF, SVM and experiment harness; Phase 2 training runner, ablations, fusion | Phase 2 runner, ablations and fusion contributed (AI-assisted, see log below); personal review and viva preparation pending |
| Neelanshu | Metrics, timing and error analysis; Phase 2 evaluation | Phase 2 prediction audit, official-test run, paired error analysis and timing contributed (AI-assisted, see log below); personal review and viva preparation pending |

## Add real evidence as work happens

For each student add dated entries containing the file or experiment reviewed,
what was understood or changed, the verification performed, and real commit IDs
where applicable. Reviews and substantive explanations count as real work;
matching commit counts is not a meaningful goal.

Suggested entry format:

```text
Date:
Member:
File/run/figure:
Actual change or review:
Verification and finding:
Commit or evidence path:
```

## Recorded work

### 2026-10-28 — Sudeshhna — near-duplicate audit of the data layer

```text
Date: 2026-10-28
Member: Sudeshhna
File/run/figure: src/clothing_ml/near_duplicates.py, scripts/near_duplicate_audit.py,
  scripts/near_duplicate_figure.py, tests/test_near_duplicates.py,
  data/near_duplicate_audit.json, figures/eda/near_duplicates.png,
  docs/data_quality_review.md
Actual change or review: Reviewed the existing data layer, then added the
  exhaustive near-duplicate audit that data/training_audit.json explicitly did
  not claim and docs/protocol.md listed as an open limit. No existing module,
  split, frozen artifact or prediction file was modified.
Verification and finding: `python -m clothing_ml prepare` regenerates every
  committed data artifact byte-identically on a clean clone. The 80/20 split was
  re-derived from an independent IDX parser and split call; fit and validation
  membership match data/split_indices.npz exactly. All four official file MD5s
  match the published digests. Exact-duplicate counts in training_audit.json
  reproduce, and were cross-checked with np.unique: 60,000 unique training
  images and 10,000 unique test images.
  New measurement: five validation images sit within RMS 5 of a fit-pool image
  (closest pair official_train:10205 / official_train:23412, RMS 1.14), and six
  official test images sit within RMS 5 of a training image (closest pair
  official_test:4998 / official_train:13360, RMS 0.74, maximum pixel difference
  4). SHA-256 hashing cannot see these because the frames are not byte-identical.
  Impact is bounded and small: the validation-selected model
  full_hog_svm_3fd9d87a classifies all six correctly, and excluding them moves
  test accuracy from 90.9100% to 90.9045%.
  A float32 precision defect was found and fixed while writing the audit: in the
  expanded distance form, 784-dimensional 0-255 rows cancel badly enough that a
  byte-identical pair can report RMS above 0.2 and argmin can pick the wrong
  neighbour. Distances are accumulated in float64 and candidates below RMS 15 are
  recomputed in exact integer arithmetic; tests/test_near_duplicates.py pins this.
Commit or evidence path: branch phase1/data-quality-sudeshhna;
  data/near_duplicate_audit.json; docs/data_quality_review.md
AI assistance: Prepared with Claude, on the same transparent basis as the rest of
  this repository. The measurements are reproducible from the committed scripts.
  Sudeshhna's own review of this work, and her ability to explain it in the
  individual viva, are separate and are NOT claimed by this entry.
```

### 2026-10-28 — Puneet — features, Logistic Regression and kNN

```text
Date: 2026-10-28
Member: Puneet
File/run/figure: tests/test_features.py, scripts/knn_neighbours.py,
  figures/full/knn_neighbours_{hog,pixels}.png/.pdf, figures/full/knn_neighbours.json,
  scripts/build_features_notebook.py, notebooks/02_features_lr_knn.ipynb
Actual change or review: Reviewed src/clothing_ml/features.py, models.py and the
  LR/kNN path through experiments.py against the Phase 1 work guide. Added only
  new files; no source module, split, frozen record, prediction or test result
  was modified, so the recorded source-file hashes still match.
  - 16 feature/pipeline tests: HOG length derived from the declared settings,
    exact divide-by-255 ordering, batch-independent deterministic features,
    finite HOG on blank/saturated images, orientation sensitivity, shape and
    feature-name rejection, feature-cache tamper detection, LR/kNN scaling
    policy and the LR learned-scalar count 10(d+1).
  - kNN neighbour analysis of every validation error for both frozen kNN rows.
  - An executed walkthrough notebook for the four LR/kNN rows.
Verification and finding: All tests pass. Refitting both frozen kNN rows on the
  48,000 fit IDs reproduces all 12,000 saved validation predictions exactly.
  HOG kNN makes 1,493 validation errors; in 1,265 a majority of the five nearest
  fit images carry the predicted label and in 429 none carries the true label
  (pixels: 1,699 / 1,452 / 514), mostly among Shirt, T-shirt/top, Pullover and
  Coat. Refitting the frozen LR rows reproduces 11,949/12,000 (pixels) and
  11,983/12,000 (HOG) saved validation predictions, with macro F1 within 0.0002;
  the notebook reports this as near-complete rather than exact agreement.
  Within the same split and training size, HOG raised validation macro F1 by
  2.44 pp for LR and 1.64 pp for kNN. Official test data was not loaded.
Commit or evidence path: main; tests/test_features.py; figures/full/knn_neighbours.json;
  notebooks/02_features_lr_knn.ipynb
AI assistance: Prepared with Claude, on the same transparent basis as the rest of
  this repository. Puneet's own review of this work, and the ability to explain it
  in the individual viva, are separate and are NOT claimed by this entry.
```

### 2026-11-17 — Samarth — Phase 2 training runner, ablations and fusion

```text
Date: 2026-11-17
Member: Samarth
File/run/figure: src/clothing_ml/phase2/ (runner, select, fusion, plots, stand-in
  data/model interfaces), configs/phase2/*.json, tests/test_phase2_training.py,
  docs/phase2_training_protocol.md, docs/phase2_samarth_results.md,
  results/phase2/, figures/phase2/full/, artifacts/phase2/full/
Actual change or review: Added a config-driven CNN training runner (AdamW,
  cross-entropy, validation-macro-F1 checkpoint with earlier-epoch tie rule,
  patience 5, atomic writes, reload-before-export, full run record), a
  fresh-process reload check, validation-only selection, a one-time freeze and a
  separate --allow-test command, and the optional embedding/HOG fusion with two
  LR controls. New subpackage only: Phase 1 source hashes are unchanged.
  The loader and CNN are stand-ins following the agreed handoff contract until
  Sudeshhna's and Puneet's modules land.
Verification and finding: 36/36 tests pass. Smoke run and the selected CNN both
  reproduce all exported validation predictions after a fresh-process reload
  (1,000/1,000 and 12,000/12,000). Validation macro F1: cnn2 0.9267, no dropout
  0.9231, +-2 px translation 0.9202, CNN-3 0.9308 (selected). Fusion LR 0.9261 vs
  embedding-only LR 0.9319 and HOG-only LR 0.8832: fusion not supported. Official
  test not opened; four rows frozen (manifest 250f974a...).
Commit or evidence path: branch phase2/training-samarth; docs/phase2_samarth_results.md
AI assistance: Prepared with Claude, on the same transparent basis as the rest of
  this repository. Samarth's own review of this work, and the ability to explain it
  in the individual viva, are separate and are NOT claimed by this entry.
```

### 2026-11-18 — Puneet — Phase 2 SmallCNN architecture, tests, and feature interface

```text
Date: 2026-11-18
Member: Puneet
File/run/figure: src/clothing_ml/phase2/models.py, tests/test_phase2_models.py,
  scripts/phase2_model_demo.py, docs/phase2_cnn_architecture_puneet.md
Actual change or review: Designed, implemented, tested, and documented the Phase 2
  SmallCNN model architecture supporting depth=2 (CNN-2) and depth=3 (CNN-3),
  configurable dropout, and the 64-dimensional feature interface (embed/encode) for
  the ML+DL hybrid experiment.
Verification and finding:
  - Exact parameter arithmetic derived and asserted:
    CNN-2: 320 + 18,496 + 200,768 + 650 = 220,234 trainable parameters.
    CNN-3 (32, 64, 128 channels): 320 + 18,496 + 73,856 + 73,792 + 650 = 167,114
    trainable parameters; with (32, 64, 64 channels): 93,322 trainable parameters.
    In both, the third pooling layer downsamples spatial resolution to 3x3, shrinking the
    dense projection input and saving significant dense weights.
  - Automated unit tests passed: synthetic forward pass [B, 10], finite
    CrossEntropyLoss, non-zero Conv1 backprop gradients, embed/encode [B, 64],
    dropout=0.0 train/eval determinism, and atomic checkpoint roundtrip.
  - Live demo script verified on real Fashion-MNIST images from data/raw on CPU.
  - Complete handoff document with IEEE LaTeX report section prepared for Nakul.
Commit or evidence path: branch phase2/cnn-architecture-puneet; PR #6;
  docs/phase2_cnn_architecture_puneet.md; scripts/phase2_model_demo.py
AI assistance: Prepared on the same transparent basis as Phase 1. The code,
  tests, arithmetic proofs, and demo scripts are verified and reproducible.
```

### 2026-11-18 — Sudeshhna — Phase 2 CNN data adapter and augmentation ablation

```text
Date: 2026-11-18
Member: Sudeshhna
File/run/figure: src/clothing_ml/phase2/data.py, tests/test_phase2_data.py,
  scripts/phase2_data_audit.py, scripts/phase2_augmentation_report.py,
  configs/phase2/cnn2_aug_p50.json, results/phase2/data_audit/,
  results/phase2/full/runs/full_cnn2_aug_p50_s42_fe3c3cdd/,
  figures/phase2/data/, docs/phase2_sudeshhna_data.md
Actual change or review: Replaced the stand-in loader with the CNN data adapter
  (ClothingImages: raw uint8 -> float32 [1,28,28] divided by 255 once, original ID
  on every item), one loader factory (make_loaders), handoff batch checks
  (check_batch) and the work-guide fit-only translation policy (+-2 px, zero fill,
  no wraparound, p = 0.5). The runner-facing names are unchanged; select.py now
  restricts candidates to the declared run_order 1-4.
Verification and finding: 7 new tests pass (full suite green). The new adapter is
  bit-identical to the stand-in on full validation tensors and full shuffled fit
  epochs for both earlier policies; CNN-3 still reproduces 12,000/12,000 validation
  predictions after reload. Split audit: 4,800 fit / 1,200 validation per class,
  split SHA-256 matches Phase 1 provenance and every Phase 2 run; two validation
  passes identical; 125 wraparound cases pass. Validation-only ablation:
  p = 0.5 shift 0.9279 macro F1 vs 0.9267 unaugmented (+0.0012, one seed, not
  claimed as robust) vs 0.9202 shifting every image; Shirt recall 0.723 vs 0.772.
  Official test was not loaded.
Commit or evidence path: branch phase2/data-adapter-sudeshhna; docs/phase2_sudeshhna_data.md
AI assistance: Prepared with Claude, on the same transparent basis as the rest of
  this repository. Sudeshhna's own review of this work, and her ability to explain
  it in the individual viva, are separate and are NOT claimed by this entry.
```

Before submission, follow the faculty's actual AI-use/disclosure policy. That
policy has not been provided, so this repository records assistance transparently
without inventing a course rule or claiming permission from the faculty.

Follow any disclosure requirements specified by the faculty.

### 2026-11-19 — Neelanshu — Phase 2 evaluation, timing and paired errors

```text
Date: 2026-11-19
Member: Neelanshu
File/run/figure: src/clothing_ml/phase2/evaluation.py, tests/test_phase2_evaluation.py,
  results/phase2/full/evaluation/, figures/phase2/full/evaluation/,
  results/phase2/full/test_access_log.md, docs/phase2_neelanshu_evaluation.md
Actual change or review: Ran the frozen Phase 2 rows on the official test
  (freeze 250f974a…7a662). Fixed a one-line bug in select.final_test, which
  called .predict on the joblib dict instead of bundle["pipeline"]; the crash
  happened after the CNN row and before any summary was written. Added the
  evaluation module: prediction audit, one-row-per-run table, per-class table,
  paired outcome groups and label-pair counts, example panels, same-machine
  timing and complexity columns.
Verification and finding: 27 prediction files pass ID/label checks and their
  recomputed metrics equal the stored records. CNN-3 vs HOG + SVM on test:
  +1.23 pp accuracy (bootstrap 95% +0.73 to +1.75 pp), +0.0118 macro F1,
  425 only-CNN vs 302 only-SVM correct. Shirt recall falls 73.3% -> 71.7% while
  T-shirt -> Shirt errors drop 104 -> 64. The refit Phase 1 SVM matches all
  10,000 saved test predictions; on one M3 CPU thread it takes 19.7 ms/image
  vs 0.31 ms for the CNN. 43 tests pass.
Commit or evidence path: branch phase2/evaluation-neelanshu
```

### 2026-11-19 — Nakul — Phase 2 integration and final artifacts

```text
Date: 2026-11-19
Member: Nakul
File/run/figure: src/clothing_ml/phase2/reporting.py, scripts/build_phase2_*,
  notebooks/01_phase2_master.ipynb/.html, report/phase2/, presentation/phase2/,
  docs/phase2_literature_review.md, docs/phase2_integration_nakul.md
Actual change or review: Reviewed remote branch and PR ancestry; integrated
  Sudeshhna's outstanding data adapter (PR #8), retaining Puneet's model exports
  and both contribution logs. Added CPU PyTorch to CI. Assembled the Phase 2
  literature review, executed master notebook, IEEE report and 20-slide deck
  with short Hinglish presenter notes. Re-derived metrics independently from
  integer confusion counts and retained all original selection/test evidence.
Verification and finding: 60 tests pass. The integrated selected checkpoint
  matches 12,000/12,000 validation predictions. A fresh isolated two-epoch smoke
  training run reloads with 1,000/1,000 predictions matching. All 31 full-profile
  prediction files pass original-ID, authoritative-label and metric checks.
  Notebook executed in order with no errors; four-page public and private
  reports compiled and were visually reviewed. Deck has editable tables, five
  native charts with embedded data workbooks, and 20 Hinglish note parts.
  CNN improves aggregate accuracy by 1.23 percentage points over SVM, while
  Shirt recall falls. Fusion does not improve over embedding-only LR.
Commit or evidence path: PR #8; results/phase2/integration/audit.json;
  docs/phase2_integration_nakul.md; final artifact paths above.
```
