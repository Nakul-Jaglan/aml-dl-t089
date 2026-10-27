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
| Sudeshhna | Data quality, EDA, shared splits | Near-duplicate audit contributed (AI-assisted, see log below); personal review and viva preparation pending |
| Puneet | Pixels/HOG, LR and kNN | Feature tests, kNN neighbour analysis and LR/kNN notebook contributed (AI-assisted, see log below); personal review and viva preparation pending |
| Samarth | RF, SVM and experiment harness | Human review and own contribution pending |
| Neelanshu | Metrics, timing and error analysis | Human review and own contribution pending |

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

Before submission, follow the faculty's actual AI-use/disclosure policy. That
policy has not been provided, so this repository records assistance transparently
without inventing a course rule or claiming permission from the faculty.

Follow any disclosure requirements specified by the faculty.
