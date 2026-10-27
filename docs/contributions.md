# Contributions and AI assistance

## What is already prepared

At Nakul's request, Codex generated the initial code, automated checks, executed
experiments, documentation, notebook and report drafts. These are **AI-assisted
starting artifacts**. Their existence does not demonstrate that a student has
independently written, reviewed or understood them. No commits are fabricated
under teammates' identities, and no independent teammate work is claimed.

Nakul's current integration task can use these artifacts after review. All five
students still need to understand the method, check the evidence and record their
actual reviews and changes for the individual viva.

## Planned ownership from the team guides

| Member | Planned responsibility | Individual review/contribution status |
|---|---|---|
| Nakul | Integration, literature, repository, master notebook, report assembly | AI-assisted materials prepared; human review/viva practice pending |
| Sudeshhna | Data quality, EDA, shared splits | Near-duplicate audit contributed (AI-assisted, see log below); personal review and viva preparation pending |
| Puneet | Pixels/HOG, LR and kNN | Human review and own contribution pending |
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

Before submission, follow the faculty's actual AI-use/disclosure policy. That
policy has not been provided, so this repository records assistance transparently
without inventing a course rule or claiming permission from the faculty.

