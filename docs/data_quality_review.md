# Data quality review: near-duplicate audit

Scope: the data-quality, EDA and split ownership described in the Phase 1 team
guides. This document records a review of the existing data layer and one new
measurement that closes a gap the repository previously declared open.

Prepared with AI assistance (Claude), on the same transparent basis as the rest
of the repository. See `docs/contributions.md`. The review findings below are
reproducible by anyone from the committed scripts; a student's personal
understanding of them is a separate thing and is recorded separately.

---

## 1. What was reviewed, and what held

The existing data layer was checked rather than assumed:

| Check | Method | Result |
|---|---|---|
| Committed artifacts reproduce | `python -m clothing_ml prepare` on a clean clone | `data/*.npz`, `provenance.json` and `training_audit.json` regenerate **byte-identically**; `git status` stays clean |
| Official files are authentic | MD5 against the digests published in the Fashion-MNIST repository | all four match |
| Split membership is not an artefact of one implementation | re-derived the 80/20 stratified split from an **independent** IDX parser and split call, then compared as sets | fit and validation membership **identical** to `data/split_indices.npz` (order differs only because the independent version sorts) |
| Split is disjoint and complete | set intersection; union against `0..59999` | disjoint, union exact, every id unique and in range |
| Stratification | per-class counts | exactly 4,800 fit and 1,200 validation in all ten categories; largest fit-vs-validation proportion gap 0.000000 |
| Audit claims in `training_audit.json` | recomputed shape, dtype, range, finiteness, class counts, SHA-256 duplicate grouping | every value reproduced |
| Exact duplicates | SHA-256 over raw image bytes, cross-checked with `np.unique` over the flattened array | 60,000 unique images from 60,000; the test partition likewise holds 10,000 unique images |
| Existing test suite | `python -m pytest -q` | 8 passed |

Nothing in the existing data layer needed correcting. `saved_split()` already
refuses to load a split that disagrees with the declared seed and protocol,
which is a stronger guarantee than a saved seed alone.

---

## 2. The gap this review closes

`data/training_audit.json` states plainly:

> "No near-duplicate or semantic relabeling audit claimed."

and `docs/protocol.md` lists under Limits:

> "Exact hash checks do not find all near-duplicates."

Both are honest, and both left the question unmeasured. Byte equality is the
wrong test on its own: a validation image that is a near-copy of a fit image
inflates validation scores exactly as an identical copy would, and hashing
cannot see it. The same applies between the training and test partitions,
where it would inflate the headline test number.

`scripts/near_duplicate_audit.py` now measures it **exhaustively** — every
validation image against all 48,000 fit images, and every test image against
all 60,000 training images. Results in `data/near_duplicate_audit.json`,
figure in `figures/eda/near_duplicates.png`.

### Method

Nearest-neighbour Euclidean distance in raw pixel space, reported as a
root-mean-square (RMS) pixel difference on the 0-255 scale, so the number
reads directly as "how far apart are these two pictures, per pixel". Every
candidate below RMS 15 is recomputed in exact integer arithmetic, so the
reported values carry no floating-point error.

Distances are accumulated in float64 deliberately. In float32 the expanded
form `|a|² + |b|² - 2ab` cancels catastrophically for 784-dimensional 0-255
rows — squared norms reach about 5×10⁷, and a byte-identical pair can report
an RMS difference above 0.2. That is precisely the regime this audit measures,
and it would also let `argmin` pick the wrong neighbour among close
candidates. `tests/test_near_duplicates.py` pins this behaviour.

### Results

| | validation vs fit pool | official test vs official training |
|---|---:|---:|
| comparisons | 12,000 × 48,000 | 10,000 × 60,000 |
| minimum RMS | **1.14** | **0.74** |
| 1st percentile | 14.50 | 14.12 |
| median | 31.76 | 31.54 |
| maximum | 88.54 | 82.47 |
| pairs below RMS 1 | 0 | 1 |
| pairs below RMS 2 | 2 | 1 |
| pairs below RMS 5 | **5** (0.042%) | **6** (0.060%) |
| pairs below RMS 10 | 17 | 18 |
| nearest neighbour shares the label | 85.05% | 84.97% |

**Finding 1 — near-duplicates exist inside the training partition.**
`official_train:10205` and `official_train:23412`, both coats, differ by
RMS 1.14 (343 of 784 pixels differ, maximum difference 5). One is in
validation and the other in the fit pool. SHA-256 hashing did not detect this
because the frames are not byte-identical. Four more such pairs exist below
RMS 5.

**Finding 2 — the official test partition contains near-copies of training
images.** `official_test:4998` and `official_train:13360`, both T-shirt/top,
differ by RMS 0.74 with a maximum pixel difference of 4. Six test images have
a training near-twin below RMS 5.

**Finding 3 — the effect on the reported result is negligible, and now
measured rather than assumed.** The validation-selected model
(`full_hog_svm_3fd9d87a`, HOG + RBF SVM) classifies all six of those test
images correctly. Removing them moves test accuracy from **90.9100%** to
**90.9045%** — 0.0055 percentage points, far below the differences the
comparison resolves. The headline number stands; it is now supported by a
measurement instead of by silence.

### What this does not establish

- Pixel distance only. It cannot detect the same product photographed at
  another angle, scale or exposure, nor semantically repeated catalogue items.
  No claim is made that the dataset is free of related examples.
- A small RMS difference is strong evidence of a near-identical frame, not
  proof that one source photograph was reused.
- The test-side comparison is a **post-freeze** audit. `results/full/freeze.json`
  and `test_receipt.json` predate it. It reports a limitation of numbers
  already published; it must not be used to reselect a model, drop hard test
  examples or re-tune anything. Nothing in this review changes a model,
  feature, split or prediction file.

---

## 3. Reproducing

```bash
python -m clothing_ml prepare            # regenerates and verifies the data layer
python scripts/near_duplicate_audit.py   # ~10 s, writes data/near_duplicate_audit.json
python scripts/near_duplicate_figure.py  # writes figures/eda/near_duplicates.{png,pdf}
python -m pytest -q                      # 12 tests
```

The audit retrains nothing and touches no frozen artifact.
