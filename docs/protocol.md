# Phase 1 protocol, declared before final test access

## Question and scope

How does a fixed local-edge representation (HOG) change classification compared
with raw intensities, across four classical classifiers? This phase does not train
a CNN, claim CNN superiority, or implement a hybrid ML/DL model.

## Data and IDs

- Use the checksum-verified official Fashion-MNIST IDX files.
- Keep the official test partition separate. The normal training/EDA commands
  do not download or load its content.
- Split official training 80/20, stratified by labels, random seed 42.
- `data/split_indices.npz`: 48,000 fit-pool / 12,000 validation original IDs.
- `data/budget_indices.npz`: a fixed stratified 12,000-example subset of the fit
  pool plus the unchanged 12,000 validation examples.
- `data/full_indices.npz`: the whole fit pool plus the same validation examples.
- Every final row in a comparison uses identical IDs and training size.
- A numeric ID is meaningful only together with `official_train`/`official_test`.

The smoke subset is for checking the program, not a main result. Budget tuning
is a separate development stage. If a full profile is produced, all eight models
are fitted afresh on the same 48,000 examples using the hyperparameters chosen
on budget validation. Scalers are also fitted afresh on those fit rows. No
validation labels enter any fit. The final overall model is selected from the
eight full-profile validation results before test evaluation.

## Representations and preprocessing

Both paths accept raw uint8 batches shaped `(N,28,28)`, preserve row order,
divide pixels by 255 once and use float32 fixed features.

- Pixels: reshape to 784 intensities, in a consistent recoverable order.
- HOG: orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2),
  block_norm='L2-Hys', transform_sqrt=False, channel_axis=None: 1,296 values.
- StandardScaler for LR, SVM and HOG kNN; passthrough for RF and pixel kNN.
- All learned standardization stays inside training Pipelines.
- The feature cache fingerprints raw input bytes, IDs, feature settings and
  scikit-image version. It contains no globally fitted scaler.

Flattening does not erase values or recoverable pixel positions. It does not
give these classifiers a built-in local-neighborhood inductive bias. HOG
explicitly summarizes local gradients but its extraction rule is not learned
from clothing labels. These statements motivate the comparison without
predetermining its winner.

## Search and freeze

The compact development grid has two candidates per row, 16 total:

| Family | Development candidates | Fixed settings |
|---|---|---|
| LR | C=0.1, 1 | lbfgs; max_iter=2,000 |
| kNN | k=3, 5 | distance weighting; Euclidean; brute force |
| RF | max_depth=20, None | 200 trees; leaf size 1; sqrt feature sampling |
| SVM | C=1, 10 | RBF; gamma='scale'; probability=False |

All stochastic settings with an exposed relevant seed use 42. Native numeric
threads and model n_jobs are 1 in recorded runs. Validation macro F1 is primary;
tie-breakers are accuracy, lower validation prediction time, then run ID.
Only completed, converged candidates can become finalists. Warnings and failures
are retained. This small grid supports a controlled baseline, not a claim of
optimal tuning.

`freeze.json` contains one finalist per row, exact parameters, serialized-model
hashes, split hash, validation metrics, the overall selected run ID and a
self-integrity hash. Final testing verifies the manifest and model hashes.

## Final evaluation and reporting

Use the full 10,000-image official test partition for every frozen finalist.
Save y_true/y_pred joined to original IDs, not just aggregate scores.
Validate identity, membership, category IDs, authoritative labels and stage.
Compute accuracy, macro F1 over labels 0-9, per-class recall/support and both
count and row-normalized confusion matrices.

Do not choose a new winner or retune from test scores. Paired test error panels
are descriptive after freeze. Each shows the first four original IDs in both
correct / only pixels correct / only HOG correct / both wrong groups, within
the same classifier. Store the selection IDs and group counts.

## Timing and complexity

Time already loaded models on one machine. Use fixed test IDs 0-1023 in batches
of 128, one warmup and five repeats under one native thread. Report median,
minimum and maximum amortized milliseconds per image. This does not measure
batch-size-one request latency. Disk reads and model loading are excluded.

1. Pipeline-only: cached unscaled fixed features -> scaler -> classifier.
2. End to end: raw image -> normalization -> pixels/HOG -> scaler -> classifier.

Record fit time separately. It includes scaler/classifier fitting, excludes
fixed feature extraction and excludes development search time. The feature
cache access/compute time is logged separately and explicitly may include cache hits.

LR has coefficient/intercept scalars; kNN stores feature rows; RF stores tree
nodes; RBF SVM stores support vectors and decision coefficients. Report those
different quantities as labeled complexity/storage proxies and give serialized
bytes under a common compression policy. They are not interchangeable counts
of neural-network-style parameters.

## Limits

One split, one seed, a modest search, fixed HOG settings and implementation-specific
scalers limit generalization of comparisons. Preprocessing differences mean the
results compare the declared feature pipelines, not a pure HOG ablation with
identical scalers everywhere. Exact hash checks do not find all near-duplicates;
that gap is now measured exhaustively in `data/near_duplicate_audit.json` and
reviewed in `docs/data_quality_review.md`. The test-side part of that audit is
post-freeze and descriptive: it reports a limitation of the published numbers and
is not grounds for reselection or re-tuning.
No generalization to arbitrary color catalogue photos or causal explanation of
errors is established. Any later model revision requires a new evaluation policy;
the already examined test set is no longer an untouched selection-independent exam.

## Provenance after figure review

The freeze preserves the source commit and module hashes used for model fitting.
Later presentation-only changes to plot legend placement and report/notebook layout
do not alter model, feature, data or evaluation code, frozen artifacts, predictions
or selection. The committed source history records that distinction. Do not rewrite
the freeze hash to hide later presentation work.

