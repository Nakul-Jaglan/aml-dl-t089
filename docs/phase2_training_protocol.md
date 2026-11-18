# Phase 2 training protocol (training runner, ablations, fusion)

Owner: Samarth. Versioned with the code; any change after this file is
committed is a new protocol revision and must say why. Shared decisions
(final test rows, report wording) remain with Nakul's team protocol.

## Data and IDs

- Official Fashion-MNIST, checksum-verified by the Phase 1 loader.
- `data/full_indices.npz`: 48,000 fit IDs and 12,000 validation IDs, all from
  `official_train`. `data/smoke_indices.npz`: 2,000 / 1,000 stratified subsets.
- Category IDs 0-9 unchanged. Every prediction row keeps `source_split` and the
  original row ID. Each run record stores the split-file SHA-256 and digests of
  the exact fit and validation ID lists, plus the first five ID-to-label pairs.
- Input transform: divide by 255 once (Phase 1 `normalized_images`), tensor
  `[B,1,28,28]` float32. No mean/std standardization.
- The training commands load only `official_train`. Official test is opened only
  by `python -m clothing_ml.phase2 test --allow-test` after `freeze`.

## Run matrix (seed 42, identical fit/validation IDs)

| run_order | Config | Change from cnn2 | Purpose |
|---:|---|---|---|
| 0 | `smoke.json` | smoke split, 2 epochs | plumbing only, never selectable |
| 1 | `cnn2.json` | — | reference: 2 blocks, dropout 0.30 |
| 2 | `cnn2_p0.json` | dropout 0.00 | dropout ablation |
| 3 | `cnn2_aug.json` | fit-only ±2 px translation | augmentation ablation |
| 4 | `cnn3.json` | third conv block | architecture comparison (depth + head size) |
| 5 | `cnn2_aug_p50.json` | fit-only ±2 px shift with p = 0.5 | revision (Sudeshhna): work-guide policy, run after the freeze; validation-only, never selectable |

Architecture (stand-in following the agreed handoff until Puneet's module lands):
each block is Conv3x3(pad 1) → ReLU → MaxPool2x2 with 32, 64 (, 128) channels;
then Flatten → Linear(→64) → ReLU (= the 64-value embedding) → Dropout(p) →
Linear(64→10). Parameter counts and per-layer shapes are saved in every run record.

Augmentation policy (declared before running): independent integer shifts
dx, dy ∈ {-2,…,2} per fit image per draw, zero fill, image size kept at 28×28.
Applied only to the fit loader; validation, embedding extraction and test use
canonical images.

## Optimisation and checkpoint rule

AdamW (lr 0.001, weight decay 0.0001), `nn.CrossEntropyLoss` on logits, batch 128,
up to 25 epochs, patience 5. After each epoch, validation macro F1 is computed on
the whole 12,000-image partition. A strictly higher score saves the checkpoint
immediately (temp file + atomic replace) and resets patience; a tie keeps the
earlier epoch. Training stops after 5 consecutive non-improving epochs. The best
file is reloaded before validation predictions are exported. Best epoch and stop
epoch are both logged.

Reproducibility: seeds for Python, NumPy and torch; `torch.use_deterministic_algorithms`;
`num_workers=0` with seeded generators for shuffling and augmentation. Device,
torch thread count and package versions are recorded. Cross-device bit-exactness
is not claimed.

## Model choice (validation only)

Main CNN = highest validation macro F1 among the four primary seed-42 runs,
then validation accuracy, then `run_order`. Implemented in `select.py` and
written to `results/phase2/full/cnn_selection.json`.

## Optional hybrid (declared before fitting the heads)

Encoder = the selected CNN checkpoint, frozen (hash pinned). Embedding = dense
ReLU output before dropout, eval mode, `inference_mode`, canonical images.
HOG = the Phase 1 1,296-value descriptor from the same raw IDs.

| Row | Features | Role |
|---|---:|---|
| selected_cnn | (own head) | practical CNN reference |
| emb_lr | 64 | controls the head change |
| hog_lr | 1,296 | controls the HOG branch |
| fusion_lr | 1,360 | ML + DL fusion |

Each LR head = StandardScaler (fitted on fit rows inside the Pipeline) →
LogisticRegression(lbfgs, max_iter 2,000, seed 42), C ∈ {0.1, 1}, chosen by
validation macro F1, then accuracy, then C order. Only converged fits are eligible.
Rows are joined by (source_split, sample_id) with label equality asserted.
Fusion is judged first against the two LR controls on validation, then against
the selected CNN for accuracy, Shirt/T-shirt/Pullover recall and end-to-end cost
(raw image → features → scaler → LR, 1,024 fixed validation IDs, one thread).

## Freeze and test

`freeze` writes `results/phase2/full/freeze.json` once: selected CNN, the three
LR heads if the extension ran, artifact hashes, ID digests, selection rules,
the Phase 1 reference (`full_hog_svm_3fd9d87a`, manifest hash) and the list of
validation-only rows. `test --allow-test` verifies every hash, writes an access
receipt, evaluates only the frozen rows on the 10,000 official test images and
refuses to overwrite an existing test summary. Ablation rows stay validation-only.

## Commands

```bash
python -m clothing_ml.phase2 train --config configs/phase2/smoke.json
python -m clothing_ml.phase2 verify --profile smoke --run-id smoke_cnn2_s42_e7074488
python -m clothing_ml.phase2 train --config configs/phase2/cnn2.json configs/phase2/cnn2_p0.json configs/phase2/cnn2_aug.json configs/phase2/cnn3.json
python -m clothing_ml.phase2 select --profile full
python -m clothing_ml.phase2 verify --profile full --run-id <selected run_id>
python -m clothing_ml.phase2 curves --profile full
python -m clothing_ml.phase2 fusion --profile full
python -m clothing_ml.phase2 freeze --profile full
# Team decision, after the freeze is reviewed:
python -m clothing_ml.phase2 test --profile full --allow-test
```
