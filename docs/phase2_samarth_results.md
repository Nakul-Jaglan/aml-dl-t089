# Phase 2 — Samarth: training runner, ablations and fusion (validation evidence)

All numbers below are **validation** results on the same 48,000 fit / 12,000
validation `official_train` IDs (`data/full_indices.npz`, seed 42, CPU, Apple M3 Pro,
torch 2.15.0). Official test has **not** been opened by Phase 2; the frozen rows
are listed in `results/phase2/full/freeze.json` (manifest
`250f974a…7a662`). Protocol: [`docs/phase2_training_protocol.md`](phase2_training_protocol.md).

## Smoke run (first working session)

`smoke_cnn2_s42_e7074488`: 2,000 fit / 1,000 validation smoke IDs, 2 epochs on CPU.
Data, loss, atomic checkpoint, prediction export and a **fresh-process reload**
all work: the reloaded checkpoint reproduces 1,000/1,000 exported validation
predictions (`results/phase2/smoke/runs/smoke_cnn2_s42_e7074488/reload_check.json`).
The first five ID-to-label pairs are stored in each run record.

## Candidate table (deliverable)

Source: `results/phase2/full/validation_candidates.csv`.

| Config | Change vs cnn2 | Params | Val macro F1 | Val acc. | Best epoch | Stop epoch | Shirt recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `full_cnn2_s42_3617f59e` | reference (2 blocks, p=0.30) | 220,234 | 0.9267 | 0.9269 | 24 | 25 | 0.772 |
| `full_cnn2_p0_s42_1d36ac31` | dropout 0.00 | 220,234 | 0.9231 | 0.9230 | 17 | 22 | 0.805 |
| `full_cnn2_aug_s42_fdeb788e` | fit-only ±2 px shift | 220,234 | 0.9202 | 0.9210 | 25 | 25 | 0.691 |
| **`full_cnn3_s42_80a2edee`** | third block | 167,114 | **0.9308** | **0.9309** | 22 | 25 | 0.791 |

Selected CNN (rule: val macro F1 → val accuracy → run order): **CNN-3**.
Its checkpoint was reloaded in a fresh process and reproduced all 12,000
exported validation predictions (`runs/full_cnn3_s42_80a2edee/reload_check.json`).
Phase 1 reference on the same validation IDs: HOG + RBF SVM, val macro F1 0.9114.

Learning curves: `figures/phase2/full/learning_curves.png` and
`figures/phase2/full/val_macro_f1_by_epoch.png`.

### One-sentence interpretations (validation only, single seed)

- **Dropout 0.30 vs 0.00 (−0.0036 F1 without dropout):** without dropout the fit loss
  keeps falling while validation loss bottoms out near epoch 12 (0.221) and climbs to 0.288 by epoch 22, and the best epoch
  comes earlier (17 vs 24), consistent with dropout reducing overfitting at this
  layer; the gap is small for one seed.
- **Translation vs canonical fit (−0.0065 F1 with shift):** the ±2 px policy made
  fit harder (fit loss stays above validation loss) and was still improving at the
  25-epoch cap (best = last epoch), so under this fixed budget it hurt, mainly via
  Shirt recall (−8.1 pp) while T-shirt and Pullover recall rose slightly.
- **Two vs three blocks (+0.0041 F1 for CNN-3):** the extra block and pooling
  shrink the dense head (3,136 → 1,152 inputs) and total parameters (220k → 167k)
  yet improve validation F1 and Shirt/T-shirt/Pullover recall; this is an
  architecture comparison, not a pure depth effect.
- **Budget caveat:** cnn2 and cnn2_aug peaked at epochs 24–25, so the 25-epoch cap,
  not patience, ended them. Seed repeats were **not** run (compute time prioritised
  the four declared rows); differences of <0.01 F1 on one seed are not claimed as robust.

### CNN-3 layer and shape table

| Layer | Output | Params |
|---|---|---:|
| Conv3x3 (1→32) + ReLU | 32×28×28 | 320 |
| MaxPool 2×2 | 32×14×14 | 0 |
| Conv3x3 (32→64) + ReLU | 64×14×14 | 18,496 |
| MaxPool 2×2 | 64×7×7 | 0 |
| Conv3x3 (64→128) + ReLU | 128×7×7 | 73,856 |
| MaxPool 2×2 | 128×3×3 | 0 |
| Flatten → Linear(1152→64) + ReLU = embedding | 64 | 73,792 |
| Dropout p=0.30 | 64 | 0 |
| Linear(64→10) logits | 10 | 650 |

## Optional hybrid: CNN embedding + HOG → Logistic Regression

Encoder = frozen CNN-3 checkpoint (`e29a9d77…`). Source:
`results/phase2/full/fusion/fusion_summary.csv`, `fusion_candidates.csv`,
`fusion_selection.json`, `fusion_timing.json`.

| Row | Features | C | Val macro F1 | Val acc. | Shirt | T-shirt | Pullover | End-to-end ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selected CNN (own head) | 64 | – | 0.9308 | 0.9309 | 0.791 | 0.918 | 0.898 | 0.156 |
| embedding + LR | 64 | 0.1 | **0.9319** | **0.9318** | 0.814 | 0.901 | 0.894 | 0.157 |
| HOG + LR | 1,296 | 0.1 | 0.8832 | 0.8831 | 0.700 | 0.846 | 0.823 | 0.205 |
| embedding + HOG + LR | 1,360 | 0.1 | 0.9261 | 0.9260 | 0.805 | 0.878 | 0.883 | 0.365 |

Timing: median amortised cost, 1,024 fixed validation IDs, batch 128, one thread,
raw uint8 image → normalisation/HOG/CNN → scaler → LR. Development timing only;
final timing belongs to Neelanshu's comparison.

Paired validation errors (fusion vs embedding-only LR): 165 fixed by fusion,
235 broken by fusion, 653 wrong in both.

**What this supports.** Fusion beats HOG-only by a wide margin but is **worse**
than the embedding-only control (−0.0058 F1) and the CNN's own head, at ~2.3× the
inference cost. So on this pipeline, HOG did not add useful edge information on top
of the learned 64-value embedding; the HOG block (1,296 standardised features vs 64)
probably dominates the shared L2 penalty. Swapping the neural head for LR changes
little (+0.0011 F1). This answers the secondary question honestly; the core CNN
comparison is unaffected. The result is specific to this encoder, scaler and C grid.

## Frozen rows for the official test (team decision to run)

`selected_cnn` (`full_cnn3_s42_80a2edee`), `emb_lr`, `hog_lr`, `fusion_lr`, plus the
frozen Phase 1 reference `full_hog_svm_3fd9d87a` (reuse its saved test predictions).
The ablation rows cnn2, cnn2_p0 and cnn2_aug stay **validation-only**.

```bash
python -m clothing_ml.phase2 test --profile full --allow-test
```

Checkpoints and LR heads (3.5 MB) are committed under `artifacts/phase2/full/` as
an explicit exception to the "no model artifacts in Git" rule, because the freeze
and test step verify their exact SHA-256 hashes and CPU retraining on another
machine is not guaranteed to be bit-identical.

## Report paragraph (for Nakul)

> We trained two small CNN designs with AdamW (lr 10⁻³, decoupled weight decay
> 10⁻⁴ [Loshchilov & Hutter, 2019]) and cross-entropy on 48,000 fit images, keeping
> the epoch with the highest validation macro F1 (patience 5, ≤25 epochs). On the
> identical 12,000 validation images, a three-block CNN (167k parameters) reached
> 0.9308 macro F1, versus 0.9267 for the two-block reference, 0.9231 without
> dropout and 0.9202 with ±2 px fit-only translation; the Phase 1 HOG + RBF SVM
> scored 0.9114 on the same IDs. Removing dropout [Srivastava et al., 2014] made
> validation loss climb after epoch ~17, and translation slowed convergence within
> the fixed budget. An optional hybrid concatenating the frozen 64-value CNN
> embedding with 1,296 HOG features [Dalal & Triggs, 2005] in a Logistic Regression
> scored 0.9261, below the embedding-only control (0.9319) at about 2.3× the
> inference cost, so the fusion is not supported here. All selections used
> validation data only; single-seed differences under 0.01 are not claimed as robust.

## Presenter talking points

1. Runner: one config → one run record (seed, device, versions, config/split digests, history, checkpoint hash).
2. Checkpoint rule: best validation macro F1, earlier epoch on ties, patience 5; best file reloaded before export.
3. Live demo: show `learning_curves.png`, best epoch 22 for CNN-3, then `verify` reproduces 12,000/12,000 predictions.
4. Ablations: dropout helps a little; shift augmentation hurt within 25 epochs; CNN-3 best with fewer parameters.
5. Hybrid: explain HOG-only and embedding-only rows first, then fusion — negative but clean result.

Hinglish: "CNN-3 validation macro F1 se select hua, epoch 22 ka checkpoint. Fusion
mein embedding aur HOG join kiya, par embedding-only LR se thoda kam aaya —
toh HOG ne yahan extra value nahi di, aur cost double ho gaya."
