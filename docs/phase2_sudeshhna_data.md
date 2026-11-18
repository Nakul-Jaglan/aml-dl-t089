# Phase 2 — Sudeshhna: CNN data adapter and augmentation policy

Owner: Sudeshhna. Code: [`src/clothing_ml/phase2/data.py`](../src/clothing_ml/phase2/data.py),
tests: [`tests/test_phase2_data.py`](../tests/test_phase2_data.py), evidence scripts:
[`scripts/phase2_data_audit.py`](../scripts/phase2_data_audit.py) and
[`scripts/phase2_augmentation_report.py`](../scripts/phase2_augmentation_report.py).
Everything here reads **official_train only**; official test was never loaded.

```bash
python -m pytest -q tests/test_phase2_data.py
python scripts/phase2_data_audit.py            # split audit, first batches, stable-eval, wraparound, figures
python -m clothing_ml.phase2 train --config configs/phase2/cnn2_aug_p50.json
python -m clothing_ml.phase2 verify --profile full --run-id full_cnn2_aug_p50_s42_fe3c3cdd
python scripts/phase2_augmentation_report.py   # ablation table + curves from saved records
```

## 1. Data adapter (handoff to Puneet and Samarth)

`ClothingImages(raw, labels, ids, transform=None)` is the adapter from the work guide:
it indexes the raw uint8 partition by **original row ID**, converts to float32,
adds the channel axis and divides by 255 **once**, applies an optional transform,
and returns `(image [1,28,28], label, original_id)`. It rejects float input (guards
against double scaling), duplicate IDs, out-of-range IDs and misaligned labels.
HOG is untouched: it still receives raw uint8 images through the Phase 1 feature code.

| Piece | Contract |
|---|---|
| `make_loaders(profile, augmentation, seed=42, batch_size=128)` | the one loader factory: `{"fit", "val"}` loaders + metadata (split file, SHA-256, sizes, policy) |
| fit loader | seeded shuffle (`torch.Generator`, seed 42), `num_workers=0`, fit-only transform |
| val / test loader | saved order, no random transform, `drop_last=False` |
| collated batch | image float32 `[B,1,28,28]` in [0,1], label int64 `[B]`, ID int64 `[B]` |
| `check_batch(batch, raw, labels)` | asserts dtype/shape/range/finiteness, label == `labels[id]`, image == `raw[id]/255` |
| `FashionIDDataset(...)` | runner-facing name used by `runner.py`/`fusion.py`; now a thin subclass of `ClothingImages` |

**Compatibility with the runs already trained.** Samarth's runs used a stand-in
loader. I checked the new adapter against it on the full split: canonical
validation tensors and full shuffled fit epochs (375 batches) are **bit-identical**
for both `none` and the legacy `translate2` policy, so every existing run record,
checkpoint and the freeze manifest stay valid. After the change,
`verify` on the selected CNN-3 again reproduces 12,000/12,000 saved validation
predictions (`results/phase2/full/runs/full_cnn3_s42_80a2edee/reload_check.json`).

## 2. Split audit (first working session)

Source: `results/phase2/data_audit/data_audit.json`, `class_counts.csv`, `first_batch.csv`.

- Raw official_train: shape (60000, 28, 28), uint8, pixel min 0 / max 255, ten
  category names taken from `clothing_ml.data.CLASS_NAMES`.
- `data/full_indices.npz`: 48,000 fit + 12,000 validation IDs, no overlap, together
  exactly rows 0–59,999; **4,800 fit and 1,200 validation per class**.
  `data/smoke_indices.npz`: 2,000 / 1,000 (200 / 100 per class), a subset of the full split.
- Split SHA-256 `9172928b…80ec9` matches `data/provenance.json`,
  `results/full/protocol.json` (Phase 1) and the `split_sha256` of every Phase 2
  run record (smoke: `800178fd…9645b`).
- First smoke fit batch (seed 42): IDs 9422 Bag, 10277 Coat, 13136 Shirt, 29452 Sneaker,
  57204 Trouser — each `1x28x28` float32 in [0, 1] and equal to `raw[id]/255`.
- Stable evaluation: two passes over the full validation loader give identical
  images, labels and IDs (24 batches, 12,000 items, in saved order); one fit epoch
  covers each fit ID exactly once in shuffled order.
- Example grid (four seeded fit images per class): `figures/phase2/data/example_grid.png`.

## 3. Augmentation policy

Declared policy (`translate2_p50`, work guide p.6): for each fit image, with
probability 0.5 draw independent integer offsets dx, dy ∈ [−2, 2] and shift with
zero fill and no wraparound; otherwise return the same input. Validation, test and
hybrid embedding extraction always use the canonical image. Seed 42 drives the
transform's own generator; the unaugmented partner is the cnn2 reference with the
same seed, split and settings.

Checks (in `data_audit.json` and `tests/test_phase2_data.py`):
- **Wraparound:** a white pixel at each corner and near an edge, for all 25 offsets
  (125 cases): it either moves inside the frame or disappears, never reappears on
  the opposite side; the uncovered border is exactly 0; shape stays 1×28×28, range [0, 1].
- **Rate:** over one fit epoch, 49.96 % of images drew a shift and 52.0 % were left
  unchanged (the rest drew (0, 0)). The legacy `translate2` shifts 96.1 %.
- **Visual:** `figures/phase2/data/shift_comparison.png` — T-shirt/top, Pullover,
  Coat, Shirt and the three footwear classes, original vs sampled shift vs the
  largest (corner) shift, captions carry label and original ID. A corner shift
  removes at most 7 % of garment intensity in these examples (T-shirt sleeve),
  so ±2 px keeps labels readable while ±3 or more would start cutting sleeves and heels.

## 4. Augmentation ablation (validation only)

Source: `results/phase2/data_audit/augmentation_ablation.csv/.json`,
`figures/phase2/data/augmentation_curves.png`. All three rows share seed 42, the same
48,000 fit / 12,000 validation IDs (digests asserted), 2 blocks, dropout 0.30, AdamW
(lr 0.001, wd 0.0001), batch 128, ≤25 epochs, patience 5.

| Run | Fit transform | Val macro F1 | Val acc. | Shirt | T-shirt | Pullover | Best / stop epoch | Final fit / val loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `full_cnn2_s42_3617f59e` | none (reference) | 0.9267 | 0.9269 | 0.772 | 0.905 | 0.860 | 24 / 25 | 0.135 / 0.247 |
| `full_cnn2_aug_s42_fdeb788e` | shift every image (legacy) | 0.9202 | 0.9210 | 0.691 | 0.921 | 0.873 | 25 / 25 | 0.269 / 0.220 |
| `full_cnn2_aug_p50_s42_fe3c3cdd` | **shift with p = 0.5** | **0.9279** | **0.9287** | 0.723 | 0.933 | 0.902 | 25 / 25 | 0.223 / 0.199 |

Paired validation errors vs the reference (joined by original ID): the p = 0.5 policy
fixes 284 images and breaks 263 (593 wrong in both); on Shirt it fixes 52 and breaks 110.
The p = 0.5 checkpoint reloads in a fresh process and reproduces 12,000/12,000 saved
validation predictions (`runs/full_cnn2_aug_p50_s42_fe3c3cdd/reload_check.json`).

**What this answers.**
- Applying the shift to half the images instead of all of them turns the result from
  −0.0065 to **+0.0012 macro F1** against the unaugmented partner. +0.0012 on one seed
  is within the noise we already see between rows, so the honest conclusion is
  *no clear gain in macro F1* — but the gentler policy does not hurt.
- Regularisation is visible in the curves: without augmentation, fit loss keeps falling
  (0.135) while validation loss bottoms at epoch 18 (0.221) and rises to 0.247. With
  p = 0.5, validation loss is still falling at epoch 25 (0.199, the lowest of the
  three) and the fit/validation gap is small, so it overfits less.
- Both shifted runs peaked at the last epoch: the 25-epoch cap, not patience, ended
  them. A longer budget might favour augmentation more; that is not claimed here.
- **Shirt recall drops** (−4.8 pp with p = 0.5, −8.1 pp with every-image shift) while
  T-shirt/top (+2.8 pp) and Pullover (+4.2 pp) rise; the lost Shirts go mostly to
  T-shirt/top and Coat. Shifting seems to blur the small position cues (collar,
  button placket) that separate Shirt from its neighbours, so translation does **not**
  reduce the Shirt confusion that the project question asks about.
- Timing: 269 s vs 220 s for the reference on CPU (on-the-fly shifting + CPU contention).

**Status.** `cnn2_aug_p50` (run_order 5) was trained after the CNN selection and freeze
were written, so it is a validation-only ablation and **not** a selection candidate;
`select.py` now restricts candidates to the declared run_order 1–4 so re-running
`select` cannot change the frozen choice. Its val F1 (0.9279) is below the selected
CNN-3 (0.9308) anyway.

## 5. Data quality and limits (paragraph for Nakul)

> Phase 2 reuses the official Fashion-MNIST images [Xiao et al., 2017] and the exact
> Phase 1 partition (`data/full_indices.npz`, SHA-256 `9172928b…`): 48,000 fit and
> 12,000 validation images that together cover all 60,000 official training rows
> without overlap, with exactly 4,800 / 1,200 images per category, and the untouched
> 10,000-image official test set. Each CNN input is the raw uint8 image divided by 255
> once and shaped [1, 28, 28]; every item keeps its original row ID so predictions can
> be joined across models regardless of DataLoader order. The only augmentation tested
> is a fit-only translation of at most ±2 px with zero fill, applied with probability
> 0.5; validation, test and embedding extraction use canonical images, and two
> passes over the validation loader are identical. On validation this policy changed
> macro F1 by +0.0012 against the unaugmented partner (0.9279 vs 0.9267), lowered the
> fit–validation loss gap, but reduced Shirt recall by 4.8 pp; shifting every image
> instead lowered macro F1 by 0.0065. The Phase 1 audit found five validation images
> and six official-test images within pixel-RMS 5 of a training image; we keep the full
> official test as the headline and treat these as a known dataset limitation. Pixel
> RMS is sensitive to translation, scale and lighting and cannot decide whether two
> different photos show the same garment.

## 6. Viva preparation

| Question | Answer |
|---|---|
| Why keep IDs after shuffling? | So predictions and inputs can be joined correctly across models even when the DataLoader changes order; `check_batch` asserts `label == labels[id]`. |
| Why scale once? | Raw pixels are 0–255. Dividing twice squeezes inputs into [0, 1/255]. The adapter refuses float input to prevent it. |
| What would leak information? | Fitting a scaler on validation/test, choosing augmentation from test results, or using evaluation labels in fitting. Test was never loaded here. |
| Why leave validation unaugmented? | It must estimate performance on the canonical benchmark images and be a stable criterion: two passes are identical. |
| Do classes stay balanced? | Yes: 6,000 per class in official train, 4,800 fit + 1,200 validation each. |
| Why ±2 px, zero fill? | Larger shifts start cutting sleeves and heels at 28×28; wraparound would paste a sleeve on the opposite edge. 125 edge cases checked. |
| Why p = 0.5? | Half the batches still show canonical images, so the model keeps learning the centred benchmark view; shifting every image under-fit in 25 epochs (−0.0065 F1). |
| Did augmentation help? | Not clearly: +0.0012 F1 on one seed, lower val loss, but Shirt recall fell 4.8 pp. A flat result still answers the ablation. |

**Live demo.** `python scripts/phase2_data_audit.py` prints the first smoke batch
(ID, label, `1x28x28`, float32, range, `matches_raw_over_255=True`); show the same ID in
`figures/phase2/data/example_grid.png` or plot `load_official("train")[0][id]`; point to
`stable_evaluation.two_passes_identical_images = true` in `data_audit.json`.

Hinglish: "Same IDs use kiye hain taaki Phase 1 aur Phase 2 ka comparison fair rahe.
CNN ko [1, 28, 28] tensor milta hai, 255 se sirf ek baar divide. Random shift sirf
augmented training run mein, aadhe images pe, ±2 pixel tak. Macro F1 lagbhag same raha,
par Shirt recall gir gaya — toh shift ne Shirt confusion solve nahi kiya."

## Presenter talking points

1. Adapter contract: (image, label, ID), float32 [B,1,28,28], int64 labels, original IDs.
2. Audit: split hash matches Phase 1 and every run; 4,800 / 1,200 per class; stable validation.
3. Policy: ±2 px, zero fill, no wraparound, p = 0.5, fit only — show `shift_comparison.png`.
4. Result: `augmentation_curves.png` — tiny F1 change, smaller overfitting gap, Shirt recall drop.
5. Limits: near-duplicates (5 val, 6 test under RMS 5), single seed, 25-epoch cap.
