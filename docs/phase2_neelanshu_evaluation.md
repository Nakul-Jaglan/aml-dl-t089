# Phase 2 · Neelanshu: evaluation, timing and paired error analysis

Every number below comes from committed prediction files. The script joins each file to the
official labels by `(source_split, sample_id)` before it scores anything. Nothing here retunes
a model.

```bash
python -m pip install -e '.[dev,phase2]'
python -m clothing_ml.phase2.evaluation all          # audit -> paired -> timing -> compare
python -m clothing_ml.phase2.evaluation lookup --id 40   # live demo
python -m pytest -q tests/test_phase2_evaluation.py
```

Code: [`src/clothing_ml/phase2/evaluation.py`](../src/clothing_ml/phase2/evaluation.py).
Outputs: [`results/phase2/full/evaluation/`](../results/phase2/full/evaluation/) and
[`figures/phase2/full/evaluation/`](../figures/phase2/full/evaluation/).
Timing refits the Phase 1 SVM once (about 4 minutes) because its artifact is git-ignored;
the other steps take under a minute.

## 1. Official test access

The four frozen Phase 2 rows reached the official test on 2026-11-18 (UTC) under freeze manifest
`250f974a…7a662`. The first complete run hit a bug in `final_test`: it called `.predict` on the
saved joblib dict instead of `bundle["pipeline"]`. I fixed that one line in `select.py` and reran
the same frozen rows. The CNN numbers matched across both runs. The fix touches no model, hash
or selection rule. [`test_access_log.md`](../results/phase2/full/test_access_log.md) lists every
attempt, and `test_frozen_lr_heads_are_bundles_with_a_pipeline` guards against the bug.

## 2. Prediction audit (deliverable 1)

[`prediction_audit.json`](../results/phase2/full/evaluation/prediction_audit.json) covers 27 files:
16 Phase 1 files (8 rows × val/test), 4 Phase 2 test files and 7 Phase 2 validation files.
Each file passed these checks:

- required columns `run_id, stage, source_split, sample_id, y_true, y_pred`, and one run ID;
- `stage` and `source_split` match the partition (`val`/`official_train`, `test`/`official_test`);
- exact ID membership with no duplicates (12,000 saved validation IDs or test IDs 0–9,999);
- category IDs 0–9 and `y_true` equal to the official label for that sample ID;
- support sums to n, diagonal / n equals accuracy, and every test class has support 1,000;
- recomputed accuracy, macro F1 and (where saved) the full confusion matrix equal the stored record
  to 1e-12.

[`metrics_export.csv`](../results/phase2/full/evaluation/metrics_export.csv) lists run ID, stage,
n, scores and the SHA-256 of each file Nakul or a second reviewer can re-hash.

## 3. Result table (one row per run)

Source: [`results_table.csv`](../results/phase2/full/evaluation/results_table.csv). The selection
rule picked every row before anyone opened the Phase 2 test. Timing: same machine, one CPU
thread, official-test IDs 0–1,023, batch 128 (section 5).

| Row | Run ID | Val macro F1 | Test acc. | Test macro F1 | Test Shirt recall | Median ms/image (range) |
|---|---|---:|---:|---:|---:|---:|
| **Selected CNN (CNN-3)** | `full_cnn3_s42_80a2edee` | 0.9308 | **92.14%** | **0.9207** | 71.7% | **0.312** (0.293–0.421) |
| Phase 1 HOG + RBF SVM | `full_hog_svm_3fd9d87a` | 0.9114 | 90.91% | 0.9090 | 73.3% | 19.748 (19.302–19.903) |
| Embedding + LR (control) | `full_emb_lr_C0.1_033b7a83` | 0.9319 | 92.21% | 0.9218 | 74.6% | 0.315 (0.294–0.393) |
| HOG + LR (control) | `full_hog_lr_C0.1_c1406307` | 0.8832 | 87.60% | 0.8757 | 66.1% | 0.205 (0.204–0.215) |
| Embedding + HOG + LR (hybrid) | `full_fusion_lr_C0.1_db46d180` | 0.9261 | 92.06% | 0.9206 | 76.0% | 0.518 (0.515–0.554) |
| CNN-2 reference | `full_cnn2_s42_3617f59e` | 0.9267 | validation only | | | |
| CNN-2, p = 0 | `full_cnn2_p0_s42_1d36ac31` | 0.9231 | validation only | | | |
| CNN-2 + shift | `full_cnn2_aug_s42_fdeb788e` | 0.9202 | validation only | | | |

The other seven Phase 1 rows stay in the CSV as context, under their original selection rule and
the same 10,000 test IDs. Per-class precision, recall, F1 and support for all ten labels, for
every file, are in [`per_class_metrics.csv`](../results/phase2/full/evaluation/per_class_metrics.csv).

### CNN vs SVM, per class (official test, support 1,000 each)

Source: [`per_class_cnn_vs_svm_test.csv`](../results/phase2/full/evaluation/per_class_cnn_vs_svm_test.csv).
Deltas use unrounded values and round once.

| Class | CNN recall | SVM recall | Δ recall (pp) | CNN precision | SVM precision |
|---|---:|---:|---:|---:|---:|
| T-shirt/top | 90.6% | 85.7% | +4.9 | 84.0% | 86.7% |
| Trouser | 99.0% | 97.6% | +1.4 | 98.8% | 98.4% |
| Pullover | 89.7% | 86.2% | +3.5 | 86.2% | 86.1% |
| Dress | 92.1% | 91.2% | +0.9 | 93.8% | 90.1% |
| Coat | 87.3% | 86.4% | +0.9 | 87.4% | 84.5% |
| Sandal | 98.6% | 96.5% | +2.1 | 98.8% | 98.3% |
| **Shirt** | **71.7%** | **73.3%** | **−1.6** | **80.9%** | **75.2%** |
| Sneaker | 98.3% | 97.3% | +1.0 | 94.7% | 94.2% |
| Bag | 98.6% | 98.3% | +0.3 | 98.2% | 97.5% |
| Ankle boot | 95.5% | 96.6% | −1.1 | 98.4% | 98.0% |

Overall the CNN gains +1.23 pp accuracy and +0.0118 macro F1. A paired bootstrap (seed 42,
2,000 repeats, resampling aligned test-ID pairs) gives 95% intervals of [+0.73, +1.75] pp for
accuracy and [+0.0067, +0.0170] for macro F1. Those intervals describe sampling variability of
this one 10,000-image benchmark. They say nothing about training-seed sensitivity (one seed
only) or transfer to real catalogue photos. Source:
[`cnn_vs_svm_deltas.json`](../results/phase2/full/evaluation/cnn_vs_svm_deltas.json).

## 4. Paired garment errors (deliverable 3)

Both models predicted the same 10,000 official-test IDs against the same labels. Source:
[`paired_groups.csv`](../results/phase2/full/evaluation/paired_groups.csv),
[`paired_summary.json`](../results/phase2/full/evaluation/paired_summary.json).

| Group | All classes | True T-shirt/Pullover/Coat/Shirt | True Shirt |
|---|---:|---:|---:|
| Both correct | 8,789 | 3,085 | 630 |
| Only CNN correct | 425 | 308 | 87 |
| Only HOG + SVM correct | 302 | 231 | 103 |
| Both wrong | 484 | 376 | 180 |
| **Total** | **10,000** | **4,000** | **1,000** |

Check: (425 − 302) / 10,000 = 0.0123, the exact accuracy delta. Exact McNemar on the 727
discordant pairs: p = 5.8 × 10⁻⁶. 74% of the discordant pairs (539 of 727) have a true label among the four
upper-body classes.

### Label-pair directions among the confusable classes

Source: [`label_pair_counts.csv`](../results/phase2/full/evaluation/label_pair_counts.csv)
(true → predicted, true-row support 1,000; label order T-shirt/top 0, Pullover 2, Coat 4, Shirt 6).

| True → predicted | CNN | SVM | Change |
|---|---:|---:|---:|
| Shirt → T-shirt/top | 134 | 103 | +31 |
| Shirt → Pullover | 60 | 63 | −3 |
| Shirt → Coat | 57 | 64 | −7 |
| T-shirt/top → Shirt | 64 | 104 | −40 |
| Pullover → Shirt | 33 | 59 | −26 |
| Coat → Shirt | 48 | 42 | +6 |
| Pullover → Coat | 42 | 57 | −15 |
| Coat → Pullover | 59 | 51 | +8 |

**Reading.** The CNN predicts "Shirt" less often. It sends 40 fewer true T-shirts and 26 fewer
true Pullovers to Shirt, so Shirt precision rises from 75.2% to 80.9% and T-shirt and Pullover
recall rise by 4.9 and 3.5 pp. The same shift sends 31 more true Shirts to T-shirt/top, so Shirt
recall falls by 1.6 pp. Shirt F1 still improves, from 0.742 to 0.760. In short, the CNN moves the
Shirt/T-shirt boundary; it doesn't remove the confusion. Shirt stays the weakest class for every
row in the table.

### Figures and example IDs

- [`confusion_cnn_vs_svm_test.png`](../figures/phase2/full/evaluation/confusion_cnn_vs_svm_test.png):
  row-normalised 10-class matrices for both models plus a CNN − SVM difference panel in pp.
- [`paired_examples_all.png`](../figures/phase2/full/evaluation/paired_examples_all.png) and
  [`paired_examples_upper_body.png`](../figures/phase2/full/evaluation/paired_examples_upper_body.png):
  four examples per outcome group.
- Selection rule: the first four sorted original test IDs in each group, across all classes and
  then restricted to true T-shirt/Pullover/Coat/Shirt. No hand picking. The IDs, true labels and
  both predictions are in
  [`figure_example_ids.csv`](../results/phase2/full/evaluation/figure_example_ids.csv).

### Captions for Nakul

Observations state what the files show. Hypotheses are marked as such; a single image can't
prove why a model decided something.

1. **Confusion matrices.** Row-normalised official-test confusion for CNN-3 and HOG + RBF SVM
   (1,000 images per true class). Right panel: CNN minus SVM in percentage points. The CNN gains
   on the T-shirt, Pullover and Sandal diagonals and loses 1.6 pp on Shirt, where Shirt → T-shirt
   errors rise from 10.3% to 13.4%.
2. **Paired examples (upper body).** The first four sorted test IDs in each outcome group among
   true T-shirt/top, Pullover, Coat and Shirt. Labels: true class, CNN prediction | SVM prediction.
3. **ID 40 (only CNN correct).** Observation: the SVM predicted T-shirt/top for this short-sleeved
   Shirt and the CNN predicted Shirt. Hypothesis: the visible button placket may have helped a
   learned filter more than a gradient histogram.
4. **IDs 136, 222, 227 (only CNN correct).** Observation: the SVM predicted Shirt for three
   Pullovers. Hypothesis: IDs 136 and 227 have plain, closed fronts; the SVM's habit of labelling plain
   long-sleeved tops "Shirt" matches the 59 → 33 drop in Pullover → Shirt.
5. **IDs 17, 57 (only SVM correct).** Observation: the CNN predicted Shirt for two Coats.
   Hypothesis: a busy print (ID 17) and a front zip read like shirt texture at 28 × 28.
6. **IDs 89, 98 (both wrong).** Observation: both models predicted Pullover for a Shirt (ID 89) and
   a Coat (ID 98). At this resolution, a dark long-sleeved top with no visible fastening gives
   either feature type little to separate the classes.

## 5. Timing and model size (deliverable 2)

Source: [`timing.json`](../results/phase2/full/evaluation/timing.json),
[`complexity.json`](../results/phase2/full/evaluation/complexity.json).

**Protocol (matches Phase 1).** Apple M3 (8 logical CPUs), macOS 26.2, Python 3.12.5,
torch 2.15.0, scikit-learn 1.9.1, scikit-image 0.26.0. Official-test raw uint8 IDs 0–1,023,
batch 128, one warm-up batch, five measured repeats. `torch.set_num_threads(1)`,
`set_num_interop_threads(1)` and `threadpool_limits(1)` held for every pipeline. The timer wraps
raw image → scaling → features → (scaler) → prediction → class IDs and excludes loading and disk
I/O, so the figure is amortised batch cost. Before timing, each pipeline reproduced its saved test
predictions on all 1,024 IDs. CPU only; I report no accelerator numbers.

**Phase 1 SVM artifact.** Git doesn't track the Phase 1 joblib file. I refit HOG + RBF SVM (C = 10)
with the recorded settings on the same 48,000 fit IDs, into git-ignored `work/`. The refit has the
recorded 20,916 support vectors and matches **10,000 / 10,000** saved test predictions, so the
same-machine timing measures the frozen model's decision function. Phase 1's historical figure on
another machine was 21.7 ms/image.

| Pipeline | Median ms/image | Range | Single-image latency (ms, batch 1, 128 IDs) |
|---|---:|---:|---:|
| HOG + RBF SVM | 19.748 | 19.302–19.903 | 19.19 |
| Selected CNN | 0.312 | 0.293–0.421 | 0.27 |
| Embedding + LR | 0.315 | 0.294–0.393 | 0.37 |
| HOG + LR | 0.205 | 0.204–0.215 | 0.33 |
| Embedding + HOG + LR | 0.518 | 0.515–0.554 | 0.58 |

The CNN runs about **63×** faster per image than the HOG + SVM (19.748 / 0.312). Samarth's
development timing (0.156 ms/image for the CNN) used validation IDs on an M3 Pro; compare rows
within one table only.

| Row | Size | Unit | Scaler scalars | Artifact |
|---|---:|---|---:|---|
| Selected CNN | 167,114 | trainable float32 parameters | 0 | 673,525 B, `torch.save` zip, uncompressed |
| HOG + RBF SVM | 27,295,425 | stored decision-array scalars (storage proxy, 20,916 support vectors) | 2,592 | 92,083,815 B, joblib compress=3 |
| Embedding + LR | 650 + 166,464 frozen encoder | LR coefs + intercepts; shipped encoder scalars | 128 | 4,415 B + CNN checkpoint |
| HOG + LR | 12,970 | LR coefs + intercepts | 2,592 | 79,571 B |
| Embedding + HOG + LR | 13,610 + 166,464 frozen encoder | LR coefs + intercepts; shipped encoder scalars | 2,720 | 82,514 B + CNN checkpoint |

The SVM value counts stored arrays, not learned weights, so it isn't directly comparable to a
parameter count. HOG + LR has fewer stored values than the CNN yet runs only 1.5× faster: HOG
extraction in Python dominates its time, which answers the viva question about weights vs speed.

## 6. Answer to the project question

On this public benchmark, learned features improve macro F1 (+0.0118; bootstrap 95% interval
+0.0067 to +0.0170) and accuracy (+1.23 pp), and the selected CNN is also about 63× cheaper per
image than the Phase 1 SVM on one CPU thread. The recurring upper-body confusion stays. The CNN
cuts T-shirt → Shirt and Pullover → Shirt errors but adds Shirt → T-shirt errors, so Shirt recall
drops 1.6 pp while Shirt precision and F1 rise. For inventory tagging the CNN is the better row on
accuracy, F1, speed and storage; an inventory team that needs high Shirt recall should still
route low-confidence upper-body tags to a person.

On the hybrid: on test, fusion (0.9206 F1) ties the CNN (0.9207) and sits below the embedding-only
control (0.9218) at 1.7× the CNN's cost, consistent with Samarth's validation finding. Fusion has
the highest Shirt recall (76.0%), but that came out of the test set, not from a rule we fixed
beforehand, so treat it as exploratory.

**Limitations.** One training seed; one benchmark split; Fashion-MNIST images are centred
grayscale 28 × 28 product shots; timing comes from one laptop CPU.

## Report paragraph (for Nakul)

> We scored every frozen row from its saved predictions after joining by original partition and
> sample ID; 27 prediction files passed identity and label checks, and recomputed metrics equal the
> stored records. On the official 10,000-image test, the validation-selected three-block CNN reached
> 92.14% accuracy and 0.9207 macro F1, against 90.91% and 0.9090 for the frozen Phase 1 HOG + RBF SVM
> (paired bootstrap 95% interval for the accuracy gain: +0.73 to +1.75 pp; exact McNemar p < 10⁻⁵ on
> 727 discordant images). The CNN raised T-shirt/top and Pullover recall by 4.9 and 3.5 pp but lowered
> Shirt recall from 73.3% to 71.7%, because it sends more true Shirts to T-shirt/top (134 vs 103) while
> sending fewer T-shirts to Shirt (64 vs 104). On one Apple M3 CPU thread with batch 128, the CNN took
> 0.31 ms per image end to end against 19.7 ms for the SVM, with 167,114 parameters against 27.3 M stored
> SVM decision values. These intervals cover only this benchmark sample; we trained one seed.

## Presenter talking points (Neelanshu)

1. Audit first: 27 files, joined by ID, recomputed metrics equal the records. Live demo:
   `lookup --id 40`, then recompute Shirt recall (717 / 1,000 and 733 / 1,000).
2. Headline: +1.23 pp accuracy and +0.0118 F1, interval excludes zero, single split and seed.
3. Paired groups: 425 only-CNN vs 302 only-SVM; the difference divided by 10,000 equals the
   accuracy delta.
4. Shirt: the CNN moves the boundary. Fewer T-shirt → Shirt errors, more Shirt → T-shirt errors.
5. Cost: 0.31 vs 19.7 ms/image, 167k parameters vs 27.3 M stored SVM values. Fewer weights don't
   guarantee speed (HOG + LR shows that).

Hinglish: "Accuracy overall result batati hai, Shirt recall sirf actual Shirts mein se sahi
pehchaani hui images ka fraction hai. CNN ne T-shirt ko Shirt bolna kam kiya, par Shirt ko
T-shirt zyada bola, isliye Shirt recall 1.6 point gira. Same 10,000 test IDs par dono models ki
galtiyan compare ki hain, aur CNN 63 guna fast hai."

## Viva preparation

| Question | Answer |
|---|---|
| Why macro F1? | Each of the ten classes counts equally and mixes precision with recall inside that class. |
| Does 92% accuracy mean 92% Shirt recall? | No. Shirt recall is 71.7%; accuracy averages over all 10,000 images. |
| What makes the comparison paired? | Both models predict the same original test IDs with the same official labels. |
| Fewer weights but slower? | Yes. HOG + LR stores 12,970 values yet spends most of its time on HOG extraction; time depends on operations, threads, batch size and hardware. |
| CNN better F1 and faster: anything to hedge? | One seed and one split; the bootstrap covers sampling of test images only. Shirt recall fell. |
