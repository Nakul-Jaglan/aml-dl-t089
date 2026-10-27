# Earlier Phase 1 presentation outline

The finished 22-slide deck is in `presentation/`. Use
`presentation/speaker_notes.md` for the final slide numbering and presenter notes.
The 12-slide outline below records the earlier content plan.

This is a 12-slide content outline, not an exported PowerPoint deck. Adjust its
length to the faculty's actual time limit. Each presenter must review their
material and the whole pipeline. Numbers come from frozen full-profile results.

## 1. Clothing category recognition (Nakul)
Input: a 28x28 grayscale clothing image. Output: one of ten labels.
Explain the inventory-tagging context. State that this presentation covers Phase 1.

## 2. Research question and prior work (Nakul)
Ask whether local gradient structure helps compared with raw intensity, and
whether the answer depends on the classifier. Mention Xiao et al. (2017), Dalal
and Triggs (2005), and the distinct model families. No new algorithm is claimed.

## 3. Dataset and quality checks (Sudeshhna)
Show `figures/eda/training_gallery.png`. Report the executed checks: balanced
classes, finite uint8 input, no exact training duplicates and no all-zero images.
Explain that hash checks do not detect every near-duplicate or verify every label.

## 4. Fair split and development sequence (Sudeshhna)
Explain 48,000 fit / 12,000 validation / 10,000 official test. Describe the 12,000
fit-subset search, then full-size final fitting. Keep original sample IDs attached.

## 5. Pixels and HOG (Puneet)
Show `figures/eda/hog_example.png`. Derive 784 versus 1,296 features. Explain fixed
local edges, overlapping blocks and the difference between normalization and
training-only standardization. HOG is larger here and is not learned end to end.

## 6. Four decision rules (Puneet and Samarth)
LR learns linear scores. kNN votes among stored neighbors. RF combines randomized
trees. RBF SVM uses nonlinear margin-based decisions. Explain C, k and depth using
the settings actually used. Mention the model-specific scaling policy.

## 7. Model selection and freeze (Samarth)
Show the validation table. Primary metric: macro F1. Freeze one row per model/feature
and the overall selected model before reading test data. All final rows use 48,000
fit images. Save exact artifacts, IDs, warnings and hashes.

## 8. Aggregate results (Neelanshu)
Show `figures/full/model_comparison.png`. The preselected hog +
RBF SVM achieved 90.91% test accuracy and
0.9090 macro F1. HOG-minus-pixel F1 differences in percentage points:
LR: +3.63, kNN: +2.26, RF: -0.35, RBF SVM: +1.65. Avoid claiming statistical significance from one split.

## 9. Recall and garment confusion (Neelanshu)
Show `figures/full/per_class_recall.png` or the focused report confusion figure.
The selected model's weakest recall was Shirt, 73.3%.
Rows mean actual labels and columns mean predictions. Explain one measured pair.

## 10. Paired errors (Neelanshu)
Show `figures/full/paired_errors_svm.png`. Explain the deterministic first-ID rule
and the both-correct / pixels-only / HOG-only / both-wrong groups. Visible shapes
suggest possible causes but do not establish causal explanations or wrong labels.

## 11. Runtime, memory and reproducibility (Nakul)
Show `figures/full/inference_time.png`. Explain batch-amortized end-to-end timing,
including HOG, versus Pipeline-only prediction. Describe the different classical
complexity measures and the executed notebook's independent metric checks.

## 12. Conclusion and Phase 2 (Nakul)
State the measured trade-offs, weak categories and benchmark limits. Propose a
small CNN using the same split, validation-based architecture selection and fair
timing. Its performance is unmeasured. Finish with a short saved-prediction demo.

## Demo backup
Open `notebooks/00_phase1_master.ipynb`, select original test ID 0, show true label
and the frozen prediction. Recompute a metric from its CSV. The saved notebook,
figures and report are the backup if the network or live environment fails.

## Sources
Use `docs/literature_review.md` and `report/references.bib` for the cited sources.
Every result slide should cite the matching `results/full/` table or prediction file.
