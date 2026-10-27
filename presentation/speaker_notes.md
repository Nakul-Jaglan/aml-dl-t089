# Phase 1 presenter notes

Slides 1–18 form the main presentation. Slides 19–22 are backup slides.
Speaking assignments follow the team responsibilities. Adjust pacing to the faculty time limit.

## 1. Automatic clothing category recognition

Presenter: **Nakul**

This project studies automatic category tagging for a small clothing catalogue. The input is one 28 by 28 grayscale image and the output is one of ten Fashion-MNIST categories. This presentation covers the completed classical machine learning experiments in Phase 1. The broader project title includes a small CNN, but its training and comparison belong to Phase 2. Introduce all five team members. The assigned roles are responsibility areas and speaking assignments. They do not replace each member’s review of the shared project.

Sources:
- docs/course_instructions.md
- data/provenance.json

## 2. Research question and prior work

Presenter: **Nakul**

The comparison asks whether fixed local gradient features help more than raw pixel intensities, and whether the answer depends on the classifier. Fashion-MNIST provides a standard dataset with visually similar clothing classes. Dalal and Triggs motivate local gradient histograms, although their original task was human detection. Random Forests and support-vector networks provide different decision rules for the same inputs. This project applies established methods. It does not propose a new algorithm. The CNN is a later test of learned visual features, and no Phase 2 result is available.

Sources:
- https://arxiv.org/abs/1708.07747
- https://doi.org/10.1109/CVPR.2005.177
- https://doi.org/10.1023/A:1010933404324
- https://doi.org/10.1007/BF00994018
- docs/literature_review.md

## 3. Team responsibilities

Presenter: **Nakul**

Introduce the division of responsibility and the speaking handoffs. Nakul covers integration, research and reporting. Sudeshhna explains the dataset and validation split. Puneet explains the feature representations and the simpler classifiers. Samarth explains Random Forest, SVM and model selection. Neelanshu explains metrics, errors and runtime. Each member should rehearse their own slides and understand the entire pipeline. The contribution log is the place to record actual individual reviews or changes.

Sources:
- docs/contributions.md

## 4. Fashion-MNIST dataset

Presenter: **Sudeshhna**

Fashion-MNIST has 60,000 official training images and 10,000 official test images. Each image contains 28 by 28 grayscale values, with an intact integer category label. The ten official categories include T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot. The gallery shows seeded examples from the fit pool only. Each training category contains 6,000 images. The benchmark images are small and standardized, so this experiment does not establish performance on phone photographs, cluttered backgrounds or every clothing type.

Sources:
- https://github.com/zalandoresearch/fashion-mnist
- data/training_audit.json
- figures/eda/gallery_ids.json

## 5. Data checks and fair partitions

Presenter: **Sudeshhna**

The downloaded IDX files match the published checksums. The audit confirms the expected shape, integer pixel range zero to 255, finite values and all ten balanced labels. It found no all-zero training images, exact training duplicate groups, conflicting-label exact duplicates or exact duplicate overlap between fit and validation. These checks do not verify every semantic label or detect every near-duplicate. A stratified seed-42 split creates 48,000 fit images and 12,000 validation images from the official training set. The official test partition stays separate. Scalers learn their statistics only from fit data, and sample IDs preserve alignment across all pipelines.

Sources:
- data/training_audit.json
- data/provenance.json
- data/split_indices.npz
- src/clothing_ml/data.py

## 6. Pixels and HOG features

Presenter: **Puneet**

First divide uint8 image intensities by 255 exactly once. Flattening preserves the 784 numeric values, but the classifier does not receive an explicit neighborhood representation. HOG calculates gradient direction histograms over four by four pixel cells. A 28-pixel side contains seven cells. Two by two cell blocks overlap with a one-cell stride, so there are six by six blocks. Each block has four cells and nine orientation bins, giving 6 times 6 times 4 times 9 equals 1,296 features. L2-Hys normalizes each block. This HOG vector is larger than the pixel vector. It is a fixed descriptor, not a learned CNN representation. Image orientation stays unchanged.

Sources:
- src/clothing_ml/features.py
- figures/eda/hog_example.png
- https://doi.org/10.1109/CVPR.2005.177

## 7. Logistic Regression and kNN

Presenter: **Puneet**

Multiclass Logistic Regression learns one linear score per category and fits the scores with regularization. A larger C allows weaker regularization. It is fast to evaluate, but a linear boundary can miss complex relationships. kNN stores the fit examples and assigns labels using nearby examples in feature space. Here it uses brute-force Euclidean distances and distance-weighted voting. k controls how many neighbors vote. kNN has almost no optimization cost, but prediction can be expensive because it compares a query with stored examples. LR uses StandardScaler for both feature types. HOG kNN uses StandardScaler, while pixel kNN uses normalized intensities without learned standardization. The comparison therefore concerns declared pipelines rather than feature extraction in isolation.

Sources:
- src/clothing_ml/models.py
- docs/protocol.md
- results/full/freeze.json

## 8. Random Forest and RBF SVM

Presenter: **Samarth**

Random Forest averages decisions from randomized trees. Bootstrap samples and random feature subsets reduce dependence between the trees. Each forest here contains 200 trees and uses square-root feature sampling. Maximum depth controls how detailed the tree rules can become. RF receives no StandardScaler because its threshold rules do not require it. RBF SVM uses a nonlinear similarity function and a margin-based decision rule. C controls the penalty for training errors relative to margin regularization, and gamma controls how local the RBF similarity is. Both SVM representations use StandardScaler. Gamma stays at scale, C is selected from a small grid, and probability estimation is disabled. These models illustrate different ways to use the same fixed image features.

Sources:
- src/clothing_ml/models.py
- results/full/freeze.json
- https://doi.org/10.1023/A:1010933404324
- https://doi.org/10.1007/BF00994018

## 9. Model selection before test access

Presenter: **Samarth**

The smoke profile checks that all pipeline types run. It cannot unlock official-test evaluation. The compact search uses a stratified 12,000-image subset of the fit pool and the same 12,000-image validation set. Each of eight model-feature rows has two candidates, giving 16 search fits. LR searches C 0.1 and 1. kNN searches k 3 and 5. RF searches maximum depth 20 and None. RBF SVM searches C 1 and 10. Selected settings then train afresh on all 48,000 fit images. Full-profile validation macro F1 chooses the overall pipeline. Accuracy, prediction time and run ID break ties in that order. A hashed manifest freezes artifacts and selection before official-test access. This is a compact search, not exhaustive optimization.

Sources:
- docs/protocol.md
- results/budget/validation_candidates.csv
- results/full/freeze.json
- results/full/test_receipt.json

## 10. Evaluation measures

Presenter: **Neelanshu**

Accuracy is the fraction of all images that receive the correct category. For a single class, precision measures how often that prediction is correct, and recall measures how many true members of that class the model finds. F1 is two times precision times recall divided by their sum. Macro F1 averages the ten class F1 values with equal class weight. It is not the harmonic mean of macro precision and macro recall. Per-class recall exposes a weak category even when overall accuracy is high. Confusion matrix rows are true categories and columns are predicted categories. We also measure inference time and model-specific complexity because classification quality alone does not determine usability.

Sources:
- src/clothing_ml/evaluate.py
- tests/test_integrity.py
- results/full/per_class_recall.csv

## 11. HOG helps three classifier families

Presenter: **Neelanshu**

This native chart compares official-test macro F1 across the eight frozen pipelines. The HOG SVM row had already won validation selection. It reaches 90.91 percent test accuracy and macro F1 0.9090. Relative to pixels, HOG increases macro F1 by 3.63 percentage points for LR, 2.26 for kNN and 1.65 for SVM. It reduces RF macro F1 by 0.35 points. The effect therefore depends on the classifier and preprocessing. These results come from one split and a compact search. Do not claim a statistically significant difference, a universal winner or improvement caused only by HOG in isolation.

Sources:
- results/full/test_summary.csv
- results/full/freeze.json
- presentation/evidence.json

## 12. Shirt recall remains the weakest

Presenter: **Neelanshu**

The chart shows official-test recall for the selected HOG SVM only. Each category has 1,000 test images. Shirt recall is 73.3 percent, which means 733 of the 1,000 true Shirts receive the Shirt label. Bag recall is 98.3 percent. The difference demonstrates why a single aggregate accuracy value is insufficient. Some upper-body categories share outlines and local edge patterns at 28 by 28 resolution. That is a plausible interpretation of the observed errors, not a proven causal mechanism. The next slide examines measured confusion counts rather than relying on visual impressions alone.

Sources:
- results/full/full_hog_svm_3fd9d87a_test_metrics.json
- results/full/per_class_recall.csv

## 13. Confusion among upper-body categories

Presenter: **Neelanshu**

Read the table by row. The actual category is on the left and each column gives its predicted category. The Other column combines the six categories not displayed separately, so every row still sums to 1,000. Of the true T-shirt/top images, 104 are predicted as Shirt. Of the true Shirts, 103 are predicted as T-shirt/top, 63 as Pullover and 64 as Coat. This mutual confusion explains much of the weakest recall. The diagonal counts are correct classifications. The table is a focused view of the full ten by ten confusion matrix, not a separate experiment or a newly selected subset for model tuning.

Sources:
- results/full/full_hog_svm_3fd9d87a_test_metrics.json
- presentation/evidence.json

## 14. Paired SVM prediction examples

Presenter: **Neelanshu**

These original test images compare the frozen pixel SVM and HOG SVM on identical sample IDs. The examples follow a deterministic first-ID rule within each outcome group. They do not estimate group prevalence. In the HOG-only-correct group, HOG succeeds where the pixel SVM fails. In the both-wrong group, fixed gradients still fail. Read the true label and both predictions before interpreting a shape. A single visual example can suggest a reason to investigate, but it does not establish a causal explanation or prove a dataset label is wrong. Group counts for the complete official test are in the saved error evidence.

Sources:
- results/full/error_example_ids.csv
- report/figures/error_examples.png
- src/clothing_ml/plots.py

## 15. Inference cost changes the trade-off

Presenter: **Neelanshu**

The table reports median end-to-end inference time per image, amortized over batches. It includes normalization, feature extraction, training-fitted scaling where applicable and classifier prediction. It excludes disk reads and model loading. Measurements use official-test IDs zero through 1,023, batches of 128, one warm-up batch and five repeats, with one native thread on the recorded ARM64 macOS machine. HOG SVM is the strongest measured classifier but costs 21.7090 milliseconds per image in this setting. Pixel LR costs 0.0014 milliseconds, with lower accuracy. These are batch costs on one machine, not interactive single-image latency or guaranteed deployment performance. Pipeline-only timing and repeat ranges remain in the evidence files.

Sources:
- results/full/timing.json
- results/full/test_summary.csv

## 16. Reproducible Phase 1 evidence

Presenter: **Nakul**

The repository retains source modules, exact saved split membership, dataset checksums, candidate logs and the hashed selection manifest. Compressed predictions preserve original IDs and true labels. An independent audit rebuilds the reported metrics from the prediction files, while the executed notebook presents the analysis with saved outputs. Eight integrity tests passed, including split checks, metric checks, scaling behavior, prediction alignment and the test gate. The GitHub workflow repeats the lightweight checks. A fresh reproduction writes to an isolated result directory so that the checked-in frozen evidence stays intact. Large model bundles and downloaded data stay local. The deck itself reads recorded results and does not retrain any model.

Sources:
- results/full/evidence_audit.json
- tests/test_integrity.py
- .github/workflows/checks.yml
- notebooks/00_phase1_master.ipynb
- docs/protocol.md

## 17. Conclusions and Phase 2 proposal

Presenter: **Nakul**

Phase 1 establishes a controlled comparison of two feature representations and four classical classifiers. The validation-selected HOG SVM achieves 90.91 percent test accuracy, and HOG helps three model families under this protocol. Shirt remains the weakest class and inference cost increases substantially for the strongest model. Limits include one split, a small search, model-specific scaling and standardized low-resolution images. Phase 2 should test a small two- or three-block CNN with pooling and dropout, training-only validation for early stopping and architecture choices, and the same evaluation measures. The learned hierarchy may help distinguish similar clothes, but that remains an unmeasured hypothesis. Test results already seen in Phase 1 must not guide CNN tuning.

Sources:
- results/full/test_summary.csv
- results/full/full_hog_svm_3fd9d87a_test_metrics.json
- docs/course_instructions.md
- report/main.tex

## 18. Selected references

Presenter: **Nakul**

The full literature review and bibliography are available in the report and repository. Fashion-MNIST supplies the benchmark. Dalal and Triggs motivate HOG. Breiman introduces Random Forests. Cortes and Vapnik supply support-vector-network background. Scikit-learn provides the implementation and evaluation tools. LeCun and colleagues provide background for the later CNN. The team literature notes distinguish full-text reading from abstract-only access. The original HOG paper studies human detection, so cite it as the source of the descriptor rather than direct evidence about this clothing experiment.

Sources:
- docs/literature_review.md
- report/references.bib

## 19. All eight finalists

Presenter: **Samarth and Neelanshu** (backup)

This backup table shows validation macro F1 alongside official-test accuracy and macro F1 for every finalist. All final rows use 48,000 fit examples and the same 12,000 validation IDs. HOG SVM has the highest validation macro F1, 0.9114, which selected it before test access. Its test F1 is 0.9090. Compare like-for-like model families when explaining the feature effect. These values are rounded for display, while the CSVs retain full precision. The selection manifest records exact artifacts and parameters.

Sources:
- results/full/validation_summary.csv
- results/full/test_summary.csv

## 20. Selected settings and search scope

Presenter: **Samarth** (backup)

The table lists the settings selected separately for each feature-model pair, plus the scaling policy. The compact search examines LR C 0.1 and 1, kNN k 3 and 5, RF depth 20 and None, and SVM C 1 and 10. RF keeps 200 trees and square-root feature sampling. LR uses LBFGS and at most 2,000 iterations. kNN uses distance weighting and brute-force Euclidean distance. SVM uses RBF, gamma scale and probability false. Other defaults and installed versions are captured in the saved run records. A small grid is a resource-conscious development choice and does not prove that each model family reached its best possible performance.

Sources:
- results/full/freeze.json
- results/budget/validation_candidates.csv
- src/clothing_ml/models.py

## 21. Model size and complexity

Presenter: **Nakul** (backup)

Classical models need different complexity descriptions. LR has learned coefficient and intercept scalars. kNN stores training feature values and has no comparable learned coefficient count. RF complexity here counts all tree nodes. SVM uses the stored support-vector and decision-array scalar count as a storage proxy. None of these quantities is directly interchangeable with neural-network trainable parameters. The right column reports compressed serialized bundle size in MiB, which is a common artifact measure but is not peak RAM. HOG SVM stores 20,916 support vectors. Scaler mean and scale arrays are counted separately in the evidence. Do not interpret the larger quantity as a general measure of statistical quality.

Sources:
- results/full/complexity.json
- results/full/freeze.json
- results/full/test_summary.csv

## 22. Viva and saved-prediction demo

Presenter: **All five members** (backup)

For an offline demo, open the executed master notebook or its HTML companion. Find original test ID zero in the saved predictions. Its true category is Ankle boot and the selected HOG SVM predicts Ankle boot. Show the matching ID in another model’s prediction file to explain alignment. Recompute accuracy as the count of correct predictions divided by 10,000, using the saved CSV rather than training again. For viva practice, each member should explain the fit/validation/test distinction, the HOG dimension calculation and one weakness of the selected model. If asked whether CNN is better, say it has not been evaluated yet. The saved artifacts support the demonstration without network access.

Sources:
- notebooks/00_phase1_master.ipynb
- notebooks/00_phase1_master.html
- results/full/predictions/full_hog_svm_3fd9d87a_test.csv.gz
- docs/START_HERE_NAKUL.md