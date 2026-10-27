# Related work and literature notes

This is a source-backed draft for review, not a claim that every teammate has
personally read the papers. References were checked on 27 October 2026.
Paper findings below are separated from this repository's experiment results.

## Research question

For basic clothing-category recognition, does a fixed edge descriptor improve
on raw intensities consistently, or does its usefulness depend on the classifier?
The controlled experiment crosses two representations with four model families.
It is a reproducible empirical comparison, not a newly proposed algorithm.

## Research sources

### Xiao, Rasul and Vollgraf (2017): Fashion-MNIST

The paper introduces a compact fashion-image benchmark intended to retain the
accessibility and format of MNIST while changing the visual classification
problem. It supports using a controlled public dataset for this project. Its
benchmark results are context, not a target or a substitute for our own runs.
The dataset does not by itself establish performance on unprocessed real catalogue
photos.

- Paper: https://arxiv.org/abs/1708.07747
- Readable full text: https://arxiv.org/html/1708.07747v2
- Official data/label map: https://github.com/zalandoresearch/fashion-mnist
- Reviewed: abstract, dataset construction, experiment description.
- Use here: data provenance and benchmark motivation; observed dataset checks
  are generated independently in `data/training_audit.json`.

### Dalal and Triggs (2005): Histograms of Oriented Gradients for Human Detection

This work studies local gradient histograms and contrast normalization for
pedestrian detection using an SVM. It motivates preserving local edge layout
instead of relying only on intensities. Our application adapts the descriptor
to small clothing images; their detection task and performance cannot be
transferred directly. HOG settings are therefore declared and checked here.

- DOI: https://doi.org/10.1109/CVPR.2005.177
- Paper: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/dalal-cvpr-05.pdf
- Reviewed: abstract and descriptor/method overview; not a replication of all
  original ablations.
- Use here: rationale for an explicit local-edge baseline. Our HOG dimension is
  derived from our own 28x28/4x4/2x2 configuration.

### Breiman (2001): Random Forests

Breiman analyzes ensembles of randomized trees, including the roles of tree
strength and dependence. This motivates an ensemble baseline that differs
from linear scoring and neighbor voting. We use the documented scikit-learn
implementation and validate depth choices; no forest result from the paper
is reported as our clothing result.

- Author-hosted paper: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
- DOI: https://doi.org/10.1023/A:1010933404324
- Reviewed: abstract, introductory definition and random-feature motivation.
- Use here: theoretical grounding for randomized tree ensembles.

### Cortes and Vapnik (1995): Support-vector networks

The publisher abstract describes nonlinear feature mappings and separating
surfaces extended to nonseparable data. This grounds a margin-based comparator.
Our ten-class RBF SVC configuration is an implementation choice, not the
original paper's experiment.

- Publisher: https://link.springer.com/article/10.1007/BF00994018
- DOI: https://doi.org/10.1007/BF00994018
- Reviewed: publisher abstract and bibliographic record. Full publisher text
  was paywalled; detailed implementation claims use official scikit-learn docs.
- Use here: high-level SVM motivation, without pretending to have audited every
  original derivation or benchmark.

### LeCun, Bottou, Bengio and Haffner (1998): Gradient-Based Learning Applied to Document Recognition

This is background for the later move from fixed descriptors to learned spatial
features. It concerns document recognition, so it is not evidence of a clothing
CNN result. No neural network is trained in Phase 1.

- Author-hosted record/PDF: https://leon.bottou.org/papers/lecun-98h
- DOI: https://doi.org/10.1109/5.726791
- Reviewed: bibliographic record and abstract context; deeper architecture study
  is reserved for Phase 2 preparation.

## Implementation and evaluation sources

- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- kNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- HOG: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
- Leakage: https://scikit-learn.org/stable/common_pitfalls.html
- Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

These pages support the actual APIs, multiclass behavior, hyperparameters and
metric definitions. Installed versions, rather than a moving `stable` URL alone,
are recorded in `requirements-lock.txt` and each experiment environment record.

## Synthesis for the report

Raw intensities offer a minimally engineered reference. HOG injects a fixed local
structure before training. LR, kNN, RF and RBF SVM then expose different ways of
using those representations. The gap investigated here is whether the benefit
of fixed local edges survives across these decision rules and confusing clothing
categories. Test results are needed to answer that question; a favorable HOG
example alone is insufficient. A future CNN could learn task-specific spatial
features, but its advantage remains an untested hypothesis until Phase 2.

No quoted passage or third-party accuracy is copied into our results table.

