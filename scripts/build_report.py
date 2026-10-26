"""Generate the IEEE-format draft and presentation notes from frozen measured evidence."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.features import HOG_SETTINGS, normalized_images
from skimage.feature import hog
from clothing_ml.utils import ROOT, read_json

PROFILE = "full"
RESULTS = ROOT / "results" / PROFILE
REPORT = ROOT / "report"
GENERATED = REPORT / "generated"
FIGURES = REPORT / "figures"
for directory in [REPORT, GENERATED, FIGURES]:
    directory.mkdir(parents=True, exist_ok=True)
manifest = read_json(RESULTS / "freeze.json")
summary = pd.read_csv(RESULTS / "test_summary.csv")
validation = pd.read_csv(RESULTS / "validation_summary.csv")
recalls = pd.read_csv(RESULTS / "per_class_recall.csv")
complexities = {r["run_id"]: r for r in read_json(RESULTS / "complexity.json")}
selected_id = manifest["selected_run_id"]
selected = summary.set_index("run_id").loc[selected_id]
audit = read_json(ROOT / "data/training_audit.json")
env = read_json(RESULTS / "environment.json")
metrics = read_json(RESULTS / f"{selected_id}_test_metrics.json")
model_names = {"lr": "LR", "knn": "kNN", "rf": "RF", "svm": "RBF SVM"}
feature_names = {"pixels": "Pixels", "hog": "HOG"}


def escape(text):
    for old, new in [("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")]:
        text = str(text).replace(old, new)
    return text


def pct(value):
    return f"{100 * value:.2f}\\%"


rows = []
for row in manifest["finalists"]:
    measured = summary.set_index("run_id").loc[row["run_id"]]
    mark = r"$^*$" if row["run_id"] == selected_id else ""
    rows.append(f"{feature_names[row['feature']]} & {model_names[row['model']]}{mark} & {100*measured.accuracy:.2f} & {measured.macro_f1:.4f} & {measured.fit_seconds:.1f} " + r"\\")
(GENERATED / "main_table.tex").write_text("\n".join(rows) + "\n")
short_names = ["T/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
rows = []
for row in manifest["finalists"]:
    values = recalls[(recalls.stage == "test") & (recalls.run_id == row["run_id"])].sort_values("label").recall
    rows.append(f"{feature_names[row['feature']]} + {model_names[row['model']]} & " + " & ".join(f"{100*v:.1f}" for v in values) + r" \\")
(GENERATED / "recall_table.tex").write_text("\n".join(rows) + "\n")
timing_rows, complexity_rows = [], []
for row in manifest["finalists"]:
    measured = summary.set_index("run_id").loc[row["run_id"]]
    item = complexities[row["run_id"]]
    label = f"{feature_names[row['feature']]} + {model_names[row['model']]}"
    timing_rows.append(f"{label} & {measured.predict_ms_per_image:.4f} & {measured.end_to_end_ms_per_image:.4f} & {measured.serialized_bytes/2**20:.2f}" + r" \\")
    type_code = {"lr": "W", "knn": "X", "rf": "N", "svm": "S"}[row["model"]]
    count = f"{item['complexity_value']:,}"
    complexity_rows.append(f"{label} & {type_code} & {count}" + r" \\")
(GENERATED / "timing_table.tex").write_text("\n".join(timing_rows) + "\n")
(GENERATED / "complexity_table.tex").write_text("\n".join(complexity_rows) + "\n")

selected_recalls = recalls[(recalls.stage == "test") & (recalls.run_id == selected_id)].sort_values("recall")
weak = selected_recalls.iloc[0]
cm = np.asarray(metrics["confusion"])
off_diagonal = cm.copy()
np.fill_diagonal(off_diagonal, 0)
top = np.argsort(off_diagonal.ravel())[::-1][:3]
confusion_prose = "; ".join(f"{CLASS_NAMES[k//10]} predicted as {CLASS_NAMES[k%10]} ({off_diagonal.ravel()[k]} examples)" for k in top)
deltas = []
for model in ["lr", "knn", "rf", "svm"]:
    r = summary[summary.model == model].set_index("feature")
    deltas.append((model, float(100 * (r.loc["hog", "macro_f1"] - r.loc["pixels", "macro_f1"]))))
delta_prose = ", ".join(f"{model_names[m]}: {delta:+.2f}" for m, delta in deltas)
delta_macros = "\n".join([
    r"\newcommand{\SelectedName}{" + escape(f"{feature_names[selected.feature]} + {model_names[selected.model]}") + "}",
    r"\newcommand{\SelectedAccuracy}{" + pct(selected.accuracy) + "}",
    r"\newcommand{\SelectedFone}{" + f"{selected.macro_f1:.4f}" + "}",
    r"\newcommand{\WeakClass}{" + escape(weak["category"]) + "}",
    r"\newcommand{\WeakRecall}{" + pct(weak["recall"]) + "}",
    r"\newcommand{\SelectedLatency}{" + f"{selected.end_to_end_ms_per_image:.4f}" + "}",
    r"\newcommand{\DeltaText}{" + escape(delta_prose) + "}",
    r"\newcommand{\ConfusionText}{" + escape(confusion_prose) + "}",
    r"\newcommand{\LibraryVersion}{" + escape(env["packages"]["scikit-learn"]) + "}",
    r"\newcommand{\ImageVersion}{" + escape(env["packages"]["scikit-image"]) + "}",
])
(GENERATED / "values.tex").write_text(delta_macros + "\n")

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
# Publication figures use physical dimensions suited to IEEE columns. Do not
# shrink a 10-inch notebook chart into a 3.5-inch column with unreadable labels.
train_images, train_labels = load_official("train")
with np.load(ROOT / "data/full_indices.npz") as split:
    fit_ids, validation_ids = split["train_idx"], split["val_idx"]
fig, axis = plt.subplots(figsize=(4.2, 3.0))
x = np.arange(10)
axis.bar(x-.18, np.bincount(train_labels[fit_ids], minlength=10), .36, color="#285F99", label="Fit pool")
axis.bar(x+.18, np.bincount(train_labels[validation_ids], minlength=10), .36, color="#B86C16", label="Validation", hatch="//")
axis.set_xticks(x)
axis.tick_params(labelsize=10)
axis.set_xlabel("Category ID (official order)", fontsize=11)
axis.set_ylabel("Images", fontsize=11)
axis.legend(frameon=False, fontsize=10, loc="upper center", ncol=2)
axis.set_ylim(0, 6200)
fig.tight_layout()
fig.savefig(FIGURES / "class_counts.pdf", bbox_inches="tight")
plt.close(fig)
example_id = int(fit_ids[train_labels[fit_ids] == 6][0])
one = normalized_images(train_images[example_id:example_id+1])[0]
_, edge = hog(one, visualize=True, **HOG_SETTINGS)
fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.2))
for axis, image, title in zip(axes, [one, edge], [f"Fit shirt, ID {example_id}", "Fixed HOG view"]):
    axis.imshow(image, cmap="gray")
    axis.set_title(title, fontsize=11)
    axis.axis("off")
fig.tight_layout()
fig.savefig(FIGURES / "hog_example.pdf", bbox_inches="tight")
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
for axis, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro F1"]):
    for feature, offset, color in [("pixels", -.18, "#285F99"), ("hog", .18, "#B86C16")]:
        values = [summary[(summary.feature == feature) & (summary.model == model)][metric].iloc[0] for model in model_names]
        bars = axis.barh(np.arange(4)+offset, values, height=.34, color=color, label=feature_names[feature], hatch="//" if feature == "hog" else None)
        for bar, value in zip(bars, values):
            axis.text(value+.015, bar.get_y()+bar.get_height()/2, f"{value:.3f}", va="center", fontsize=10)
    axis.set_yticks(range(4), list(model_names.values()), fontsize=10)
    axis.tick_params(labelsize=10)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.13)
    axis.set_xlabel(title+" (fraction)", fontsize=11)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, .13, 1, 1))
fig.savefig(FIGURES / "model_comparison.pdf", bbox_inches="tight")
plt.close(fig)

chosen_classes = [0, 2, 4, 6]
sub = cm[np.ix_(chosen_classes, chosen_classes)]
fig, axis = plt.subplots(figsize=(4.8, 4.5))
im = axis.imshow(sub, cmap="Blues", vmin=0, vmax=1000)
axis.set_xticks(range(4), ["T-shirt/\ntop", "Pullover", "Coat", "Shirt"], fontsize=11)
axis.set_yticks(range(4), ["T-shirt/top", "Pullover", "Coat", "Shirt"], fontsize=11)
axis.set_xlabel("Predicted category")
axis.set_ylabel("True category")
for i in range(4):
    for j in range(4):
        axis.text(j, i, str(sub[i,j]), ha="center", va="center", fontsize=12,
                  color="white" if sub[i,j] > 600 else "#17293D")
fig.tight_layout()
fig.savefig(FIGURES / "focused_confusion.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "focused_confusion.png", dpi=180, bbox_inches="tight")
plt.close(fig)
images, labels = load_official("test")
predictions = {}
for feature in ["pixels", "hog"]:
    row = next(r for r in manifest["finalists"] if r["feature"] == feature and r["model"] == "svm")
    frame = pd.read_csv(RESULTS / "predictions" / f"{row['run_id']}_test.csv.gz").sort_values("sample_id")
    predictions[feature] = frame.y_pred.to_numpy()
pc, hc = predictions["pixels"] == labels, predictions["hog"] == labels
fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.7))
for group_no, (group_name, mask) in enumerate([("Only HOG correct", ~pc & hc), ("Both wrong", ~pc & ~hc)]):
    for axis, idx in zip(axes[group_no], np.flatnonzero(mask)[:2]):
        axis.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
        axis.axis("off")
        axis.set_title(f"{group_name}, ID {idx}\nTrue: {CLASS_NAMES[labels[idx]]}\nPixels: {CLASS_NAMES[predictions['pixels'][idx]]}\nHOG: {CLASS_NAMES[predictions['hog'][idx]]}", fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / "error_examples.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "error_examples.png", dpi=180, bbox_inches="tight")
plt.close(fig)

template = r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{array}
\usepackage[hidelinks]{hyperref}
\input{generated/values.tex}
\begin{document}
\title{Clothing Category Recognition with Pixels and HOG\\
\large Phase 1: Classical Machine Learning}
\ifdefined\IncludePrivateAuthors
\input{author_details.local.tex}
\else
\author{\IEEEauthorblockN{Nakul, Sudeshhna, Puneet, Samarth and Neelanshu}
\IEEEauthorblockA{Section A, CSAI Course Project}}
\fi
\maketitle

\begin{abstract}
We compare normalized pixel intensities and Histogram of Oriented Gradients
(HOG) for ten-class clothing recognition using Logistic Regression, k nearest
neighbors, Random Forest and an RBF Support Vector Machine. Fashion-MNIST's
official training partition supplies a seeded 48,000/12,000 fit/validation split;
the official 10,000-image test partition remains separate. A small development
search uses a shared 12,000-example fit subset, after which all eight final
pipelines are fitted on the same 48,000 examples. The validation-selected
\SelectedName{} obtains \SelectedAccuracy{} test accuracy and \SelectedFone{}
macro F1. Its weakest category recall is \WeakClass{} at \WeakRecall{}.
We report all ten class recalls, paired error examples, inference timing and
model-specific storage measures. The findings concern the declared benchmark
pipelines and do not establish a CNN result or real-catalogue deployment performance.
\end{abstract}

\begin{IEEEkeywords}
Fashion-MNIST, HOG, classical classification, validation, error analysis
\end{IEEEkeywords}

\section{Introduction}
Automatic category tags can help organize a small clothing inventory. An image
classifier maps an input picture to a category label, but visually similar
garments make aggregate accuracy an incomplete description of usefulness.
This project asks whether a fixed local-edge descriptor changes recognition
relative to raw intensities and whether the effect depends on the classifier.

The Phase 1 contribution is a controlled two-representation by four-classifier
comparison with traceable predictions, explicit validation selection and
class-specific error analysis. It is an empirical course baseline, not a new
learning algorithm. The broader project title includes a small CNN, but CNN
training and the classical-versus-CNN comparison are reserved for Phase 2.

\section{Related Work}
Fashion-MNIST provides a compact public benchmark for visual classification
and motivates a reproducible comparison on shared labeled images
\cite{xiao2017fashion}. Dataset benchmark scores are contextual evidence,
not this team's results or guaranteed targets.

Dalal and Triggs study HOG for human detection using locally normalized edge
histograms \cite{dalal2005hog}. Their task differs from clothing-category
classification, so we adapt the descriptor settings without transferring
their performance claims. Breiman's randomized-tree ensemble formulation
provides a contrasting model family \cite{breiman2001forests}. The
support-vector formulation motivates margin-based classification
\cite{cortes1995support}; our RBF, multiclass implementation follows the
documented software rather than reproducing that paper's experiments.

The scikit-learn implementation supplies consistent fit/predict interfaces
\cite{pedregosa2011sklearn}, while scikit-image documents the HOG extraction
used here \cite{skimagehog}. LeCun et al. provide background for later learned
spatial representations \cite{lecun1998gradient}. That direction remains a
Phase 2 hypothesis, with no CNN outcome inferred from Phase 1.

\section{Data Quality and Exploration}
The official data contain 28$\times$28 grayscale images in ten clothing
categories. We preserve the official 60,000-example training and 10,000-example
test partitions. Raw files are downloaded from the official repository and
verified against its published MD5 content checksums. Integer category IDs
remain intact and original row IDs identify every prediction.

Only official training data are used for exploratory choices. Executed checks
confirm uint8 pixels in the range 0--255, finite values, one label per image,
all ten labels, no all-zero images and no exact duplicate image groups.
Consequently, there are no exact fit/validation duplicate groups or conflicting
labels within exact-image groups in the training audit. Hash equality does not
detect all near-duplicates or establish semantic label correctness. No examples
are removed or relabeled.

A stratified random split with seed 42 yields 48,000 fit and 12,000 validation
examples, with 4,800 and 1,200 examples per category respectively. Saved indices
and file hashes prevent teammates from silently using different membership.
Figure~\ref{fig:counts} shows the class balance. Seeded galleries and mean
images, retained with the notebook, help check orientation and appearance
variation. Similar silhouettes motivate, but do not predetermine, the later
shirt/T-shirt/pullover error analysis.

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{figures/class_counts.pdf}
\caption{Class counts in the saved split of official training data. Both pools
are balanced. These are data checks rather than performance measurements.}
\label{fig:counts}
\end{figure}

\section{Features and Models}
\subsection{Fixed Image Representations}
Both paths accept upright raw uint8 images and divide intensities by 255 once.
The pixel path flattens the grid into 784 ordered values. This preserves values
and recoverable positions but supplies no built-in neighborhood or translation
structure to a conventional classifier.

HOG uses 9 orientation bins, 4$\times$4-pixel cells, 2$\times$2-cell blocks and
L2-Hys normalization, with no square-root transform. Seven cells per axis yield
six overlapping blocks per axis, hence
\begin{equation}
d_{HOG}=6\cdot6\cdot2\cdot2\cdot9=1296.
\end{equation}
HOG is therefore a structural representation rather than dimensionality
reduction in this configuration. Its extraction rule is fixed before label-based
training. Figure~\ref{fig:hog} illustrates its edge-direction view.

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{figures/hog_example.pdf}
\caption{A fit-partition shirt and its HOG visualization. The visualization
illustrates a fixed descriptor, not learned classifier attention.}
\label{fig:hog}
\end{figure}

StandardScaler is learned inside training Pipelines for Logistic Regression,
SVM and HOG kNN. Pixel kNN and both forest variants use the fixed features
directly. This declared, model-specific choice makes the comparison one of
complete pipelines. Fixed per-image extraction may be cached; learned means
and scales must be fitted only on the fit partition \cite{sklearnpitfalls}.

\subsection{Logistic Regression}
Class scores are $z_c=w_c^Tx+b_c$ and softmax gives
$p(c|x)=\exp(z_c)/\sum_j\exp(z_j)$. Multiclass cross-entropy with regularization
learns the coefficients. Smaller $C$ means stronger regularization. We use
LBFGS with at most 2,000 iterations and validate $C\in\{0.1,1\}$. This is linear
in the supplied representation and need not exploit the same information from
pixels and HOG.

\subsection{k Nearest Neighbors}
kNN stores the fit feature vectors and predicts from a distance-weighted vote
among the nearest examples under Euclidean distance. We validate $k\in\{3,5\}$
with brute-force search. A small fit time is not a low total-compute guarantee:
neighbor search and stored training data make inference and memory important.

\subsection{Random Forest}
We combine 200 bootstrap-trained trees, with square-root feature sampling and
minimum leaf size 1. Gini impurity is $1-\sum_c p_c^2$. Randomization diversifies
trees; depth controls each tree's flexibility. The implementation averages
tree class probabilities. We validate depth 20 versus unrestricted depth,
keeping random seed 42. Tree counts and nodes describe structure, rather than
a directly comparable linear-model parameter count.

\subsection{RBF Support Vector Machine}
The kernel is $K(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$. $C$ controls the
violation penalty and $\gamma$ the locality of similarity. SVC uses pairwise
classifiers for the ten classes. We validate $C\in\{1,10\}$ and hold
\texttt{gamma='scale'} fixed. Probability estimation is disabled because the
requested metrics need predicted labels. A recorded version-1.9 deprecation
warning concerns that explicit probability option; it is not a convergence
failure. Kernel fitting and support-vector prediction can be costly.

\section{Experimental Protocol}
\subsection{Development and Final Fitting}
The compact search has two candidates for each of eight representation/model
rows, or 16 development fits. Every candidate sees the same stratified 12,000
fit-pool examples and unchanged 12,000 validation examples. A separate
2,000/1,000 smoke path checks the interfaces and cannot unlock final testing.

Within each row, the development candidate with the highest validation macro
F1 is selected, with accuracy, lower validation prediction time and run ID as
tie-breakers. All eight selected configurations are then fitted afresh on the
full 48,000 fit examples. This is not a fit on training plus validation.
The overall model is selected from the eight full-profile validation results
using the same rule. Its ID, all model hashes, settings, split hash and scores
are frozen before loading official test content. Unconverged or failed runs
cannot be selected. Search logs, including the recorded API warning, remain
available.

All final models use identical fit, validation and test IDs. The test results
describe the eight frozen finalists and do not trigger a new selection or
hyperparameter change. The repository stores raw predictions in compressed
CSV files, checks their IDs and true labels, and regenerates metrics in the
executed master notebook.

\subsection{Metrics and Timing}
Accuracy is the correct fraction. For category $c$,
\begin{equation}
R_c=\frac{TP_c}{TP_c+FN_c},\qquad
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}.
\end{equation}
Macro F1 is $\frac{1}{10}\sum_c F1_c$, with labels 0--9 explicitly included and
undefined precision/F1 set to zero. Per-class support accompanies recall.
Rows of a confusion matrix are true classes; row-normalized diagonals equal
recall.

Fit time includes scaler and classifier fitting but excludes fixed feature
extraction and the development search. Inference uses already loaded models
and 1,024 fixed test images in batches of 128, one warmup and five repeats
under one native thread. Pipeline timing starts at cached fixed features.
End-to-end timing includes raw-image normalization, pixel/HOG extraction,
scaler and classifier, excluding disk reads and model loading. We report
median amortized milliseconds per image, not single-request latency.

The recorded environment uses Python 3.12, scikit-learn \LibraryVersion{} and
scikit-image \ImageVersion{} on an Apple ARM64 macOS machine with 16 GiB RAM.
Exact package versions and repeat durations are retained. Absolute timing
depends on hardware, software and concurrent system load.

\section{Results and Error Analysis}
\subsection{Aggregate and Class-Specific Results}
Table~\ref{tab:main} reports the eight full-size final rows. The model selected
before test evaluation, \SelectedName{}, reaches \SelectedAccuracy{} accuracy
and \SelectedFone{} macro F1 on official test data. Figure~\ref{fig:models}
compares the fixed pipelines within classifier families. The separate
most-frequent dummy is a validation reference, not one of the four required
model families.

\begin{table}[t]
\caption{Official test results; all rows fit on the same 48,000 images.
$^*$ marks the overall validation-selected model. Fit time is in seconds.}
\label{tab:main}\centering\small
\begin{tabular}{llrrr}\toprule
Feature & Model & Acc. (\%) & Macro F1 & Fit (s)\\\midrule
\input{generated/main_table.tex}
\bottomrule\end{tabular}
\end{table}

\begin{figure*}[t]
\centering\includegraphics[width=0.97\textwidth]{figures/model_comparison.pdf}
\caption{Pixels versus HOG within each frozen classifier. The comparison changes
the declared representation pipeline; it is not evidence that every possible
HOG/scaler configuration dominates every pixel configuration.}
\label{fig:models}
\end{figure*}

The observed HOG-minus-pixel macro-F1 changes, in percentage points, are
\DeltaText{}. The direction and magnitude should be read within each family,
with the fixed preprocessing and limited tuning budget in mind. These differences
are not claims of statistical significance. Table~\ref{tab:recall} preserves
all ten class recalls instead of hiding weak categories behind aggregate scores.
The validation-selected model's weakest category is \WeakClass{} with
\WeakRecall{} recall.

\begin{table*}[t]
\caption{Test recall in percent for every class; each category has 1,000 actual
test images. T/top means T-shirt/top and Boot means Ankle boot.}
\label{tab:recall}\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrrr}\toprule
Pipeline & T/top & Trouser & Pullover & Dress & Coat & Sandal & Shirt & Sneaker & Bag & Boot\\\midrule
\input{generated/recall_table.tex}
\bottomrule\end{tabular}
\end{table*}

\subsection{Observed Confusions and Paired Examples}
For the selected model, the three largest off-diagonal counts are
\ConfusionText{}. Figure~\ref{fig:confusion} focuses on four garment categories;
the complete ten-by-ten count and normalized matrices remain in the notebook
and figure directory. Shared outlines at low resolution are plausible sources
of ambiguity, but the observed mistakes alone do not prove their cause or
that the official labels are wrong.

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{figures/focused_confusion.pdf}
\caption{Selected model's test counts for true/predicted T-shirt/top, Pullover,
Coat and Shirt. This is a submatrix, so omitted-category errors are not shown
and displayed rows need not total 1,000. Full matrices are retained separately.}
\label{fig:confusion}
\end{figure}

We align pixel and HOG predictions on identical original IDs within each
classifier, divide images into both-correct, pixels-only-correct,
HOG-only-correct and both-wrong groups, and display the first original IDs in
each group. This rule prevents selectively showing only favorable HOG examples.
Figure~\ref{fig:errors} presents a subset for SVM; complete symmetric panels
and group counts are retained. These post-freeze test examples are descriptive
and were not used to modify the model.

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{figures/error_examples.pdf}
\caption{Paired RBF SVM examples. Each displayed row uses the first two
original test IDs in its outcome group. Additional pixels-only-correct and
both-correct examples are saved with the complete error-analysis artifacts.}
\label{fig:errors}
\end{figure}

\subsection{Inference and Model Storage}
Table~\ref{tab:timing} shows median amortized inference costs and compressed
bundle storage under the same serialization policy. The selected pipeline's
end-to-end measurement is \SelectedLatency{} ms per image. HOG extraction can
change the deployment trade-off even when a classifier predicts quickly.
Search time, training time, model loading and live single-image latency answer
different questions and should not be substituted for this measurement.

\begin{table}[t]
\caption{Inference milliseconds per image, batch-amortized. Storage includes
the compressed model bundle and metadata, in MiB.}
\label{tab:timing}\centering\small
\begin{tabular}{lrrr}\toprule
Pipeline & Predict & End to end & MiB\\\midrule
\input{generated/timing_table.tex}
\bottomrule\end{tabular}
\end{table}

\begin{table}[t]
\caption{Model-specific complexity evidence. Quantities have different meanings
and are not directly interchangeable parameter counts.}
\label{tab:complexity}\centering\small
\begin{tabular}{llr}\toprule
Pipeline & Type & Count\\\midrule
\input{generated/complexity_table.tex}
\bottomrule\end{tabular}
\end{table}

Table~\ref{tab:complexity} uses W for learned classifier coefficients plus
intercepts, X for stored kNN training-feature scalars, N for total forest nodes,
and S for SVC support-vector, dual-coefficient and intercept array scalars.
kNN has no learned coefficient count; this is recorded as not applicable rather
than computationally free. Scaler mean/scale counts are stored separately.
Storage also depends on data type, compression and implementation details.

\section{Limitations and Phase 2 Plan}
The results use a single fixed split, one seed, two candidates per row and one
HOG configuration. They do not establish globally optimal hyperparameters,
statistically significant differences, or robustness to domain shift. The
benchmark excludes much of the variation in production catalogue photographs.
Exact duplicate checks are narrower than a semantic or near-duplicate audit.
The representation comparison includes the disclosed scaler policy.

In Phase 2, a small two- or three-block CNN can test whether task-learned local
features improve the remaining category confusions. That experiment should
reuse the saved fit/validation/test membership, select architecture and early
stopping using training-derived validation data, and compare accuracy, macro
F1, recall and documented inference cost fairly. Since Phase 1 already reveals
test outcomes, architecture choices must not be tuned to those outcomes.
Any claimed hybrid method requires a meaningful technical rationale and its
own evidence; no hybrid improvement is claimed here.

\section{Reproducibility and Contributions}
The repository at \url{https://github.com/Nakul-Jaglan/aml-dl-t089} contains
source modules, meaningful integrity tests, saved split IDs, checksums,
validation candidate logs, a hashed freeze manifest, ID-aligned predictions,
metrics, figures and an executed master notebook. A fresh reproduction writes
to an isolated output directory so checked-in frozen evidence stays intact.
Large trained bundles, raw downloads and feature caches are regenerated locally
instead of committed to Git.

These initial artifacts were generated with AI assistance at Nakul's request.
Planned ownership assigns Nakul integration/research/report assembly,
Sudeshhna data/EDA, Puneet features/LR/kNN, Samarth RF/SVM/experiments and
Neelanshu evaluation/error analysis. These plans do not assert completed
independent work. Each member must review the relevant implementation, record
actual changes or verification, and demonstrate whole-project understanding
in the individual viva. Section A is confirmed. Student identifiers are included
only in the private submission copy; institutional affiliation details and any
faculty-specific template requirements still require team confirmation.

\section{Conclusion}
The controlled Phase 1 experiment compares eight classical pipelines under
shared data membership and validation-only model selection. The selected
\SelectedName{} achieves \SelectedAccuracy{} official-test accuracy and
\SelectedFone{} macro F1. Per-class recalls, paired errors and measured compute
costs qualify the aggregate result. HOG's contribution is empirical and
classifier-dependent under the declared preprocessing policy. Learned spatial
features remain a concrete question for Phase 2 rather than a result asserted
in advance.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""
# Inline measured table rows: LaTeX's robust \input can leave a token between
# the final row terminator and booktabs' \noalign in some engines.
for table in ["main_table.tex", "recall_table.tex", "timing_table.tex", "complexity_table.tex"]:
    template = template.replace(r"\input{generated/" + table + "}", (GENERATED / table).read_text().strip())
(REPORT / "main.tex").write_text(template)
private_details = ROOT / "private/student_details.json"
if private_details.exists():
    details = read_json(private_details)
    author_lines = [escape(s["name"]) + " (" + escape(s["roll_number"]) + ")" for s in details["students"]]
    author_tex = r"\author{\IEEEauthorblockN{" + author_lines[0] + ", " + author_lines[1] + r"\\" + ", ".join(author_lines[2:]) + "}" + "\n"
    author_tex += r"\IEEEauthorblockA{Section " + escape(details["section"]) + r", CSAI Course Project}}" + "\n"
    (REPORT / "author_details.local.tex").write_text(author_tex)
    (REPORT / "submission.tex").write_text("\\def\\IncludePrivateAuthors{1}\n\\input{main.tex}\n")
(REPORT / "references.bib").write_text(r"""@misc{xiao2017fashion,
  author={Han Xiao and Kashif Rasul and Roland Vollgraf},
  title={{Fashion-MNIST}: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  year={2017}, eprint={1708.07747}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/1708.07747}
}
@inproceedings{dalal2005hog,
  author={Navneet Dalal and Bill Triggs},
  title={Histograms of Oriented Gradients for Human Detection},
  booktitle={IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
  year={2005}, volume={1}, pages={886--893}, doi={10.1109/CVPR.2005.177}
}
@article{breiman2001forests,
  author={Leo Breiman}, title={Random Forests}, journal={Machine Learning},
  year={2001}, volume={45}, pages={5--32}, doi={10.1023/A:1010933404324}
}
@article{cortes1995support,
  author={Corinna Cortes and Vladimir Vapnik}, title={Support-vector networks},
  journal={Machine Learning}, year={1995}, volume={20}, pages={273--297},
  doi={10.1007/BF00994018}
}
@article{pedregosa2011sklearn,
  author={Fabian Pedregosa and others}, title={Scikit-learn: Machine Learning in Python},
  journal={Journal of Machine Learning Research}, year={2011}, volume={12},
  pages={2825--2830}, url={https://jmlr.org/papers/v12/pedregosa11a.html}
}
@article{lecun1998gradient,
  author={Yann LeCun and Leon Bottou and Yoshua Bengio and Patrick Haffner},
  title={Gradient-Based Learning Applied to Document Recognition},
  journal={Proceedings of the IEEE}, year={1998}, volume={86}, number={11},
  pages={2278--2324}, doi={10.1109/5.726791}
}
@misc{skimagehog,
  author={{scikit-image developers}}, title={{HOG} API documentation}, year={2026},
  howpublished={\url{https://scikit-image.org/docs/stable/api/skimage.feature.html}},
  note={Accessed 27 October 2026; installed version recorded with experiments}
}
@misc{sklearnpitfalls,
  author={{scikit-learn developers}}, title={Common pitfalls and recommended practices},
  year={2026}, howpublished={\url{https://scikit-learn.org/stable/common_pitfalls.html}},
  note={Accessed 27 October 2026; training-only preprocessing and pipelines}
}
""")

notes = f"""# Phase 1 presentation outline and speaker notes

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
Show `figures/full/model_comparison.png`. The preselected {selected.feature} +
{model_names[selected.model]} achieved {selected.accuracy:.2%} test accuracy and
{selected.macro_f1:.4f} macro F1. HOG-minus-pixel F1 differences in percentage points:
{delta_prose}. Avoid claiming statistical significance from one split.

## 9. Recall and garment confusion (Neelanshu)
Show `figures/full/per_class_recall.png` or the focused report confusion figure.
The selected model's weakest recall was {weak['category']}, {weak['recall']:.1%}.
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
"""
(ROOT / "docs/presentation_outline.md").write_text(notes)
print("Generated evidence-backed LaTeX source, report figures and presentation notes.")
