"""Nearest training neighbours behind frozen kNN validation errors (validation only).

Refits each frozen kNN row on the same 48,000 fit IDs, confirms the refit model
reproduces every saved validation prediction, then inspects which fit images
voted for the wrong label. Official test data is never loaded.
"""
from __future__ import annotations

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.evaluate import predict_batches, validate_predictions
from clothing_ml.experiments import validate_manifest
from clothing_ml.features import cached_features
from clothing_ml.models import make_model
from clothing_ml.plots import save_figure, setup_style
from clothing_ml.utils import ROOT, read_json, sha256, write_json

PANEL_ROWS = 6
result = ROOT / "results/full"
manifest = read_json(result / "freeze.json")
validate_manifest(manifest)
images, labels = load_official("train")
split_path = ROOT / "data/full_indices.npz"
with np.load(split_path) as indices:
    fit_ids, val_ids = indices["train_idx"], indices["val_idx"]

setup_style()
summary = {"source": "official_train validation partition only; official test not loaded",
           "split_sha256": sha256(split_path), "rows": {}}
for row in [r for r in manifest["finalists"] if r["model"] == "knn"]:
    feature, run_id, k = row["feature"], row["run_id"], row["parameters"]["n_neighbors"]
    Xfit, _ = cached_features(images[fit_ids], fit_ids, "official_train", feature)
    Xval, _ = cached_features(images[val_ids], val_ids, "official_train", feature)
    pipe = make_model("knn", feature, row["parameters"]).fit(Xfit, labels[fit_ids])
    saved = validate_predictions(pd.read_csv(result / "predictions" / f"{run_id}_val.csv.gz"),
                                 val_ids, labels, "val", run_id)
    refit = predict_batches(pipe, Xval)
    if not np.array_equal(refit, saved.y_pred.to_numpy()):
        raise ValueError(f"Refit {run_id} does not reproduce its saved validation predictions")

    errors = np.flatnonzero(refit != labels[val_ids])
    distances, positions = pipe.named_steps["model"].kneighbors(pipe[:-1].transform(Xval[errors]))
    neighbour_labels = labels[fit_ids[positions]]
    predicted, true = refit[errors], labels[val_ids[errors]]
    votes_for_prediction = (neighbour_labels == predicted[:, None]).sum(axis=1)
    votes_for_truth = (neighbour_labels == true[:, None]).sum(axis=1)

    pairs = Counter(zip(true.tolist(), predicted.tolist())).most_common(PANEL_ROWS)
    chosen = [min(np.flatnonzero((true == t) & (predicted == p)), key=lambda i: val_ids[errors[i]])
              for (t, p), _ in pairs]
    fig, axes = plt.subplots(len(chosen), k + 1, figsize=(1.55 * (k + 1), 1.8 * len(chosen)))
    panel = []
    for r, i in enumerate(chosen):
        query = int(val_ids[errors[i]])
        axes[r, 0].imshow(images[query], cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title(f"val ID {query}\ntrue {CLASS_NAMES[true[i]]}\npred {CLASS_NAMES[predicted[i]]}",
                             fontsize=7, color="#9B2C2C")
        neighbours = []
        for j in range(k):
            fit_id, label = int(fit_ids[positions[i, j]]), int(neighbour_labels[i, j])
            axes[r, j + 1].imshow(images[fit_id], cmap="gray", vmin=0, vmax=255)
            colour = "#285F99" if label == true[i] else "#243446"
            axes[r, j + 1].set_title(f"fit ID {fit_id}\n{CLASS_NAMES[label]}\nd={distances[i, j]:.2f}",
                                     fontsize=7, color=colour)
            neighbours.append({"fit_id": fit_id, "label": CLASS_NAMES[label], "distance": float(distances[i, j])})
        panel.append({"val_id": query, "true": CLASS_NAMES[true[i]], "predicted": CLASS_NAMES[predicted[i]],
                      "neighbours": neighbours})
        for axis in axes[r]:
            axis.axis("off")
    fig.suptitle(f"{feature.upper()} kNN (k={k}, distance-weighted): validation errors and their "
                 f"{k} nearest fit images\nOne row per most frequent true/predicted pair; "
                 "blue neighbour titles carry the true label", fontsize=9)
    fig.tight_layout()
    save_figure(fig, ROOT / "figures/full", f"knn_neighbours_{feature}")

    summary["rows"][feature] = {
        "run_id": run_id, "k": k, "fit_n": len(fit_ids), "validation_n": len(val_ids),
        "refit_reproduces_saved_validation_predictions": True,
        "validation_errors": int(len(errors)),
        "errors_with_majority_of_neighbours_carrying_predicted_label": int((votes_for_prediction > k // 2).sum()),
        "errors_with_no_neighbour_carrying_true_label": int((votes_for_truth == 0).sum()),
        "mean_neighbours_carrying_true_label": float(votes_for_truth.mean()),
        "most_frequent_error_pairs": [{"true": CLASS_NAMES[t], "predicted": CLASS_NAMES[p], "count": c}
                                      for (t, p), c in pairs],
        "panel": panel,
    }
    print(f"{run_id}: refit matches all {len(val_ids)} saved validation predictions; "
          f"{len(errors)} errors, {summary['rows'][feature]['errors_with_no_neighbour_carrying_true_label']} "
          "with no true-label neighbour", flush=True)

write_json(ROOT / "figures/full/knn_neighbours.json", summary)
print("Wrote figures/full/knn_neighbours_{hog,pixels}.png/.pdf and knn_neighbours.json")
