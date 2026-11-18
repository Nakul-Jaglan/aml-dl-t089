"""Neelanshu's Phase 2 evaluation helpers: paired groups, label pairs, bootstrap, ID validation."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from clothing_ml.evaluate import label_metrics, prediction_frame, validate_predictions
from clothing_ml.phase2.evaluation import (GROUPS, SHIRT, TSHIRT, label_pair_counts, macro_f1_from_pairs,
                                           mcnemar_exact, outcome_groups, paired_bootstrap, per_class)


def synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    a = np.where(rng.random(n) < 0.9, y, rng.integers(0, 10, n))
    b = np.where(rng.random(n) < 0.85, y, rng.integers(0, 10, n))
    return y, a, b


def test_outcome_groups_partition_and_reproduce_accuracy_delta():
    y, a, b = synthetic()
    counts, label = outcome_groups(y, a, b)
    assert sum(counts.values()) == len(y) and set(label) <= set(GROUPS)
    delta = (counts["Only CNN correct"] - counts["Only HOG + SVM correct"]) / len(y)
    assert delta == pytest.approx((a == y).mean() - (b == y).mean(), abs=1e-12)


def test_label_pair_counts_match_confusion_cells():
    y, a, _ = synthetic()
    cm = np.asarray(label_metrics(y, a)["confusion"])
    rows = {(r["true_label"], r["pred_label"]): r for r in label_pair_counts(y, a)}
    assert rows[(SHIRT, TSHIRT)]["count"] == cm[SHIRT, TSHIRT]
    assert rows[(TSHIRT, SHIRT)]["count"] == cm[TSHIRT, SHIRT]
    assert rows[(SHIRT, TSHIRT)]["true_support"] == (y == SHIRT).sum()


def test_fast_macro_f1_and_per_class_match_sklearn():
    y, a, _ = synthetic()
    expected = f1_score(y, a, labels=np.arange(10), average="macro", zero_division=0)
    assert macro_f1_from_pairs(y, a) == pytest.approx(expected, abs=1e-12)
    _, recall, f1, support = per_class(label_metrics(y, a))
    assert f1.mean() == pytest.approx(expected, abs=1e-12)
    assert support.sum() == len(y) and recall[SHIRT] == pytest.approx((a[y == SHIRT] == SHIRT).mean())


def test_paired_bootstrap_is_seeded_and_brackets_observed_delta():
    y, a, b = synthetic(2000)
    first, second = paired_bootstrap(y, a, b, repeats=200), paired_bootstrap(y, a, b, repeats=200)
    assert first == second
    low, high = first["accuracy_delta_ci95"]
    assert low <= (a == y).mean() - (b == y).mean() <= high


def test_mcnemar_symmetric_case_is_not_significant():
    assert mcnemar_exact(10, 10)["two_sided_p"] == pytest.approx(1.0)
    assert mcnemar_exact(40, 5)["two_sided_p"] < 0.001


def test_validation_rejects_wrong_partition_and_relabelled_rows():
    labels = np.arange(30) % 10
    ids = np.arange(20)
    frame = prediction_frame("r", "test", ids, labels[ids], labels[ids])
    assert len(validate_predictions(frame, ids, labels, "test", "r")) == 20
    wrong_split = frame.assign(source_split="official_train")
    with pytest.raises(ValueError):
        validate_predictions(wrong_split, ids, labels, "test", "r")
    shifted = frame.assign(sample_id=frame.sample_id + 10)  # same CSV shape, other images
    with pytest.raises(ValueError):
        validate_predictions(shifted, ids, labels, "test", "r")
    relabelled = frame.copy()
    relabelled.loc[0, "y_true"] = (relabelled.loc[0, "y_true"] + 1) % 10
    with pytest.raises(ValueError):
        validate_predictions(relabelled, ids, labels, "test", "r")
    with pytest.raises(ValueError):
        validate_predictions(pd.concat([frame, frame.iloc[:1]]), ids, labels, "test", "r")


def test_frozen_lr_heads_are_bundles_with_a_pipeline():
    """Regression: final_test called .predict on the joblib dict instead of bundle['pipeline']."""
    import joblib
    from clothing_ml.utils import ROOT, read_json
    freeze = read_json(ROOT / "results/phase2/full/freeze.json")
    for row in (r for r in freeze["test_rows"] if r["kind"] == "lr_head"):
        bundle = joblib.load(ROOT / row["artifact_path"])
        assert isinstance(bundle, dict) and hasattr(bundle["pipeline"], "predict")
        width = {"emb_lr": 64, "hog_lr": 1296, "fusion_lr": 1360}[row["row"]]
        assert bundle["pipeline"].predict(np.zeros((2, width), dtype=np.float32)).shape == (2,)
