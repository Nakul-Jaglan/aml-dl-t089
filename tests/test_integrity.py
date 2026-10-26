"""Target failure modes that could invalidate the comparison, not library internals."""
import struct

import numpy as np
import pytest

from clothing_ml.data import parse_idx, saved_split
from clothing_ml.evaluate import label_metrics, prediction_frame, validate_predictions
from clothing_ml.experiments import final_test, validate_manifest
from clothing_ml.utils import stable_hash
from clothing_ml.features import make_features
from clothing_ml.models import make_model


def test_idx_rejects_truncated_and_wrong_shape():
    payload = struct.pack(">IIII", 2051, 2, 28, 28) + bytes(2 * 784)
    assert parse_idx(payload).shape == (2, 28, 28)
    with pytest.raises(ValueError):
        parse_idx(payload[:-1])
    with pytest.raises(ValueError):
        parse_idx(struct.pack(">IIII", 2051, 1, 27, 28) + bytes(27 * 28))


def test_saved_split_reuses_indices_and_rejects_changes(tmp_path):
    labels = np.repeat(np.arange(10), 10)
    tr, va, path = saved_split(labels, tmp_path)
    again, _, _ = saved_split(labels, tmp_path)
    assert np.array_equal(tr, again) and not np.intersect1d(tr, va).size
    np.savez(path, train_idx=tr[::-1], val_idx=va)
    with pytest.raises(ValueError, match="differs"):
        saved_split(labels, tmp_path)


def test_features_preserve_batch_order_and_reject_double_scaling():
    raw = np.random.default_rng(4).integers(0, 256, (3, 28, 28), dtype=np.uint8)
    for feature, length in [("pixels", 784), ("hog", 1296)]:
        result = make_features(raw, feature)
        assert result.shape == (3, length)
        np.testing.assert_allclose(result[1], make_features(raw[1:2], feature)[0])
        with pytest.raises(TypeError):
            make_features(raw.astype(float) / 255, feature)


def test_prediction_ids_are_aligned_and_checked():
    labels = np.arange(10)
    ids = np.array([3, 1, 8])
    frame = prediction_frame("toy", "val", ids, labels[ids], [3, 1, 0])
    reordered = validate_predictions(frame.iloc[::-1], ids, labels, "val", "toy")
    assert reordered.sample_id.tolist() == ids.tolist()
    broken = frame.copy()
    broken.loc[0, "sample_id"] = 1
    with pytest.raises(ValueError):
        validate_predictions(broken, ids, labels, "val", "toy")
    broken = frame.copy()
    broken.loc[0, "y_true"] = 0
    with pytest.raises(ValueError, match="authoritative"):
        validate_predictions(broken, ids, labels, "val", "toy")


def test_metrics_match_hand_calculation():
    metrics = label_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == .75
    assert metrics["recall"][:2] == [.5, 1]
    assert sum(map(sum, metrics["confusion"])) == 4


def test_scaler_fit_does_not_use_validation():
    X = np.arange(200, dtype=float).reshape(20, 10)
    y = np.tile([0, 1], 10)
    model = make_model("lr", "pixels", {"C": 1.0})
    model.fit(X, y)
    means = model.named_steps["scale"].mean_.copy()
    model.predict(np.full((2, 10), 1e6))
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, means)
    np.testing.assert_allclose(means, X.mean(axis=0))


def test_final_test_is_explicit_and_smoke_cannot_unlock():
    with pytest.raises(ValueError, match="allow-test"):
        final_test("budget", allow_test=False)
    with pytest.raises(ValueError, match="Smoke"):
        final_test("smoke", allow_test=True)


def test_freeze_manifest_detects_tampering():
    manifest = {"selection_stage": "validation_only", "test_used_for_selection": False,
                "selected_run_id": "toy", "finalists": []}
    manifest["manifest_sha256"] = stable_hash(manifest)
    assert validate_manifest(manifest) == manifest["manifest_sha256"]
    manifest["selected_run_id"] = "changed_after_freeze"
    with pytest.raises(ValueError, match="integrity"):
        validate_manifest(manifest)

