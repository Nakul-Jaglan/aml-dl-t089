"""Feature and LR/kNN pipeline contracts that the eight-row comparison depends on."""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from clothing_ml import features
from clothing_ml.features import HOG_SETTINGS, cached_features, make_features, normalized_images
from clothing_ml.models import make_model


def raw_batch(n=4, seed=11):
    return np.random.default_rng(seed).integers(0, 256, (n, 28, 28), dtype=np.uint8)


def test_hog_length_follows_from_declared_settings():
    cells = 28 // HOG_SETTINGS["pixels_per_cell"][0]
    blocks = cells - HOG_SETTINGS["cells_per_block"][0] + 1
    per_block = np.prod(HOG_SETTINGS["cells_per_block"]) * HOG_SETTINGS["orientations"]
    assert (cells, blocks, per_block) == (7, 6, 36)
    assert make_features(raw_batch(1), "hog").shape[1] == blocks * blocks * per_block == 1296


def test_pixels_are_exact_divide_by_255_in_row_major_order():
    raw = raw_batch()
    pixels = make_features(raw, "pixels")
    assert pixels.dtype == np.float32
    assert 0 <= pixels.min() and pixels.max() <= 1
    np.testing.assert_array_equal(pixels[2].reshape(28, 28) * 255, raw[2].astype(np.float32))
    assert pixels[0, 28] == np.float32(raw[0, 1, 0]) / 255


@pytest.mark.parametrize("feature", ["pixels", "hog"])
def test_features_are_deterministic_and_independent_of_batch_neighbours(feature):
    raw = raw_batch(5)
    first = make_features(raw, feature)
    np.testing.assert_array_equal(first, make_features(raw.copy(), feature))
    shuffled = make_features(raw[[4, 2, 0, 3, 1]], feature)
    np.testing.assert_array_equal(shuffled, first[[4, 2, 0, 3, 1]])


def test_blank_and_saturated_images_give_finite_hog():
    edge_cases = np.stack([np.zeros((28, 28), np.uint8), np.full((28, 28), 255, np.uint8)])
    result = make_features(edge_cases, "hog")
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, 0)


def test_hog_responds_to_orientation_so_input_must_stay_upright():
    image = np.zeros((1, 28, 28), np.uint8)
    image[0, 4:24, 13:15] = 255
    vertical = make_features(image, "hog")
    horizontal = make_features(np.transpose(image, (0, 2, 1)).copy(), "hog")
    assert not np.allclose(vertical, horizontal)


@pytest.mark.parametrize("bad", [np.zeros((2, 28, 27), np.uint8), np.zeros((28, 28), np.uint8),
                                 np.zeros((2, 28, 28, 1), np.uint8)])
def test_rejects_wrong_input_shapes(bad):
    with pytest.raises(ValueError):
        normalized_images(bad)


def test_rejects_unknown_feature_name():
    with pytest.raises(ValueError, match="Unknown feature"):
        make_features(raw_batch(1), "hog7")


def test_feature_cache_round_trips_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "ROOT", tmp_path)
    raw, ids = raw_batch(3), np.array([10, 5, 7])
    computed, key = cached_features(raw, ids, "official_train", "hog")
    reloaded, same_key = cached_features(raw, ids, "official_train", "hog")
    assert key == same_key
    np.testing.assert_array_equal(computed, reloaded)
    np.testing.assert_array_equal(computed, make_features(raw, "hog"))
    _, reordered_key = cached_features(raw[::-1].copy(), ids[::-1].copy(), "official_train", "hog")
    assert reordered_key != key
    cache_file = next((tmp_path / "data/cache").glob(f"hog_{key[:20]}.npy"))
    np.save(cache_file, computed + 1)
    with pytest.raises(ValueError, match="integrity"):
        cached_features(raw, ids, "official_train", "hog")


@pytest.mark.parametrize("feature,knn_scaled", [("pixels", False), ("hog", True)])
def test_lr_and_knn_scaling_policy_matches_protocol(feature, knn_scaled):
    assert isinstance(make_model("lr", feature).named_steps["scale"], StandardScaler)
    knn = make_model("knn", feature, {"n_neighbors": 5})
    assert isinstance(knn.named_steps["scale"], StandardScaler) == knn_scaled
    settings = knn.named_steps["model"].get_params()
    assert (settings["weights"], settings["algorithm"], settings["p"]) == ("distance", "brute", 2)


@pytest.mark.parametrize("feature,dimension", [("pixels", 784), ("hog", 1296)])
def test_lr_learned_scalar_count_is_ten_times_d_plus_one(feature, dimension):
    raw = np.random.default_rng(3).integers(0, 256, (40, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10), 4)
    pipe = make_model("lr", feature, {"C": 0.1}).fit(make_features(raw, feature), labels)
    model = pipe.named_steps["model"]
    assert model.coef_.shape == (10, dimension)
    assert model.coef_.size + model.intercept_.size == 10 * (dimension + 1)


def test_distance_weighted_knn_returns_label_of_an_exact_training_match():
    raw = raw_batch(20, seed=5)
    labels = np.arange(20) % 10
    for feature in ["pixels", "hog"]:
        X = make_features(raw, feature)
        knn = make_model("knn", feature, {"n_neighbors": 5}).fit(X, labels)
        np.testing.assert_array_equal(knn.predict(X), labels)
