"""Target the ways a near-duplicate audit can quietly report the wrong thing."""
import numpy as np
import pytest

from clothing_ml.near_duplicates import audit, describe, exact_rms, nearest_neighbour


def test_identical_rows_have_zero_distance_and_chunking_does_not_change_results():
    rng = np.random.default_rng(0)
    reference = rng.integers(0, 256, (40, 28, 28), dtype=np.uint8)
    query = reference[[7, 19, 3]]
    distance, index = nearest_neighbour(query, reference)
    np.testing.assert_allclose(distance, 0, atol=1e-4)
    assert index.tolist() == [7, 19, 3]
    chunked, chunked_index = nearest_neighbour(query, reference, chunk=1)
    np.testing.assert_allclose(distance, chunked, atol=1e-4)
    assert index.tolist() == chunked_index.tolist()


def test_distance_is_rms_per_pixel_not_a_raw_norm():
    left = np.zeros((1, 28, 28), dtype=np.uint8)
    right = np.full((1, 28, 28), 10, dtype=np.uint8)
    distance, _ = nearest_neighbour(left, right)
    assert distance[0] == pytest.approx(10.0, abs=1e-4)
    assert exact_rms(left[0], right[0]) == pytest.approx(10.0)


def test_exact_rms_is_integer_accurate_where_float32_would_drift():
    left = np.full((28, 28), 255, dtype=np.uint8)
    right = left.copy()
    right[0, 0] = 254
    assert exact_rms(left, right) == pytest.approx(np.sqrt(1 / 784))
    assert describe(left, right) == {"differing_pixels": 1, "max_absolute_difference": 1,
                                     "mean_absolute_difference": round(1 / 784, 4)}


def test_audit_reports_the_true_nearest_pair_and_keeps_ids_attached():
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 256, (12, 28, 28), dtype=np.uint8)
    query = rng.integers(0, 256, (5, 28, 28), dtype=np.uint8)
    planted = reference[4].copy()
    planted[0, 0] = min(255, int(planted[0, 0]) + 1)  # near-duplicate, not byte-identical
    query[2] = planted

    reference_ids = np.arange(100, 112)
    query_ids = np.arange(900, 905)
    report = audit("toy", query, query_ids, np.zeros(5, dtype=int), "official_test",
                   reference, reference_ids, np.zeros(12, dtype=int), "official_train")

    closest = report["closest_pairs"][0]
    assert closest["query"] == "official_test:902"
    assert closest["neighbour"] == "official_train:104"
    assert closest["differing_pixels"] == 1
    assert report["counts_below"]["1.0"] == 1
    assert report["n_query"] == 5 and report["n_reference"] == 12
    assert not np.array_equal(query[2], reference[4]), "planted pair must not be byte-identical"
