"""CNN data adapter and augmentation policy checks (Sudeshhna, Phase 2)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from clothing_ml.phase2.data import (  # noqa: E402
    POLICIES, ClothingImages, FashionIDDataset, RandomShift, check_batch, make_loader, make_transform, translate,
)


def toy_partition(n=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), n // 10)
    rng.shuffle(labels)
    return rng.integers(0, 256, (n, 28, 28), dtype=np.uint8), labels


def test_item_is_float32_scaled_once_with_original_id_and_label():
    raw, labels = toy_partition()
    raw[17] = 255
    image, label, original_id = ClothingImages(raw, labels, [17, 3])[0]
    assert image.dtype == torch.float32 and tuple(image.shape) == (1, 28, 28)
    assert float(image.max()) == 1.0  # scaled once: 255 -> 1.0, not 1/255
    assert (label, original_id) == (int(labels[17]), 17)


def test_adapter_rejects_double_scaling_bad_ids_and_misaligned_labels():
    raw, labels = toy_partition()
    with pytest.raises(TypeError):
        ClothingImages(raw.astype(np.float32) / 255, labels, [0])
    with pytest.raises(ValueError):
        ClothingImages(raw, labels, [1, 1])
    with pytest.raises(IndexError):
        ClothingImages(raw, labels, [len(raw)])
    with pytest.raises(ValueError):
        ClothingImages(raw, labels[:-1], [0])


def test_collated_batch_passes_handoff_checks_and_keeps_ids_after_shuffle():
    raw, labels = toy_partition()
    ids = np.arange(0, 60, 2)
    loader = make_loader(FashionIDDataset(raw, labels, ids), 8, shuffle=True, seed=42)
    seen = []
    for batch in loader:
        summary = check_batch(batch, raw, labels)
        seen.extend(batch[2].tolist())
    assert summary["image_dtype"] == "float32" and summary["label_dtype"] == "int64"
    assert sorted(seen) == ids.tolist() and seen != ids.tolist()  # shuffled, nothing lost
    assert len(loader) == 4  # drop_last=False keeps the final partial batch


def test_check_batch_catches_a_label_that_does_not_match_its_id():
    raw, labels = toy_partition()
    images, targets, ids = next(iter(make_loader(FashionIDDataset(raw, labels, [0, 1, 2]), 3, False)))
    targets = (targets + 1) % 10
    with pytest.raises(AssertionError):
        check_batch((images, targets, ids), raw, labels)


def test_translation_has_zero_fill_and_no_wraparound_for_every_offset():
    edge = torch.zeros(1, 28, 28)
    edge[0, 0, 27] = 1.0  # white pixel in the top-right corner
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            out = translate(edge, dx, dy)
            assert out.shape == (1, 28, 28) and 0.0 <= float(out.min()) and float(out.max()) <= 1.0
            inside = 0 <= 0 + dy < 28 and 0 <= 27 + dx < 28
            assert float(out.sum()) == (1.0 if inside else 0.0)  # leaves the frame, never reappears
    full = torch.ones(1, 28, 28)
    out = translate(full, 2, 1)
    assert out[:, :1, :].sum() == 0 and out[:, :, :2].sum() == 0  # uncovered border is zero
    assert out[:, 1:, 2:].min() == 1.0
    with pytest.raises(ValueError):
        translate(full, 0, 3)


def test_p50_policy_shifts_about_half_the_time_with_bounded_offsets():
    shift = RandomShift(p=0.5, seed=42)
    image = torch.rand(1, 28, 28)
    applied = []
    for _ in range(4000):
        out = shift(image)
        if shift.last_offset is None:
            assert torch.equal(out, image)
        else:
            assert max(map(abs, shift.last_offset)) <= 2
            applied.append(shift.last_offset)
    assert 0.46 < len(applied) / 4000 < 0.54
    assert {o for o in applied} == {(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)}
    again = RandomShift(p=0.5, seed=42)
    assert torch.equal(again(image), RandomShift(p=0.5, seed=42)(image))  # seeded


def test_policies_are_declared_and_fit_only():
    assert POLICIES["translate2_p50"]["p"] == 0.5 and POLICIES["translate2"]["p"] == 1.0
    assert make_transform("none") is None
    with pytest.raises(ValueError):
        make_transform("rotate15")
    raw, labels = toy_partition()
    val = FashionIDDataset(raw, labels, np.arange(20), "none")
    first = torch.cat([b[0] for b in make_loader(val, 7, False)])
    second = torch.cat([b[0] for b in make_loader(val, 7, False)])
    assert torch.equal(first, second)  # evaluation input is stable across passes
