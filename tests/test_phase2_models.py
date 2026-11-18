"""Unit tests for Phase 2 SmallCNN architecture, parameter counts, and feature interface.

Owned by Puneet.
Verifies the exact contracts, hand-checked parameter arithmetic, forward/backward passes,
and embedding output shapes specified in Phase 2 Work Guide.
"""
from pathlib import Path
import tempfile
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from clothing_ml.phase2.models import (  # noqa: E402
    SmallCNN,
    build_model,
    count_parameters,
    get_parameter_breakdown,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
    shape_table,
)


@pytest.mark.parametrize(
    "depth,channels,expected_params,expected_side",
    [
        (2, None, 220_234, 7),
        (3, None, 167_114, 3),
        (3, (32, 64, 64), 93_322, 3),
    ],
)
def test_small_cnn_hand_counted_parameters(depth, channels, expected_params, expected_side):
    """Reproduce and assert exact hand-calculated parameter arithmetic."""
    model = SmallCNN(depth=depth, channels=channels)
    assert model.spatial_side == expected_side
    measured_params = count_parameters(model)
    assert measured_params == expected_params, (
        f"Depth {depth} expected {expected_params} parameters, got {measured_params}"
    )
    assert parameter_count(model) == expected_params

    # Verify breakdown arithmetic
    breakdown = get_parameter_breakdown(model)
    assert breakdown["verified"] is True
    assert breakdown["total_trainable_parameters"] == expected_params


def test_small_cnn_invalid_depth():
    """Verify that unsupported depths raise ValueError."""
    with pytest.raises(ValueError, match="depth must be 2 or 3"):
        SmallCNN(depth=1)

    with pytest.raises(ValueError, match="depth must be 2 or 3"):
        SmallCNN(depth=4)


@pytest.mark.parametrize("depth", [2, 3])
def test_small_cnn_forward_backward_pass(depth):
    """Verify synthetic forward pass, finite loss, non-zero gradients, and encode/embed contract."""
    torch.manual_seed(42)
    batch_size = 4
    model = SmallCNN(depth=depth, dropout=0.30)
    model.train()

    # Synthetic batch
    x = torch.rand(batch_size, 1, 28, 28)
    y = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    # 1. Forward pass
    logits = model(x)
    assert logits.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {logits.shape}"

    # 2. Loss and backward
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()
    assert model.blocks[0].weight.grad is not None, "Gradient for first conv layer is None"
    assert torch.isfinite(model.blocks[0].weight.grad).all(), "Gradients contain NaN or Inf"

    # 3. Eval mode & feature interface
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode(x)
        assert embeddings.shape == (batch_size, 64), f"Expected shape ({batch_size}, 64), got {embeddings.shape}"
        assert torch.equal(embeddings, model.embed(x)), "encode(x) must match embed(x)"
        assert (embeddings >= 0).all(), "ReLU embeddings should be non-negative"

        # Deterministic predictions in eval mode
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)


def test_small_cnn_dropout_ablation_behavior():
    """Verify that dropout rate 0.0 behaves identically in train and eval modes."""
    torch.manual_seed(42)
    x = torch.rand(8, 1, 28, 28)

    model_no_drop = SmallCNN(depth=2, dropout=0.0)
    model_no_drop.train()
    out_train = model_no_drop(x)

    model_no_drop.eval()
    with torch.inference_mode():
        out_eval = model_no_drop(x)

    # With dropout=0.0, train and eval passes on identical weights should match
    torch.testing.assert_close(out_train, out_eval)


def test_small_cnn_shape_table():
    """Verify shape_table function outputs valid layers."""
    model = SmallCNN(depth=2)
    table = shape_table(model)
    assert len(table) > 0
    assert table[0]["parameters"] == 320


def test_small_cnn_checkpoint_roundtrip():
    """Verify atomic checkpoint saving and restoring with metadata fidelity."""
    model = SmallCNN(depth=3, dropout=0.15)
    dummy_config = {"lr": 0.001, "batch_size": 128}
    dummy_metrics = {"macro_f1": 0.9125, "accuracy": 0.9150}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        save_checkpoint(
            path=ckpt_path,
            model=model,
            config=dummy_config,
            epoch=12,
            metrics=dummy_metrics,
        )
        assert ckpt_path.exists()

        loaded_model, payload = load_checkpoint(ckpt_path, map_location="cpu")
        assert loaded_model.depth == 3
        assert loaded_model.dropout_rate == 0.15
        assert payload["epoch"] == 12
        assert payload["config"] == dummy_config
        assert payload["metrics"] == dummy_metrics
        assert count_parameters(loaded_model) == 167_114

        # Verify weights match
        x = torch.rand(2, 1, 28, 28)
        model.eval()
        loaded_model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(x), loaded_model(x))
            torch.testing.assert_close(model.encode(x), loaded_model.encode(x))


def test_existing_checkpoints_load_cleanly():
    """Verify that committed checkpoints in artifacts/phase2/full/ load into SmallCNN."""
    ckpt_path = Path("artifacts/phase2/full/full_cnn3_s42_80a2edee.pt")
    if not ckpt_path.exists():
        pytest.skip("Full checkpoint not present")
    model, bundle = load_checkpoint(ckpt_path)
    assert model.depth == 3
    assert count_parameters(model) == 167_114
    x = torch.zeros(1, 1, 28, 28)
    with torch.inference_mode():
        logits = model(x)
        assert logits.shape == (1, 10)
