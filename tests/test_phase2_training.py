"""Phase 2 runner checks: handoff shapes, checkpoint rule, reload fidelity, fusion joins, test guard."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from clothing_ml.phase2 import runner as runner_module  # noqa: E402
from clothing_ml.phase2.data import FashionIDDataset, make_loader, translate  # noqa: E402
from clothing_ml.phase2.fusion import joined_features, keyed  # noqa: E402
from clothing_ml.phase2.models import SmallCNN, build_model, parameter_count  # noqa: E402
from clothing_ml.phase2.runner import load_checkpoint, load_config, save_checkpoint_atomically  # noqa: E402
from clothing_ml.phase2.select import final_test  # noqa: E402


def toy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), n // 10)
    images = rng.integers(0, 256, (n, 28, 28), dtype=np.uint8)
    return images, labels


def test_loader_delivers_handoff_contract_in_stable_order():
    images, labels = toy_data()
    ids = np.array([5, 17, 3, 42])
    x, y, original = next(iter(make_loader(FashionIDDataset(images, labels, ids), 4, shuffle=False)))
    assert x.dtype == torch.float32 and tuple(x.shape) == (4, 1, 28, 28)
    assert y.dtype == torch.int64 and original.tolist() == ids.tolist()
    assert y.tolist() == labels[ids].tolist()
    assert float(x.max()) <= 1.0 and torch.allclose(x[0, 0], torch.from_numpy(images[5] / 255).float())
    with pytest.raises(ValueError):
        FashionIDDataset(images, labels, np.array([1, 1]))


def test_translation_is_a_bounded_zero_filled_shift_and_fit_only_by_config():
    image = torch.zeros(1, 28, 28)
    image[0, 10, 10] = 1.0
    shifted = translate(image, 2, -1)
    assert shifted.shape == (1, 28, 28) and shifted[0, 9, 12] == 1.0 and shifted.sum() == 1.0
    with pytest.raises(ValueError):
        translate(image, 3, 0)
    images, labels = toy_data()
    canonical = FashionIDDataset(images, labels, np.arange(10), "none")
    assert torch.equal(canonical[0][0], canonical[0][0])  # validation path is deterministic


def test_model_exposes_logits_and_64_value_embedding_before_dropout():
    for blocks, expected_flat in [(2, 64 * 7 * 7), (3, 128 * 3 * 3)]:
        model = SmallCNN(blocks=blocks, dropout=0.3).eval()
        x = torch.rand(5, 1, 28, 28)
        assert model(x).shape == (5, 10) and model.embed(x).shape == (5, 64)
        assert model.dense[1].in_features == expected_flat
        assert (model.embed(x) >= 0).all()  # after ReLU
        assert torch.allclose(model(x), model.classifier(model.embed(x)))  # eval: dropout is identity
    assert parameter_count(SmallCNN(2, 0.3)) == parameter_count(SmallCNN(2, 0.0))  # dropout adds no parameters


def test_config_digest_ignores_device_but_tracks_experiment_settings():
    a = load_config({"name": "cnn2"})
    assert load_config({"name": "cnn2", "device": "mps"})["config_sha256"] == a["config_sha256"]
    assert load_config({"name": "cnn2", "dropout": 0.0})["config_sha256"] != a["config_sha256"]
    with pytest.raises(ValueError):
        load_config({"name": "x", "augmentation": "rotate"})


def test_atomic_checkpoint_round_trip_reproduces_logits(tmp_path):
    torch.manual_seed(0)
    model = build_model({"blocks": 2, "dropout": 0.3}).eval()
    config = load_config({"name": "toy"})
    path = tmp_path / "best.pt"
    save_checkpoint_atomically(path, model, config, 3, {"macro_f1": 0.5, "accuracy": 0.5})
    assert not path.with_suffix(".tmp").exists()
    reloaded, bundle = load_checkpoint(path)
    x = torch.rand(4, 1, 28, 28)
    with torch.inference_mode():
        assert torch.equal(model(x), reloaded(x))
    assert bundle["epoch"] == 3 and bundle["arch"]["blocks"] == 2
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(path, expected_sha256="0" * 64)


def test_runner_keeps_earliest_best_epoch_and_stops_on_patience(tmp_path, monkeypatch):
    """Scripted validation scores exercise strict improvement, tie handling and patience."""
    images, labels = toy_data(100)
    fit, val = np.arange(0, 60), np.arange(60, 100)
    monkeypatch.setattr(runner_module, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "split_ids", lambda profile: (fit, val, "split-digest"))
    scores = iter([0.30, 0.50, 0.50, 0.40, 0.45, 0.49, 0.20, 0.99])
    real_evaluate = runner_module.evaluate
    calls = {"n": 0}

    def scripted(model, loader, criterion, device):
        metrics, ids, targets, preds = real_evaluate(model, loader, criterion, device)
        calls["n"] += 1
        if calls["n"] <= 7:  # epochs; the final call is the post-training reload
            metrics["macro_f1"] = next(scores)
        else:
            metrics["macro_f1"] = 0.50
        return metrics, ids, targets, preds

    monkeypatch.setattr(runner_module, "evaluate", scripted)
    config = load_config({"name": "toy", "profile": "smoke", "max_epochs": 25, "patience": 5, "batch_size": 16})
    record = runner_module.train_run(config, images, labels, log=lambda *_: None)
    assert record["best_epoch"] == 2  # tie at epoch 3 keeps the earlier epoch
    assert record["stop_epoch"] == 7  # 5 non-improving epochs after epoch 2
    assert record["stopped_early"] and record["status"] == "completed"
    history = pd.DataFrame(record["history"])
    assert history.improved.tolist() == [True, True, False, False, False, False, False]
    saved = pd.read_csv(tmp_path / "results/phase2/smoke/runs" / config["run_id"] / "val_predictions.csv.gz")
    assert saved.sample_id.tolist() == val.tolist() and saved.y_true.tolist() == labels[val].tolist()
    # A record from another config refuses to be overwritten.
    other = dict(config, config_sha256="different", run_id=config["run_id"])
    with pytest.raises(ValueError, match="different config"):
        runner_module.train_run(other, images, labels, log=lambda *_: None)


def test_fusion_join_rejects_misaligned_rows():
    labels = np.arange(10)
    ids = np.array([4, 2, 7])
    a, b = keyed(ids, labels, "official_train"), keyed(ids[::-1], labels, "official_train")
    emb, hog = np.ones((3, 64), np.float32), np.zeros((3, 1296), np.float32)
    assert joined_features(emb, a, hog, a.copy()).shape == (3, 1360)
    with pytest.raises(ValueError):
        joined_features(emb, a, hog, b)


def test_final_test_requires_explicit_flag_and_never_runs_on_smoke():
    with pytest.raises(ValueError, match="allow-test"):
        final_test("full", allow_test=False)
    with pytest.raises(ValueError, match="Smoke"):
        final_test("smoke", allow_test=True)
