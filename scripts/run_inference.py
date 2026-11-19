"""Run trained CNN-3 model inference on Fashion-MNIST images.

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --num 10
    python scripts/run_inference.py --id 40
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.phase2.models import count_parameters, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained CNN-3")
    parser.add_argument("--num", type=int, default=5, help="Number of random test images to classify")
    parser.add_argument("--id", type=int, default=None, help="Specific test image ID (0-9999)")
    args = parser.parse_args()

    ckpt_path = Path("artifacts/phase2/full/full_cnn3_s42_80a2edee.pt")
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print("=" * 75)
    print("CSAI PROJECT 11: FASHION-MNIST CNN-3 INFERENCE DEMO")
    print("=" * 75)
    print(f"Loading trained model checkpoint: {ckpt_path.name}")
    model, bundle = load_checkpoint(ckpt_path, map_location="cpu")
    model.eval()

    print(f"Model Architecture  : {bundle.get('arch', {}).get('name', 'SmallCNN')}")
    print(f"Number of Blocks    : {model.depth}")
    print(f"Trainable Parameters: {count_parameters(model):,}")
    print(f"Checkpoint Epoch    : {bundle.get('epoch')}")
    print(f"Validation Macro F1 : {bundle.get('val_macro_f1', 0.0):.4f}")
    print("-" * 75)

    # Load official test data
    test_images, test_labels = load_official("test")

    if args.id is not None:
        selected_ids = [args.id]
    else:
        # Pick 5 diverse test images
        np.random.seed(42)
        selected_ids = np.random.choice(len(test_images), size=args.num, replace=False).tolist()

    print(f"Running inference on {len(selected_ids)} test image(s)...\n")

    for idx, sample_id in enumerate(selected_ids, 1):
        raw_img = test_images[sample_id]
        true_label = test_labels[sample_id]
        true_name = CLASS_NAMES[true_label]

        # Convert to tensor [1, 1, 28, 28] normalized to [0, 1]
        x = torch.from_numpy(raw_img).float().unsqueeze(0).unsqueeze(0) / 255.0

        with torch.inference_mode():
            # 1. Feature embedding pass
            embeddings = model.encode(x)
            # 2. Classifier pass
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            pred_label = int(probabilities.argmax().item())
            confidence = float(probabilities[pred_label].item()) * 100

        pred_name = CLASS_NAMES[pred_label]
        is_correct = "✓ CORRECT" if pred_label == true_label else "✗ INCORRECT"

        print(f"[{idx}] Test ID #{sample_id:<5} | True: {true_name:<11} | Pred: {pred_name:<11} | Conf: {confidence:5.1f}% | {is_correct}")

        # Top 3 predicted classes
        top3_indices = torch.topk(probabilities, 3).indices.tolist()
        top3_probs = [float(probabilities[i]) * 100 for i in top3_indices]
        top3_str = ", ".join(f"{CLASS_NAMES[i]} ({p:.1f}%)" for i, p in zip(top3_indices, top3_probs))
        print(f"     Top 3: {top3_str}")
        print(f"     64-dim Embedding Norm: {embeddings.norm().item():.3f} (Pre-dropout ReLU features)")
        print()

    print("=" * 75)
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
