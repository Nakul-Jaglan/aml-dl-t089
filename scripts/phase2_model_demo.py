"""Demonstration and viva rehearsal script for Puneet's Phase 2 deliverables.

Demonstrates:
1. Exact layer-by-layer tensor dimension tracing.
2. Trainable parameter count reproduction (by hand arithmetic and by code).
3. Forward and backward pass on real Fashion-MNIST clothing images.
4. 64-dimensional feature embedding extraction for the hybrid pipeline.
5. Viva explanation and Hinglish talking points.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.phase2.models import SmallCNN, count_parameters, get_parameter_breakdown


def demonstrate_models():
    print("=" * 80)
    print("CSAI PROJECT 11 / PHASE 2 — PUNEET'S CNN ARCHITECTURE DEMONSTRATION")
    print("=" * 80)

    # 1. Parameter arithmetic verification
    print("\n[1] REPRODUCING TRAINABLE PARAMETER COUNTS (BY HAND & CODE)\n")
    for depth in (2, 3):
        model = SmallCNN(depth=depth, dropout=0.30)
        breakdown = get_parameter_breakdown(model)
        model_name = f"CNN-{depth}"
        print(f"--- {model_name} (depth={depth}) ---")
        for layer_name, info in breakdown["layers"].items():
            print(f"  • {layer_name:<20}: {info['formula']}  => {info['params']:,} params")
        total = breakdown["total_trainable_parameters"]
        print(f"  => TOTAL TRAINABLE PARAMETERS: {total:,}")
        assert breakdown["verified"]
        print(f"  => Match verified: {total == count_parameters(model)}\n")

    # 2. Loading a real batch from official training data
    print("-" * 80)
    print("[2] EXECUTING REAL BATCH THROUGH CNN-2 & CNN-3")
    print("-" * 80)
    raw_images, raw_labels = load_official("train")
    with np.load("data/smoke_indices.npz") as split:
        smoke_fit_ids = split["train_idx"][:4]

    # Convert to normalized float32 tensor [B, 1, 28, 28]
    batch_raw = raw_images[smoke_fit_ids]
    batch_labels = raw_labels[smoke_fit_ids]
    x = torch.from_numpy(batch_raw).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(batch_labels).long()

    print(f"Input batch shape : {list(x.shape)} (float32, min={x.min():.2f}, max={x.max():.2f})")
    print(f"Sample IDs        : {smoke_fit_ids.tolist()}")
    print(f"True Labels       : {[CLASS_NAMES[i] for i in y.tolist()]}")

    criterion = nn.CrossEntropyLoss()

    for depth in (2, 3):
        model = SmallCNN(depth=depth, dropout=0.30)
        model.train()
        model.zero_grad()

        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        grad_norm = model.blocks[0].weight.grad.norm().item()

        # Encode pass (features for hybrid model)
        model.eval()
        with torch.inference_mode():
            features = model.encode(x)

        print(f"\nCNN-{depth}:")
        print(f"  • Logits shape       : {list(logits.shape)} (Finite loss: {loss.item():.4f})")
        print(f"  • Conv1 gradient norm: {grad_norm:.6f} (Backward pass verified)")
        print(f"  • 64-dim embeddings  : {list(features.shape)} (min={features.min():.4f}, max={features.max():.4f})")

    # 3. Step-by-step trace of a Shirt image (class 6)
    print("\n" + "=" * 80)
    print("[3] STEP-BY-STEP DIMENSION TRACE FOR A SINGLE IMAGE")
    print("=" * 80)
    single_img = x[0:1]  # [1, 1, 28, 28]
    print(f"Input image        : {list(single_img.shape)}")
    m = SmallCNN(depth=2)
    m.eval()
    with torch.inference_mode():
        # Block 1
        b1_out = m.features[2](m.features[1](m.features[0](single_img)))
        print(f"After Block 1 (C1,ReLU,Pool): {list(b1_out.shape)}  (28x28 -> 14x14)")
        # Block 2
        b2_out = m.features[5](m.features[4](m.features[3](b1_out)))
        print(f"After Block 2 (C2,ReLU,Pool): {list(b2_out.shape)}   (14x14 -> 7x7)")
        # Flatten & Projection
        flat = torch.flatten(b2_out, 1)
        print(f"After Spatial Flatten       : {list(flat.shape)}   (64 * 7 * 7 = 3,136)")
        proj = torch.relu(m.dense[1](flat))
        print(f"After Linear Projection+ReLU: {list(proj.shape)}      (64 learned features)")
        logits = m.classifier(proj)
        print(f"After Classification Head   : {list(logits.shape)}      (10 class logits)")

    print("\n" + "=" * 80)
    print("[4] VIVA VOCE CHEAT SHEET & TALKING POINTS")
    print("=" * 80)
    print("""
1. Parameter Count Disparity:
   Q: Why does deeper CNN-3 have 167k params while CNN-2 has 220k params?
   A: The 3rd pooling layer floors 7/2 to 3, shrinking dense layer inputs from 3,136 to 1,152.
      Dense weights drop from 200,768 to 73,792 (saving 126,976), which outweighs the 73,856
      weights of the added Conv3 layer.

2. Pooling vs. Shirt Recall:
   Q: Why can pooling hurt Shirt recall?
   A: Max pooling discards spatial fine structure (collars, sleeves, buttons) that distinguish
      Shirts from T-shirts and Pullovers at 28x28 resolution.

3. Loss API:
   Q: Why pass raw logits to CrossEntropyLoss?
   A: nn.CrossEntropyLoss applies log-softmax internally using the LogSumExp trick, preventing
      numerical underflow/overflow.

4. Hinglish Summary (for verbal presentation):
   "Phase 1 mein fixed HOG features aur SVM use kiya tha jisme Shirt recall 73.3% tha.
    Phase 2 mein humne 2-block aur 3-block CNN design kiye jo spatial features end-to-end learn
    karte hain. Hamara model embed(x)/encode(x) ke through pre-dropout 64-dim embeddings deta hai
    jo HOG ke sath combine karke hybrid ML+DL model banane ke kaam aata hai."
""")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_models()
