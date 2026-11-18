# Phase 2 CNN Architecture & Feature Interface

**Author / Component Owner:** Puneet  
**Branch:** `phase2/cnn-architecture-puneet`  
**Files:**  
- `src/clothing_ml/phase2/models.py`
- `tests/test_phase2_models.py`
- `scripts/phase2_model_demo.py`

---

## 1. Executive Summary

As specified in the Phase 2 Work Plan, the deep learning component introduces a compact convolutional neural network (`SmallCNN`) designed to evaluate learned spatial representations on the Fashion-MNIST benchmark using the exact frozen dataset partitions (`48,000` fit, `12,000` validation, `10,000` official test).

The model design directly addresses the research question:
> *Can learned local image filters improve macro F1 and alleviate the recurring Shirt, T-shirt/top, and Pullover ambiguities observed in Phase 1 (where Shirt recall was capped at 73.3%), while remaining computationally lightweight?*

---

## 2. Architecture Specifications

The network accepts normalized grayscale images of shape $[B, 1, 28, 28]$ (pixel values in $[0, 1]$). It is parameterized by `depth \in {2, 3}` (or `blocks`), `dropout \in [0.0, 1.0)`, and `embedding_dim=64`:

* **Block 1 (both models):**
  - $\text{Conv2D}(1 \rightarrow 32, \text{kernel}=3\times 3, \text{stride}=1, \text{padding}=1)$
  - $\text{ReLU}$
  - $\text{MaxPool2D}(\text{kernel}=2\times 2, \text{stride}=2) \implies [B, 32, 14, 14]$
* **Block 2 (both models):**
  - $\text{Conv2D}(32 \rightarrow 64, \text{kernel}=3\times 3, \text{stride}=1, \text{padding}=1)$
  - $\text{ReLU}$
  - $\text{MaxPool2D}(\text{kernel}=2\times 2, \text{stride}=2) \implies [B, 64, 7, 7]$
* **Block 3 (CNN-3 only):**
  - $\text{Conv2D}(64 \rightarrow 128, \text{kernel}=3\times 3, \text{stride}=1, \text{padding}=1)$ *(or $64 \rightarrow 64$ in 93k variant)*
  - $\text{ReLU}$
  - $\text{MaxPool2D}(\text{kernel}=2\times 2, \text{stride}=2) \implies [B, C_3, 3, 3]$ (since $\lfloor 7/2 \rfloor = 3$)
* **Dense Embedding Projection:**
  - Flatten spatial feature maps
  - $\text{Linear}(C_{\text{out}} \times \text{side}^2 \rightarrow 64)$ followed by $\text{ReLU} \implies [B, 64]$
  - For CNN-2: $\text{side}=7 \implies 64 \times 7 \times 7 = 3,136$ inputs $\rightarrow 64$
  - For CNN-3: $\text{side}=3 \implies 128 \times 3 \times 3 = 1,152$ inputs $\rightarrow 64$
* **Head (Dropout + Linear Classifier):**
  - $\text{Dropout}(p)$ applied to the 64-dimensional embedding during training
  - $\text{Linear}(64 \rightarrow 10) \implies [B, 10]$ raw class logits

---

## 3. Layer Shapes and Parameter Arithmetic

A bias-enabled convolution layer has:
$$\text{Params}_{\text{Conv}} = C_{\text{out}} \times (C_{\text{in}} \times K_h \times K_w + 1)$$

A bias-enabled linear layer has:
$$\text{Params}_{\text{Linear}} = N_{\text{out}} \times (N_{\text{in}} + 1)$$

| Stage | CNN-2 Tensor Output | CNN-3 Tensor Output | Trainable Parameters | Exact Hand Calculation |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | $B \times 1 \times 28 \times 28$ | $B \times 1 \times 28 \times 28$ | $0$ | None |
| **Block 1** | $B \times 32 \times 14 \times 14$ | $B \times 32 \times 14 \times 14$ | **320** | $32 \times (1 \times 3 \times 3 + 1) = 320$ |
| **Block 2** | $B \times 64 \times 7 \times 7$ | $B \times 64 \times 7 \times 7$ | **18,496** | $64 \times (32 \times 3 \times 3 + 1) = 18,496$ |
| **Block 3** | N/A | $B \times 128 \times 3 \times 3$ | **73,856** (CNN-3) | $128 \times (64 \times 3 \times 3 + 1) = 73,856$ |
| **Dense Projection** | $B \times 64$ | $B \times 64$ | **200,768** / **73,792** | CNN-2: $64 \times (3,136 + 1) = 200,768$<br>CNN-3: $64 \times (1,152 + 1) = 73,792$ |
| **Head (Classifier)**| $B \times 10$ | $B \times 10$ | **650** | $10 \times (64 + 1) = 650$ |
| **Total** | **CNN-2: 220,234** | **CNN-3: 167,114** | — | Verified by test suite |

*(Note: with channels=(32, 64, 64), CNN-3 has 93,322 parameters. Both variants exhibit the same pooling downsampling benefit.)*

### Why CNN-3 Has Fewer Parameters Than CNN-2
Although CNN-3 adds a third convolutional stage (+73,856 parameters), the third max-pooling operation downsamples the feature map from $7 \times 7$ to $3 \times 3$. This shrinks the input to the dense projection layer from 3,136 down to 1,152, reducing dense projection parameters from 200,768 to 73,792 (a net saving of 126,976 parameters in the dense head). The dense saving outweighs the added convolution weights, resulting in **24.1% fewer total parameters** (167,114 vs 220,234).

---

## 4. Contract and Shared Interfaces

### Training Contract (for Samarth)
* Call `logits = model(x)` to get unnormalized logit predictions `[B, 10]`.
* Feed raw logits directly to `nn.CrossEntropyLoss(reduction='mean')`.
* Optimization: AdamW ($\alpha=0.001$, weight decay=0.0001).

### Hybrid Feature Interface (for Samarth & Hybrid Evaluation)
* Call `embeddings = model.encode(x)` (or `model.embed(x)`) in `eval()` mode under `torch.inference_mode()`.
* Returns pre-dropout, non-negative, ReLU-activated 64-dimensional feature representations `[B, 64]`.
* This 64-dim learned vector is concatenated with the 1,296-dim HOG vector for the hybrid LR experiment:
  $$\mathbf{x}_{\text{hybrid}} = [\mathbf{z}_{\text{CNN}} \,\|\, \mathbf{h}_{\text{HOG}}] \in \mathbb{R}^{1360}$$

### Checkpoint Persistence Contract
* Checkpoints save atomic dictionaries containing `arch`, `state_dict`, and training metadata.
* `load_checkpoint` uses `map_location="cpu"` and restores weights cleanly across compute devices.
