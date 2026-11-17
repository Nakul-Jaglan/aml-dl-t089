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

The network accepts normalized grayscale images of shape $[B, 1, 28, 28]$ (pixel values in $[0, 1]$). It is parameterized by `depth \in {2, 3}` and `dropout \in [0.0, 1.0]`:

* **Block 1 (both models):**
  - $\text{Conv2D}(1 \rightarrow 32, \text{kernel}=3\times 3, \text{stride}=1, \text{padding}=1)$
  - $\text{ReLU}$
  - $\text{MaxPool2D}(\text{kernel}=2\times 2, \text{stride}=2) \implies [B, 32, 14, 14]$
* **Block 2 (both models):**
  - $\text{Conv2D}(32 \rightarrow 64, \text{kernel}=3\times 3, \text{stride}=1, \text{padding}=1)$
  - $\text{ReLU}$
  - $\text{MaxPool2D}(\text{kernel}=2\times 2, \text{stride}=2) \implies [B, 64, 7, 7]$
* **Block 3 (CNN-3 only):**
  - $\text{Conv2D}(64 \rightarrow 64, \text{kernel}=3\times 3, \text{stride}=1, \text{padding}=1)$
  - $\text{ReLU}$
  - $\text{MaxPool2D}(\text{kernel}=2\times 2, \text{stride}=2) \implies [B, 64, 3, 3]$ (since $\lfloor 7/2 \rfloor = 3$)
* **Dense Embedding Projection:**
  - Flatten spatial feature maps
  - $\text{Linear}(64 \times \text{side}^2 \rightarrow 64)$ followed by $\text{ReLU} \implies [B, 64]$
  - For CNN-2: $\text{side}=7 \implies 64 \times 7 \times 7 = 3,136$ inputs $\rightarrow 64$
  - For CNN-3: $\text{side}=3 \implies 64 \times 3 \times 3 = 576$ inputs $\rightarrow 64$
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
| **Block 3** | N/A | $B \times 64 \times 3 \times 3$ | **36,928** (CNN-3 only) | $64 \times (64 \times 3 \times 3 + 1) = 36,928$ |
| **Dense Projection** | $B \times 64$ | $B \times 64$ | **200,768** / **36,928** | CNN-2: $64 \times (3,136 + 1) = 200,768$<br>CNN-3: $64 \times (576 + 1) = 36,928$ |
| **Head (Classifier)**| $B \times 10$ | $B \times 10$ | **650** | $10 \times (64 + 1) = 650$ |
| **Total** | **CNN-2: 220,234** | **CNN-3: 93,322** | — | Verified by test suite |

### Why CNN-3 Has Fewer Parameters Than CNN-2
Although CNN-3 adds a third convolutional layer (+36,928 parameters), the third max-pooling operation downsamples the feature map from $7 \times 7$ to $3 \times 3$. This shrinks the input to the dense layer from 3,136 down to 576, reducing the dense projection parameters from 200,768 to 36,928 (a net saving of 163,840 parameters). The dense saving far outweighs the convolution addition, resulting in **57.6% fewer total trainable weights** for CNN-3.

---

## 4. Contract and Shared Interfaces

### Training Contract (for Samarth)
* Call `logits = model(x)` to get unnormalized logit predictions `[B, 10]`.
* Feed raw logits directly to `nn.CrossEntropyLoss(reduction='mean')`.
* Optimization: AdamW ($\alpha=0.001$, weight decay=0.0001).

### Hybrid Feature Interface (for Samarth & Hybrid Evaluation)
* Call `embeddings = model.encode(x)` in `eval()` mode under `torch.inference_mode()`.
* Returns pre-dropout, non-negative, ReLU-activated 64-dimensional feature representations `[B, 64]`.
* This 64-dim learned vector is concatenated with the 1,296-dim HOG vector for the hybrid LR experiment:
  $$\mathbf{x}_{\text{hybrid}} = [\mathbf{z}_{\text{CNN}} \,\|\, \mathbf{h}_{\text{HOG}}] \in \mathbb{R}^{1360}$$

### Checkpoint Persistence Contract
* Checkpoints save atomic dictionaries containing `depth`, `dropout`, `state_dict`, `trainable_parameters`, and training metadata.
* `load_checkpoint` uses `map_location="cpu"` and restores weights cleanly across compute devices.

---

## 5. Report Contribution (Draft Paragraph for Nakul)

> *Convolutional Neural Network Architecture.* For the Phase 2 deep learning evaluation, we designed a compact convolutional neural network (`SmallCNN`) operating on $28 \times 28$ single-channel inputs. The primary reference architecture (CNN-2) consists of two convolutional stages ($3 \times 3$ kernels, padding 1, 32 and 64 channels respectively) followed by $2 \times 2$ max pooling with stride 2, shrinking spatial dimensions to $7 \times 7$. A fully connected projection layer maps the 3,136 flattened activations into a 64-dimensional latent embedding followed by a ReLU activation, before passing through a dropout layer ($p=0.30$) and a 10-way linear classification head. CNN-2 comprises exactly 220,234 trainable parameters. To assess the impact of architectural depth and receptive field expansion, a 3-block variant (CNN-3) incorporates an additional 64-channel convolutional block and max-pooling stage ($\lfloor 7/2 \rfloor = 3$), compressing the dense input to 576. Notably, this spatial downsampling reduces the projection layer from 200,768 to 36,928 parameters, yielding a total network capacity of 93,322 parameters (57.6% smaller than CNN-2). Both architectures expose an `encode` method that isolates the 64-dimensional learned representation, enabling controlled feature fusion with hand-crafted descriptors.
