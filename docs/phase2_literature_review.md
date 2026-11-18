# Phase 2 literature review

The project asks whether learned spatial features improve the fixed-feature
Fashion-MNIST baseline and how that change affects visually similar garments.
The studies below inform specific design choices. Numerical project conclusions
come from the saved run evidence in `results/phase2/`.

## Dataset and evaluation context

Xiao, Rasul and Vollgraf introduce Fashion-MNIST as a ten-category, 28 by 28
image benchmark with 60,000 training and 10,000 test items. The project reuses
that official membership and a separately saved 48,000/12,000 split inside the
training partition. Its centered grayscale inputs support a controlled comparison.
They give limited evidence about photographs with varied backgrounds or lighting.

Primary source: https://arxiv.org/abs/1708.07747

## Learned local features

LeCun, Bottou, Bengio and Haffner describe convolutional networks for document
recognition. Local connections and shared filter weights provide a rationale
for learning spatial patterns, while subsampling reduces map dimensions. We
reuse those principles in a small clothing classifier. The evaluated CNN-3 uses
32, 64 and 128 channels, then a 64-value dense embedding. Its clothing scores
come from this experiment. The document-recognition paper does not establish
those scores or an improvement in Shirt recall.

Primary author record and linked paper: https://bottou.org/papers/lecun-98h

## Regularization and optimization

Srivastava and colleagues describe dropout as random suppression of units
while training. The project places dropout after the dense ReLU embedding and
compares p=0.30 with p=0 under a shared selection rule. Removing dropout reduced
validation macro F1 by 0.0036 in the recorded seed-42 run. The ablation supports
a local observation about this setup, with other seeds unmeasured.

Primary source: https://www.jmlr.org/papers/v15/srivastava14a.html

Loshchilov and Hutter distinguish decoupled weight decay from adding an L2
penalty to an adaptive optimizer's gradient. The CNN runs use AdamW with learning
rate 0.001 and weight decay 0.0001. These are fixed development choices in the
run matrix. An optimizer comparison was outside that matrix.

Primary source: https://arxiv.org/abs/1711.05101

## Fixed gradients and the hybrid hypothesis

Dalal and Triggs introduce locally normalized gradient histograms in human
detection. The Phase 1 descriptor adapts their HOG idea to clothing images.
The secondary hypothesis is that HOG may retain edge information missing from
a 64-value CNN embedding. Fusion concatenates the two and fits LR, with
embedding-only and HOG-only LR controls under the same C grid. Fusion reaches
0.9261 validation F1, below embedding-only LR at 0.9319. The implementation is
complete and the control comparison gives no aggregate support for adding HOG.

Primary source: https://doi.org/10.1109/CVPR.2005.177

## Software and evidence

Scikit-learn provides the LR pipeline and fit-only standardization. PyTorch
provides the CNN layers, cross-entropy and AdamW implementation. Checkpoint
selection uses validation macro F1, so the loss-minimizing epoch can differ.
The report records the actual package versions and shows the saved learning
curves. The primary-paper metadata and official API references were checked
on 19 November 2026.

- https://jmlr.org/papers/v12/pedregosa11a.html
- https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html
- https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html
- https://scikit-learn.org/stable/common_pitfalls.html

## What the combined evidence supports

The CNN improves aggregate test metrics and recorded CPU batch cost against
Phase 1 HOG + SVM. Shirt recall falls despite higher precision. Translation
and fusion have mixed or negative outcomes. These findings argue for measured
controls rather than an assumption that a learned or combined representation
must improve every category. Paired bootstrap intervals describe test-image
sampling variability, with training-seed sensitivity and deployment transfer
remaining outside the measured scope.
