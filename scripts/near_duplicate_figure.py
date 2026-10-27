"""Figure for the near-duplicate audit; reads the saved JSON, retrains nothing."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clothing_ml.data import load_official
from clothing_ml.plots import save_figure, setup_style
from clothing_ml.utils import ROOT, read_json

SHOWN = 4

setup_style()
report = read_json(ROOT / "data/near_duplicate_audit.json")
images, _ = load_official("train")
test_images, _ = load_official("test")
sources = {"official_train": images, "official_test": test_images}

fig, axes = plt.subplots(3, 2 * SHOWN, figsize=(13, 6.2))
for column_block, comparison in enumerate(report["comparisons"]):
    for offset, pair in enumerate(comparison["closest_pairs"][:SHOWN]):
        column = column_block * SHOWN + offset
        query_split, query_id = pair["query"].split(":")
        neighbour_split, neighbour_id = pair["neighbour"].split(":")
        left = sources[query_split][int(query_id)]
        right = sources[neighbour_split][int(neighbour_id)]
        difference = np.abs(left.astype(np.int32) - right.astype(np.int32))

        axes[0, column].imshow(left, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(f"{pair['query']}\n{pair['query_class']}", fontsize=8)
        axes[1, column].imshow(right, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title(f"{pair['neighbour']}\n{pair['neighbour_class']}", fontsize=8)
        axes[2, column].imshow(difference, cmap="magma", vmin=0, vmax=max(1, int(difference.max())))
        axes[2, column].set_title(f"RMS {pair['rms_pixel_difference']}\n"
                                  f"{pair['differing_pixels']}/784 px, max {pair['max_absolute_difference']}",
                                  fontsize=8)
        for row in range(3):
            axes[row, column].axis("off")

for row, text in enumerate(["query image", "nearest neighbour", "absolute difference"]):
    axes[row, 0].text(-0.3, 0.5, text, transform=axes[row, 0].transAxes,
                      rotation=90, va="center", ha="center", fontsize=9)

fig.text(0.29, 0.965, "Validation versus fit pool", ha="center", fontsize=11, fontweight="bold")
fig.text(0.74, 0.965, "Official test versus official training", ha="center", fontsize=11, fontweight="bold")
fig.suptitle("Closest near-duplicate pairs found by exhaustive pixel-space search "
             "(no exact byte duplicates exist)", y=1.035, fontsize=11)
fig.tight_layout(rect=(0.015, 0, 1, 0.95), h_pad=2.0)
save_figure(fig, ROOT / "figures/eda", "near_duplicates")
print("Wrote figures/eda/near_duplicates.png and .pdf")
