"""Learning curves from saved run histories (no retraining)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..utils import ROOT  # noqa: E402
from .select import completed_runs  # noqa: E402


def learning_curves(profile="full"):
    runs = sorted((r for r in completed_runs(profile) if r["config"]["seed"] == 42),
                  key=lambda r: r["config"]["run_order"])
    if not runs:
        raise ValueError("No completed runs to plot")
    out = ROOT / "figures" / "phase2" / profile
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(runs), figsize=(3.6 * len(runs), 6.2), sharex=True, squeeze=False)
    for col, r in enumerate(runs):
        epochs = [h["epoch"] for h in r["history"]]
        ax = axes[0, col]
        ax.plot(epochs, [h["fit_loss"] for h in r["history"]], label="fit loss", color="#1f6f8b")
        ax.plot(epochs, [h["val_loss"] for h in r["history"]], label="validation loss", color="#d1495b")
        ax.axvline(r["best_epoch"], color="grey", linestyle=":", linewidth=1)
        ax.set_title(f"{r['name']} ({r['parameters']:,} params)", fontsize=10)
        ax.set_ylabel("cross-entropy (example-weighted)" if col == 0 else "")
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1, col]
        ax.plot(epochs, [h["val_macro_f1"] for h in r["history"]], marker="o", markersize=3, color="#2a9d8f")
        ax.axvline(r["best_epoch"], color="grey", linestyle=":", linewidth=1)
        ax.annotate(f"best epoch {r['best_epoch']}\nF1 {r['val_macro_f1']:.4f}\nstopped {r['stop_epoch']}",
                    xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation macro F1" if col == 0 else "")
    fig.suptitle(f"Phase 2 CNN learning curves ({profile} profile, seed 42); dotted line = saved checkpoint", fontsize=11)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"learning_curves.{suffix}", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for r in runs:
        ax.plot([h["epoch"] for h in r["history"]], [h["val_macro_f1"] for h in r["history"]], marker="o", markersize=3, label=r["name"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro F1")
    ax.set_title("Validation macro F1 by epoch (same fit/validation IDs)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"val_macro_f1_by_epoch.{suffix}", dpi=150)
    plt.close(fig)
    print(f"Wrote learning curves to {out.relative_to(ROOT)}")
