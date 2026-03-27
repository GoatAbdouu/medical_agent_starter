"""
Training History Visualisation Script.

Loads the JSON training history produced by ``train_ensemble.py`` (or
``train_skin_classifier.py``) and generates:

  1. Per-model loss curves (train vs val)
  2. Per-model accuracy curves (train vs val)
  3. Combined comparison plot (all models on the same axes)
  4. Confusion matrix (optional — requires a predictions JSON)

Usage
-----
# Plot ensemble history
python scripts/plot_training_history.py \\
    --history models/ensemble_training_history.json \\
    --output_dir outputs

# Also importable as a module (e.g. from a Colab notebook):
    from scripts.plot_training_history import plot_all
    plot_all("models/ensemble_training_history.json", output_dir="outputs")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional imports (fail gracefully so the module can be imported without a
# display server)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend – safe for servers & Colab
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

try:
    import seaborn as sns  # type: ignore

    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False

try:
    import numpy as np

    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_COLORS = {
    "EfficientNet-B3": "#2196F3",   # blue
    "MobileNetV2":     "#4CAF50",   # green
    "ResNet50":        "#FF5722",   # deep orange
    # Single-model history key
    "MobileNetV2 (single)": "#9C27B0",
}


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_history(history_path: Path) -> Dict:
    """Load and normalise a training history JSON file.

    Supports both:
    - Ensemble format: ``{ "EfficientNet-B3": {...}, "MobileNetV2": {...}, ... }``
    - Single-model format: ``{ "train_loss": [...], "val_loss": [...], ... }``
    """
    with open(history_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Detect format
    first_value = next(iter(raw.values()))
    if isinstance(first_value, dict):
        # Ensemble format — already correct
        return raw

    # Single-model format — wrap it
    name = history_path.stem.replace("_", " ").title()
    return {name: raw}


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_model_curves(
    model_name: str,
    history: Dict,
    output_dir: Path,
) -> None:
    """Save loss and accuracy curves for a single model."""
    if not _MPL_AVAILABLE:
        print("⚠️  matplotlib not installed — skipping plots.")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight="bold")

    color = _MODEL_COLORS.get(model_name, "#607D8B")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", color=color, linewidth=2)
    ax.plot(
        epochs, history["val_loss"],
        label="Val loss", color=color, linewidth=2, linestyle="--", alpha=0.7
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    ax.plot(epochs, train_acc_pct, label="Train acc", color=color, linewidth=2)
    ax.plot(
        epochs, val_acc_pct,
        label="Val acc", color=color, linewidth=2, linestyle="--", alpha=0.7
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
    out_path = output_dir / f"training_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def plot_combined_comparison(
    all_histories: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Save a combined plot comparing all models on the same axes."""
    if not _MPL_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Ensemble Models — Comparison", fontsize=14, fontweight="bold")

    for model_name, history in all_histories.items():
        color = _MODEL_COLORS.get(model_name, "#607D8B")
        epochs = list(range(1, len(history["train_loss"]) + 1))
        val_acc_pct = [a * 100 for a in history["val_acc"]]
        val_loss    = history["val_loss"]

        axes[0].plot(epochs, val_loss,    label=model_name, color=color, linewidth=2)
        axes[1].plot(epochs, val_acc_pct, label=model_name, color=color, linewidth=2)

    for ax, title, ylabel in zip(
        axes,
        ["Validation Loss", "Validation Accuracy (%)"],
        ["Loss", "Accuracy (%)"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "ensemble_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def plot_confusion_matrix(
    predictions_path: Path,
    output_dir: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot a confusion matrix from a predictions JSON file.

    The JSON should contain ``{"y_true": [...], "y_pred": [...]}``.
    """
    if not _MPL_AVAILABLE or not _NP_AVAILABLE:
        print("⚠️  matplotlib / numpy not installed — skipping confusion matrix.")
        return

    if not predictions_path.exists():
        print(f"⚠️  Predictions file not found: {predictions_path} — skipping confusion matrix.")
        return

    with open(predictions_path, encoding="utf-8") as f:
        preds = json.load(f)

    y_true: List[int] = preds["y_true"]
    y_pred: List[int] = preds["y_pred"]
    n_classes = max(max(y_true), max(y_pred)) + 1

    # Build confusion matrix manually
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(12, 10))

    if _SNS_AVAILABLE:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or range(n_classes),
            yticklabels=class_names or range(n_classes),
            ax=ax,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        if class_names:
            tick_marks = np.arange(n_classes)
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_all(
    history_path: str | Path,
    output_dir: str | Path = "outputs",
    predictions_path: Optional[str | Path] = None,
) -> None:
    """
    Generate all plots from a training history file.

    Parameters
    ----------
    history_path : str or Path
        Path to the training history JSON file.
    output_dir : str or Path
        Directory where plots are saved (created if it does not exist).
    predictions_path : str or Path, optional
        Path to a ``{"y_true": [...], "y_pred": [...]}`` JSON for the
        confusion matrix.  If ``None``, the confusion matrix is skipped.
    """
    if not _MPL_AVAILABLE:
        print("❌ matplotlib is required for plotting. Install it with: pip install matplotlib")
        return

    history_path = Path(history_path)
    output_dir = Path(output_dir)

    if not history_path.exists():
        print(f"❌ History file not found: {history_path}")
        return

    _ensure_output_dir(output_dir)

    print(f"\n📊 Loading training history: {history_path}")
    all_histories = _load_history(history_path)

    print(f"   Found {len(all_histories)} model(s): {list(all_histories.keys())}")

    # Per-model plots
    for model_name, history in all_histories.items():
        print(f"\n  Plotting {model_name}...")
        plot_model_curves(model_name, history, output_dir)

    # Combined comparison (only meaningful with >1 model)
    if len(all_histories) > 1:
        print("\n  Plotting combined comparison...")
        plot_combined_comparison(all_histories, output_dir)

    # Confusion matrix (optional)
    if predictions_path is not None:
        print("\n  Plotting confusion matrix...")
        plot_confusion_matrix(Path(predictions_path), output_dir)

    print(f"\n✅ All plots saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot training history for skin disease classifier models"
    )
    parser.add_argument(
        "--history",
        default="models/ensemble_training_history.json",
        help="Path to the training history JSON file (default: models/ensemble_training_history.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save plots (default: outputs)",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional path to a {y_true, y_pred} JSON for confusion matrix",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    plot_all(
        history_path=_args.history,
        output_dir=_args.output_dir,
        predictions_path=_args.predictions,
    )
