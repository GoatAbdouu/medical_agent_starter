"""
=============================================================
ALL-IN-ONE GRAPH GENERATOR FOR PRESENTATION
=============================================================
Run this AFTER training is done.
It reads training_history.json + the trained model + dataset
and generates ALL 5 graphs automatically.

HOW TO USE:
  python scripts/generate_graphs.py --data_dir "path/to/IMG_CLASSES"

It will create a 'graphs/' folder with:
  1. training_curves.png      (loss + accuracy over epochs)
  2. confusion_matrix.png     (which diseases get confused)
  3. metrics_comparison.png   (accuracy, precision, recall, F1)
  4. per_class_f1.png         (F1 per disease)
  5. dataset_distribution.png (shows the imbalance)
=============================================================
"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

from medical_agent.core.skin_disease_classifier import (
    FOLDER_TO_DISEASE,
    INFERENCE_TRANSFORMS,
    SkinDiseaseModel,
)


CLASS_NAMES_FR = [
    "Eczéma",
    "Verrues/Molluscum",
    "Mélanome",
    "Dermatite Atopique",
    "Carcinome Basocell.",
    "Nævus Mélanocyt.",
    "Kératose Bénigne",
    "Psoriasis",
    "Kératose Séborrh.",
    "Mycose/Teigne",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Génération des graphiques pour la présentation")
    parser.add_argument("--data_dir", required=True, help="Chemin vers le dossier IMG_CLASSES")
    parser.add_argument("--model_path", default="models/skin_disease_model.pth", help="Chemin du modèle entraîné")
    parser.add_argument("--history_path", default="models/training_history.json", help="Chemin de l'historique JSON")
    parser.add_argument("--output_dir", default="graphs", help="Dossier de sortie pour les graphiques")
    return parser.parse_args()

def plot_training_curves(history, output_dir):
    """Graph 1: Training & Validation Loss + Accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Training Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Validation Loss", markersize=4)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Training Accuracy", markersize=4)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", label="Validation Accuracy", markersize=4)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {path}")

def plot_confusion_matrix(all_labels, all_preds, output_dir):
    """Graph 2: Confusion Matrix."""
    try:
        import seaborn as sns
    except ImportError:
        print("   ⚠️  pip install seaborn for confusion matrix")
        return

    cm = confusion_matrix(all_labels, all_preds)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_percent, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=CLASS_NAMES_FR, yticklabels=CLASS_NAMES_FR,
        ax=ax, vmin=0, vmax=100, linewidths=0.5, linecolor="white",
    )
    ax.set_title("Matrice de Confusion (%)", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Classe Prédite", fontsize=13)
    ax.set_ylabel("Classe Réelle", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {path}")

def plot_metrics_comparison(report_dict, output_dir):
    """Graph 3: Accuracy / Precision / Recall / F1 bar chart."""
    wa = report_dict["weighted avg"]
    metrics = {
        "Accuracy": report_dict["accuracy"],
        "Precision": wa["precision"],
        "Recall": wa["recall"],
        "F1-Score": wa["f1-score"],
    }

    labels = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_title("Performance Metrics — MobileNetV2 (Skin Disease Classification)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = output_dir / "metrics_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {path}")

def plot_per_class_f1(report_dict, output_dir):
    """Graph 4: Per-class F1-score horizontal bar chart."""
    f1_scores = []
    for name in CLASS_NAMES_FR:
        if name in report_dict:
            f1_scores.append(report_dict[name]["f1-score"])
        else:
            f1_scores.append(0.0)

    colors = plt.cm.RdYlGn(f1_scores)

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(CLASS_NAMES_FR, [s * 100 for s in f1_scores], color=colors,
                   edgecolor="white", height=0.6)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score * 100:.1f}%", va="center", fontsize=11, fontweight="bold")

    ax.set_title("F1-Score par Classe de Maladie", fontsize=14, fontweight="bold")
    ax.set_xlabel("F1-Score (%)", fontsize=12)
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    path = output_dir / "per_class_f1.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {path}")

def plot_dataset_distribution(history, output_dir):
    """Graph 5: Dataset class distribution (shows imbalance)."""
    classes = history.get("class_names", CLASS_NAMES_FR)
    counts = history.get("class_counts", [1677, 2103, 15750, 1250, 3323, 7970, 2624, 2000, 1800, 1700])
    total = sum(counts)

    colors = []
    for c in counts:
        if c > 10000:
            colors.append("#F44336")
        elif c > 5000:
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(classes, counts, color=colors, edgecolor="white", height=0.6)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Distribution du Dataset — Déséquilibre des Classes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Nombre d'images", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    path = output_dir / "dataset_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ {path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load history ---
    history_path = Path(args.history_path)
    if not history_path.exists():
        print(f"❌ Historique introuvable : {history_path}")
        sys.exit(1)
    with open(history_path, "r") as f:
        history = json.load(f)
    print(f"📂 Historique chargé : {history_path}")

    # --- Graph 5 first (no model needed) ---
    print("\n📊 Génération des graphiques...\n")
    plot_dataset_distribution(history, output_dir)

    # --- Graph 1: Training curves ---
    plot_training_curves(history, output_dir)

    # --- Load model + dataset for graphs 2-4 ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"⚠️  Modèle introuvable : {model_path} — graphs 2-4 skipped")
        print(f"\n🎉 Graphiques sauvegardés dans : {output_dir}/")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinDiseaseModel(num_classes=10, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"   🧠 Modèle chargé sur {device}")

    val_transforms = INFERENCE_TRANSFORMS
    dataset = datasets.ImageFolder(root=args.data_dir, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"   📂 Dataset chargé : {len(dataset)} images")

    # --- Run predictions ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Graph 2: Confusion Matrix ---
    plot_confusion_matrix(all_labels, all_preds, output_dir)

    # --- Classification report ---
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES_FR,
        digits=4,
        output_dict=True,
    )

    # Print it too
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES_FR, digits=4))

    # --- Graph 3: Metrics comparison ---
    plot_metrics_comparison(report_dict, output_dir)

    # --- Graph 4: Per-class F1 ---
    plot_per_class_f1(report_dict, output_dir)

    print(f"\n🎉 Tous les graphiques sauvegardés dans : {output_dir}/")
    print("   Ouvrez-les et mettez-les dans votre présentation !")


if __name__ == "__main__":
    main()