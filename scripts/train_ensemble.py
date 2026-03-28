"""
Ensemble Training Script — trains EfficientNet-B3, MobileNetV2, and ResNet50
sequentially on the skin disease dataset.

Usage
-----
python scripts/train_ensemble.py \\
    --data_dir  path/to/IMG_CLASSES \\
    --output_dir models \\
    --epochs     40 \\
    --batch_size 32 \\
    --lr         1e-3

Each model is trained in two phases:
  Phase 1 — classifier head only (base frozen)        : PHASE1_EPOCHS epochs
  Phase 2 — full network fine-tune with differential LR: remaining epochs

The best checkpoint (highest val accuracy) is saved for each model.
Training history (loss + accuracy per epoch) is saved to:
    <output_dir>/ensemble_training_history.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets

from medical_agent.core.ensemble_classifier import (
    EfficientNetModel,
    MobileNetModel,
    ResNetModel,
)
from medical_agent.core.image_pipeline import ImagePipeline
from medical_agent.core.skin_disease_classifier import FOLDER_TO_DISEASE

PHASE1_EPOCHS = 10
EARLY_STOP_PATIENCE = 8

# Model registry: name → (class, output filename)
MODEL_REGISTRY: Dict[str, Tuple] = {
    "EfficientNet-B3": (EfficientNetModel, "efficientnet_skin.pth"),
    "MobileNetV2":     (MobileNetModel,    "mobilenet_skin.pth"),
    "ResNet50":        (ResNetModel,       "resnet_skin.pth"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ensemble skin disease classifiers (EfficientNet-B3, MobileNetV2, ResNet50)"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the root dataset folder (contains the 10 class sub-folders).",
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Directory where trained models and history are saved (default: models).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Total training epochs per model (Phase1 + Phase2, default: 40).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Phase 1 (default: 1e-3).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> Tuple[float, float]:
    """Run one training or validation epoch. Returns (loss, accuracy)."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train and optimizer is not None:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train and optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train_model(
    model_name: str,
    model_cls,
    output_path: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> Dict:
    """
    Full two-phase training loop for a single backbone model.

    Returns a history dict with train_loss, val_loss, train_acc, val_acc lists.
    """
    print(f"\n{'=' * 70}")
    print(f"  Training: {model_name}")
    print(f"{'=' * 70}")

    model: nn.Module = model_cls(num_classes=num_classes, freeze_base=True)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters — total: {total:,}  | trainable (head): {trainable:,}")

    history = {
        "model": model_name,
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    phase2_epochs = max(0, args.epochs - PHASE1_EPOCHS)

    # ------------------------------------------------------------------
    # Phase 1: classifier head only
    # ------------------------------------------------------------------
    print(f"\n  Phase 1 — Head only ({PHASE1_EPOCHS} epochs)")
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer1, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        icon = "✅" if val_acc > best_val_acc else "  "
        print(
            f"  [P1] Epoch {epoch:3d}/{PHASE1_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.2%} {icon}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"       💾 Checkpoint saved → {output_path}  (val acc {best_val_acc:.2%})")

    # ------------------------------------------------------------------
    # Phase 2: full fine-tuning with differential LRs + early stopping
    # ------------------------------------------------------------------
    if phase2_epochs > 0:
        print(f"\n  Phase 2 — Full fine-tune ({phase2_epochs} epochs, early stop patience={EARLY_STOP_PATIENCE})")
        model.unfreeze_backbone()

        trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters (Phase 2): {trainable2:,}")

        # Build parameter groups with differential learning rates
        # Backbone (features) gets a 10× smaller LR than the head
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            # The custom classifier head is always stored in .base.classifier or .base.fc
            if "classifier" in name or name.startswith("base.fc"):
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer2 = torch.optim.Adam([
            {"params": backbone_params, "lr": args.lr * 0.01},
            {"params": head_params,     "lr": args.lr * 0.1},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, mode="max", patience=5, factor=0.5
        )

        epochs_no_improve = 0
        for epoch in range(1, phase2_epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer2, device, is_train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None, device, is_train=False
            )
            scheduler.step(val_acc)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_acc"].append(round(val_acc, 4))

            icon = "✅" if val_acc > best_val_acc else "  "
            print(
                f"  [P2] Epoch {epoch:3d}/{phase2_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
                f"val loss {val_loss:.4f} acc {val_acc:.2%} {icon}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), output_path)
                print(f"       💾 Checkpoint saved → {output_path}  (val acc {best_val_acc:.2%})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(
                        f"\n  ⏹️ Early stopping after {epoch} Phase-2 epochs "
                        f"(patience={EARLY_STOP_PATIENCE})"
                    )
                    break

    print(f"\n  ✅ {model_name} — best val acc: {best_val_acc:.2%}")
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ensemble training script."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    pipeline = ImagePipeline()
    train_tf = pipeline.get_train_transforms()
    val_tf = pipeline.get_val_transforms()

    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_tf)
    num_classes = len(full_dataset.classes)

    # Class distribution
    class_counts: List[int] = [0] * num_classes
    for _, label in full_dataset.samples:
        class_counts[label] += 1

    print(f"\n📂 Dataset: {len(full_dataset)} images across {num_classes} classes")
    for idx, cls in enumerate(full_dataset.classes):
        readable = FOLDER_TO_DISEASE.get(cls, cls)
        print(f"   [{idx}] {readable:40s} — {class_counts[idx]} images")

    # 80/20 train/val split (reproducible)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Validation set uses val transforms (no augmentation)
    val_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_tf)
    val_final = Subset(val_dataset, val_subset.indices)

    # WeightedRandomSampler for class imbalance in training
    train_class_counts: List[int] = [0] * num_classes
    for i in train_subset.indices:
        train_class_counts[full_dataset.samples[i][1]] += 1

    sample_weights = [
        1.0 / train_class_counts[full_dataset.samples[i][1]]
        for i in train_subset.indices
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_final,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"\n📊 Split — train: {n_train}  val: {n_val}")

    # ------------------------------------------------------------------
    # Weighted cross-entropy loss (corrects for class imbalance)
    # ------------------------------------------------------------------
    class_weights = torch.tensor(
        [n_train / (num_classes * max(c, 1)) for c in train_class_counts],
        dtype=torch.float,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------------------------------------------------
    # Train each model
    # ------------------------------------------------------------------
    all_histories: Dict[str, Dict] = {}

    for model_name, (model_cls, filename) in MODEL_REGISTRY.items():
        output_path = output_dir / filename
        history = train_model(
            model_name=model_name,
            model_cls=model_cls,
            output_path=output_path,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=device,
            criterion=criterion,
            args=args,
        )
        all_histories[model_name] = history

    # ------------------------------------------------------------------
    # Save consolidated training history
    # ------------------------------------------------------------------
    history_path = output_dir / "ensemble_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("🎉 Ensemble training complete!")
    print(f"   Models saved in : {output_dir}")
    print(f"   History saved to: {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
