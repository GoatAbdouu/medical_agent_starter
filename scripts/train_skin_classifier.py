"""
Script d'entraînement du classificateur de maladies cutanées
Utilise MobileNetV2 avec transfer learning sur le dataset Kaggle Skin Disease
"""
import argparse
import sys
from pathlib import Path

# Ajouter la racine du projet au chemin Python
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

from medical_agent.core.skin_disease_classifier import (
    FOLDER_TO_DISEASE,
    IMAGE_SIZE,
    INFERENCE_TRANSFORMS,
    SkinDiseaseModel,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement du classificateur de maladies cutanées"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Chemin vers le dossier racine du dataset (contient les 10 sous-dossiers)",
    )
    parser.add_argument(
        "--output",
        default="models/skin_disease_model.pth",
        help="Chemin de sauvegarde du modèle (défaut: models/skin_disease_model.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Nombre d'époques d'entraînement (défaut: 20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Taille des lots (défaut: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Taux d'apprentissage initial (défaut: 1e-3)",
    )
    return parser.parse_args()

def build_transforms():
    """Retourne les transformations d'augmentation (entraînement) et de validation."""
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = INFERENCE_TRANSFORMS
    return train_transforms, val_transforms

def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Dossier de données introuvable : {data_dir}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositif utilisé : {device}")

    # -----------------------------------------------------------------
    # Chargement du dataset avec ImageFolder (utilise le tri alphabétique)
    # -----------------------------------------------------------------
    train_transforms, val_transforms = build_transforms()

    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transforms)

    # Count samples per class
    class_counts = [0] * len(full_dataset.classes)
    for _, label in full_dataset.samples:
        class_counts[label] += 1

    print("\n📂 Correspondance des classes et distribution :")
    for idx, class_name in enumerate(full_dataset.classes):
        readable = FOLDER_TO_DISEASE.get(class_name, class_name)
        print(f"   [{idx}] {class_name}  →  {readable}  ({class_counts[idx]} images)")

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    print(f"\n📊 Images totales : {n_total}")
    print(f"   ├─ Entraînement : {n_train}")
    print(f"   └─ Validation   : {n_val}")

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Create a separate ImageFolder with validation transforms using the same indices
    val_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_transforms)
    from torch.utils.data import Subset
    val_final = Subset(val_dataset, val_subset.indices)

    # Compute per-class counts from the training split to avoid division by zero
    # and ensure weights match the actual training distribution
    num_classes = len(full_dataset.classes)
    train_class_counts = [0] * num_classes
    for i in train_subset.indices:
        train_class_counts[full_dataset.samples[i][1]] += 1

    # Build sample weights for WeightedRandomSampler
    train_sample_weights = [
        1.0 / train_class_counts[full_dataset.samples[i][1]] for i in train_subset.indices
    ]
    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
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

    # -----------------------------------------------------------------
    # Modèle
    # -----------------------------------------------------------------
    model = SkinDiseaseModel(num_classes=num_classes, freeze_base=True)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modèle : MobileNetV2")
    print(f"   ├─ Paramètres totaux    : {total_params:,}")
    print(f"   └─ Paramètres entraînés : {trainable_params:,}")

    # -----------------------------------------------------------------
    # Optimiseur et scheduler
    # -----------------------------------------------------------------
    class_weights = torch.tensor(
        [n_train / (num_classes * c) for c in train_class_counts],
        dtype=torch.float,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    # -----------------------------------------------------------------
    # Boucle d'entraînement
    # -----------------------------------------------------------------
    best_val_acc = 0.0
    print(f"\n🚀 Début de l'entraînement ({args.epochs} époques) ...\n")

    for epoch in range(1, args.epochs + 1):
        # --- Entraînement ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_acc)

        # Icône de progression
        icon = "✅" if val_acc > best_val_acc else "  "

        print(
            f"Époque {epoch:3d}/{args.epochs} | "
            f"Train — perte: {train_loss:.4f}  acc: {train_acc:.2%} | "
            f"Val — perte: {val_loss:.4f}  acc: {val_acc:.2%}  {icon}"
        )

        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"   💾 Meilleur modèle sauvegardé → {output_path}  (acc val: {best_val_acc:.2%})")

    print(f"\n🎉 Entraînement terminé !")
    print(f"   Meilleure précision de validation : {best_val_acc:.2%}")
    print(f"   Modèle sauvegardé dans            : {output_path}")


if __name__ == "__main__":
    main()