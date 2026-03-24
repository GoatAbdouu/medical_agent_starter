"""
Script d'entraînement du classificateur de maladies cutanées
Utilise MobileNetV2 avec transfer learning 2 phases sur le dataset Kaggle Skin Disease
Phase 1 : entraînement de la tête seulement (base gelée)
Phase 2 : fine-tuning complet avec learning rates différenciés + early stopping
Sauvegarde automatiquement l'historique dans models/training_history.json
"""
import argparse
import json
import sys
from pathlib import Path

# Ajouter la racine du projet au chemin Python
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

from medical_agent.core.skin_disease_classifier import (
    FOLDER_TO_DISEASE,
    INFERENCE_TRANSFORMS,
    SkinDiseaseModel,
)

# Nombre d'époques réservées à la Phase 1 (entraînement de la tête seulement)
PHASE1_EPOCHS = 10


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
        default=40,
        help="Nombre total d'époques d'entraînement (défaut: 40 = 10 Phase1 + 30 Phase2)",
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
        help="Taux d'apprentissage initial pour la Phase 1 (défaut: 1e-3)",
    )
    return parser.parse_args()


def build_train_transforms():
    """Retourne les transformations d'augmentation agressives pour l'entraînement."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """Exécute une époque d'entraînement ou de validation. Retourne (loss, accuracy)."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Dossier de données introuvable : {data_dir}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_path.parent / "training_history.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositif utilisé : {device}")

    # -----------------------------------------------------------------
    # Chargement du dataset avec ImageFolder (utilise le tri alphabétique)
    # -----------------------------------------------------------------
    train_transforms = build_train_transforms()
    val_transforms = INFERENCE_TRANSFORMS

    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transforms)

    # Compter les échantillons par classe
    class_counts = [0] * len(full_dataset.classes)
    for _, label in full_dataset.samples:
        class_counts[label] += 1

    # Construire la liste des noms de classes lisibles
    class_names_fr = []
    print("\n📂 Correspondance des classes et distribution :")
    for idx, class_name in enumerate(full_dataset.classes):
        readable = FOLDER_TO_DISEASE.get(class_name, class_name)
        class_names_fr.append(readable)
        print(f"   [{idx}] {class_name}  →  {readable}  ({class_counts[idx]} images)")

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    print(f"\n📊 Images totales : {n_total}")
    print(f"   ├─ Entraînement : {n_train}")
    print(f"   └─ Validation   : {n_val}")

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Dataset de validation avec transforms de validation (sans augmentation)
    val_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_transforms)
    val_final = Subset(val_dataset, val_subset.indices)

    # Calculer les effectifs par classe dans le split d'entraînement
    num_classes = len(full_dataset.classes)
    train_class_counts = [0] * num_classes
    for i in train_subset.indices:
        train_class_counts[full_dataset.samples[i][1]] += 1

    # WeightedRandomSampler pour compenser le déséquilibre des classes
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
    print(f"\n🧠 Modèle : MobileNetV2 (tête profonde 1280→512→256→{num_classes})")
    print(f"   ├─ Paramètres totaux    : {total_params:,}")
    print(f"   └─ Paramètres entraînés : {trainable_params:,}")

    # -----------------------------------------------------------------
    # Perte pondérée pour compenser le déséquilibre des classes
    # -----------------------------------------------------------------
    class_weights = torch.tensor(
        [n_train / (num_classes * c) for c in train_class_counts],
        dtype=torch.float,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------------------------------------------
    # Historique d'entraînement (sauvegardé en JSON après chaque époque)
    # -----------------------------------------------------------------
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "class_names": class_names_fr,
        "class_counts": class_counts,
    }

    best_val_acc = 0.0
    phase2_epochs = max(0, args.epochs - PHASE1_EPOCHS)

    # =================================================================
    # PHASE 1 — Entraînement de la tête uniquement
    # =================================================================
    print(f"\n🔬 Phase 1 : Entraînement du classificateur (tête uniquement) — {PHASE1_EPOCHS} époques")

    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer1, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        # Sauvegarde crash-safe après chaque époque
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        icon = "✅" if val_acc > best_val_acc else "  "
        print(
            f"  [P1] Époque {epoch:3d}/{PHASE1_EPOCHS} | "
            f"Train — perte: {train_loss:.4f}  acc: {train_acc:.2%} | "
            f"Val — perte: {val_loss:.4f}  acc: {val_acc:.2%}  {icon}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"      💾 Meilleur modèle sauvegardé → {output_path}  (acc val: {best_val_acc:.2%})")

    # =================================================================
    # PHASE 2 — Fine-tuning complet du réseau
    # =================================================================
    if phase2_epochs > 0:
        print(f"\n🔬 Phase 2 : Fine-tuning complet du réseau — {phase2_epochs} époques")

        model.unfreeze_backbone()

        trainable_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   └─ Paramètres entraînés (Phase 2) : {trainable_params2:,}")

        # Learning rates différenciés : backbone lent, tête plus rapide
        optimizer2 = torch.optim.Adam([
            {"params": model.base.features.parameters(), "lr": 1e-5},
            {"params": model.base.classifier.parameters(), "lr": 1e-4},
        ])
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, mode="max", patience=5, factor=0.5
        )

        # Early stopping
        early_stop_patience = 8
        epochs_no_improve = 0

        for epoch in range(1, phase2_epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer2, device, is_train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)

            scheduler2.step(val_acc)

            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_acc"].append(round(val_acc, 4))

            # Sauvegarde crash-safe après chaque époque
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            icon = "✅" if val_acc > best_val_acc else "  "
            print(
                f"  [P2] Époque {epoch:3d}/{phase2_epochs} | "
                f"Train — perte: {train_loss:.4f}  acc: {train_acc:.2%} | "
                f"Val — perte: {val_loss:.4f}  acc: {val_acc:.2%}  {icon}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), output_path)
                print(f"      💾 Meilleur modèle sauvegardé → {output_path}  (acc val: {best_val_acc:.2%})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"\n⏹️  Early stopping déclenché après {epoch} époques en Phase 2 (patience={early_stop_patience})")
                    break

    print(f"\n🎉 Entraînement terminé !")
    print(f"   Meilleure précision de validation : {best_val_acc:.2%}")
    print(f"   Modèle sauvegardé dans            : {output_path}")
    print(f"   Historique sauvegardé dans         : {history_path}")


if __name__ == "__main__":
    main()