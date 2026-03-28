"""
Ensemble Skin Disease Classifier.

Three transfer-learning models (EfficientNet-B3, MobileNetV2, ResNet50)
are combined via an ensemble strategy:

  - If all three models agree on the top-1 class  → average their confidence scores.
  - If they disagree                               → select the prediction with the
                                                     highest individual confidence.

Each backbone model shares a common deep classifier head:
    in_features → Linear(512) → BN → ReLU → Dropout(0.4)
                → Linear(256) → BN → ReLU → Dropout(0.3)
                → Linear(num_classes)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import (
    EfficientNet_B3_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
)

from medical_agent.core.image_pipeline import ImagePipeline
from medical_agent.core.skin_disease_classifier import (
    DISEASE_INFO,
    FOLDER_TO_DISEASE,
    URGENT_ATTENTION_THRESHOLD,
    SkinDiagnosisCandidate,
    SkinDiagnosisResult,
)

NUM_CLASSES: int = len(FOLDER_TO_DISEASE)
CLASS_NAMES: List[str] = list(FOLDER_TO_DISEASE.keys())


# ---------------------------------------------------------------------------
# Shared helper: build a deep classifier head
# ---------------------------------------------------------------------------

def _make_classifier_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Construct a shared deep classifier head.

    Architecture: in_features → 512 → BN → ReLU → Dropout(0.4)
                              → 256 → BN → ReLU → Dropout(0.3)
                              → num_classes
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


# ---------------------------------------------------------------------------
# Individual backbone models
# ---------------------------------------------------------------------------

class EfficientNetModel(nn.Module):
    """
    EfficientNet-B3 with a custom classifier head for skin disease classification.

    Input : 3 × 224 × 224 RGB tensor
    Output: logits vector of length ``num_classes``
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_base: bool = True) -> None:
        super().__init__()
        self.base = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        in_features: int = self.base.classifier[1].in_features  # 1536 for EfficientNet-B3
        self.base.classifier = _make_classifier_head(in_features, num_classes)

    def freeze_base(self) -> None:
        """Freeze all backbone parameters (only train the classifier head)."""
        for param in self.base.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.base(x)


class MobileNetModel(nn.Module):
    """
    MobileNetV2 with a custom classifier head for skin disease classification.

    Kept API-compatible with the existing ``SkinDiseaseModel`` so that
    weights trained by ``train_skin_classifier.py`` can be reused.

    Input : 3 × 224 × 224 RGB tensor
    Output: logits vector of length ``num_classes``
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_base: bool = True) -> None:
        super().__init__()
        self.base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        in_features: int = self.base.classifier[1].in_features  # 1280 for MobileNetV2
        self.base.classifier = _make_classifier_head(in_features, num_classes)

    def freeze_base(self) -> None:
        """Freeze all backbone parameters (only train the classifier head)."""
        for param in self.base.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.base(x)


class ResNetModel(nn.Module):
    """
    ResNet50 with a custom classifier head for skin disease classification.

    Input : 3 × 224 × 224 RGB tensor
    Output: logits vector of length ``num_classes``
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_base: bool = True) -> None:
        super().__init__()
        self.base = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        in_features: int = self.base.fc.in_features  # 2048 for ResNet50
        self.base.fc = _make_classifier_head(in_features, num_classes)

    def freeze_base(self) -> None:
        """Freeze all backbone parameters (only train the classifier head)."""
        # Freeze everything except the custom head (base.fc)
        for name, param in self.base.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.base(x)


# ---------------------------------------------------------------------------
# Ensemble classifier
# ---------------------------------------------------------------------------

class EnsembleClassifier:
    """
    Ensemble of EfficientNet-B3, MobileNetV2, and ResNet50 for skin disease
    classification.

    Decision logic
    --------------
    - **All three models agree** on the top-1 class:
      → Final confidence = average of the three individual confidence scores.
    - **Models disagree**:
      → Take the prediction with the highest individual confidence among all three.

    Parameters
    ----------
    efficientnet_path : str or Path, optional
        Path to the saved EfficientNet-B3 weights (``*.pth``).
    mobilenet_path : str or Path, optional
        Path to the saved MobileNetV2 weights (``*.pth``).
    resnet_path : str or Path, optional
        Path to the saved ResNet50 weights (``*.pth``).
    """

    MODEL_NAMES: Tuple[str, ...] = ("EfficientNet-B3", "MobileNetV2", "ResNet50")

    def __init__(
        self,
        efficientnet_path: str = "models/efficientnet_skin.pth",
        mobilenet_path: str = "models/mobilenet_skin.pth",
        resnet_path: str = "models/resnet_skin.pth",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = ImagePipeline()

        # Paths
        self._paths: Dict[str, Path] = {
            "EfficientNet-B3": Path(efficientnet_path),
            "MobileNetV2": Path(mobilenet_path),
            "ResNet50": Path(resnet_path),
        }

        # Lazy-loaded model instances
        self._models: Dict[str, Optional[nn.Module]] = {
            "EfficientNet-B3": None,
            "MobileNetV2": None,
            "ResNet50": None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, name: str) -> Optional[nn.Module]:
        """Load a model from disk if not already loaded. Returns None if unavailable."""
        if self._models[name] is not None:
            return self._models[name]

        path = self._paths[name]
        if not path.exists():
            return None

        if name == "EfficientNet-B3":
            model: nn.Module = EfficientNetModel(num_classes=NUM_CLASSES, freeze_base=False)
        elif name == "MobileNetV2":
            model = MobileNetModel(num_classes=NUM_CLASSES, freeze_base=False)
        else:  # ResNet50
            model = ResNetModel(num_classes=NUM_CLASSES, freeze_base=False)

        state = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._models[name] = model
        return model

    def _run_single_model(
        self, model: nn.Module, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Run inference and return class probabilities (shape: num_classes)."""
        with torch.no_grad():
            logits = model(tensor)
            return torch.softmax(logits, dim=1)[0]

    @staticmethod
    def _probs_to_candidates(
        probs: torch.Tensor, top_n: int
    ) -> List[SkinDiagnosisCandidate]:
        """Convert a probability tensor to a list of SkinDiagnosisCandidate."""
        top_n = min(top_n, len(CLASS_NAMES))
        top_probs, top_indices = torch.topk(probs, top_n)
        candidates: List[SkinDiagnosisCandidate] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            folder = CLASS_NAMES[idx]
            readable = FOLDER_TO_DISEASE[folder]
            info = DISEASE_INFO.get(readable, {})
            candidates.append(
                SkinDiagnosisCandidate(
                    disease_name=folder,
                    readable_name=readable,
                    confidence=round(prob, 4),
                    severity=info.get("severity", "inconnu"),
                    color=info.get("color", "gray"),
                    advice=info.get("advice", "Consultez un médecin."),
                    urgency=info.get("urgency", "inconnu"),
                )
            )
        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> List[str]:
        """Return the names of models whose weight files exist on disk."""
        return [name for name, path in self._paths.items() if path.exists()]

    def get_individual_predictions(
        self, image: Image.Image, top_n: int = 5
    ) -> Dict[str, List[SkinDiagnosisCandidate]]:
        """
        Run each loaded model independently and return per-model predictions.

        Useful for displaying individual model outputs in the UI or for debugging.

        Parameters
        ----------
        image : PIL.Image.Image
            Input skin image.
        top_n : int
            Number of top predictions to return per model.

        Returns
        -------
        dict[str, list[SkinDiagnosisCandidate]]
            Mapping from model name to its list of top-n candidates.
            Models whose weights are not available are silently omitted.
        """
        tensor = self.pipeline.preprocess(image).to(self.device)
        results: Dict[str, List[SkinDiagnosisCandidate]] = {}

        for name in self.MODEL_NAMES:
            model = self._load_model(name)
            if model is None:
                continue
            probs = self._run_single_model(model, tensor)
            results[name] = self._probs_to_candidates(probs, top_n)

        return results

    def predict(self, image: Image.Image, top_n: int = 5) -> SkinDiagnosisResult:
        """
        Run ensemble inference on a single PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input skin image (any mode; will be converted to RGB internally).
        top_n : int
            Number of candidate diagnoses to include in the result.

        Returns
        -------
        SkinDiagnosisResult
            Ensemble result with top-n candidates, urgency flag, and disclaimer.

        Notes
        -----
        If no model weights are found, the method raises ``RuntimeError``.
        If only some models are available, the ensemble uses those that are.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.pipeline.preprocess(image).to(self.device)

        # Collect per-model probabilities
        all_probs: Dict[str, torch.Tensor] = {}
        for name in self.MODEL_NAMES:
            model = self._load_model(name)
            if model is not None:
                all_probs[name] = self._run_single_model(model, tensor)

        if not all_probs:
            raise RuntimeError(
                "Aucun modèle d'ensemble disponible. "
                "Entraînez d'abord les modèles avec scripts/train_ensemble.py"
            )

        # Determine top-1 class per model
        top1_per_model = {
            name: int(probs.argmax().item()) for name, probs in all_probs.items()
        }

        # Ensemble decision logic
        unique_top1 = set(top1_per_model.values())

        if len(all_probs) >= 3 and len(unique_top1) == 1:
            # All models agree → average probabilities
            stacked = torch.stack(list(all_probs.values()))
            ensemble_probs = stacked.mean(dim=0)
            agreement = True
        else:
            # Models disagree (or fewer than 3 models) → highest-confidence wins
            best_name = max(
                all_probs.keys(),
                key=lambda n: float(all_probs[n].max()),
            )
            ensemble_probs = all_probs[best_name]
            agreement = False

        candidates = self._probs_to_candidates(ensemble_probs, top_n)
        top_diagnosis = candidates[0] if candidates else None

        needs_urgent = any(
            c.urgency in ("critique", "urgent") and c.confidence > URGENT_ATTENTION_THRESHOLD
            for c in candidates
        )

        # Append ensemble metadata to the disclaimer
        model_list = ", ".join(all_probs.keys())
        strategy = (
            "moyenne des confidences (tous les modèles sont d'accord)"
            if agreement
            else "confiance individuelle la plus élevée (les modèles divergent)"
        )
        disclaimer = (
            f"⚠️ Avertissement : Ce diagnostic est généré par une intelligence artificielle "
            f"(ensemble {model_list} — stratégie : {strategy}) "
            f"et ne remplace en aucun cas l'avis d'un médecin ou d'un dermatologue qualifié. "
            f"Consultez toujours un professionnel de santé pour tout problème cutané. "
            f"⚠️ This is not medical advice. Consult a healthcare professional."
        )

        result = SkinDiagnosisResult(
            candidates=candidates,
            top_diagnosis=top_diagnosis,
            needs_urgent_attention=needs_urgent,
            disclaimer=disclaimer,
        )
        # Attach the agreement flag as a dynamic attribute so callers can inspect it
        # without requiring a dataclass change.
        result._ensemble_agreement = agreement  # type: ignore[attr-defined]
        return result
