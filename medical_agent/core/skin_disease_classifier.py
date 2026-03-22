"""
Classificateur de Maladies Cutanées
Utilise MobileNetV2 pour la reconnaissance d'images de maladies de la peau
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

IMAGE_SIZE = 128

FOLDER_TO_DISEASE = {
    "1. Eczema 1677": "Eczéma",
    "2. Warts Molluscum and other Viral Infections - 2103": "Verrues / Molluscum",
    "3. Melanoma 15.75k": "Mélanome",
    "4. Atopic Dermatitis - 1.25k": "Dermatite Atopique",
    "5. Basal Cell Carcinoma (BCC) 3323": "Carcinome Basocellulaire",
    "6. Melanocytic Nevi (Moles) - 7970": "Nævus Mélanocytaire (Grain de beauté)",
    "7. Benign Keratosis-like Lesions (BKL) 2624": "Kératose Bénigne",
    "8. Psoriasis pictures Lichen Planus and related diseases - 2k": "Psoriasis / Lichen Plan",
    "9. Seborrheic Keratoses and other Benign Tumors - 1.8k": "Kératose Séborrhéique",
    "10. Tinea Ringworm Candidiasis and other Fungal Infections - 1702": "Mycose / Teigne",
}

DISEASE_INFO = {
    "Eczéma": {
        "severity": "modéré",
        "color": "yellow",
        "advice": (
            "Hydratez régulièrement la peau avec des émollients. "
            "Évitez les irritants et allergènes connus. "
            "Consultez un dermatologue pour un traitement adapté (corticoïdes topiques si nécessaire)."
        ),
        "urgency": "modéré",
    },
    "Verrues / Molluscum": {
        "severity": "léger",
        "color": "green",
        "advice": (
            "Les verrues et molluscums sont bénins mais contagieux. "
            "Évitez de les gratter ou de les toucher. "
            "Consultez un médecin ou un dermatologue pour un traitement (cryothérapie, acide salicylique)."
        ),
        "urgency": "léger",
    },
    "Mélanome": {
        "severity": "critique",
        "color": "red",
        "advice": (
            "⚠️ URGENCE MÉDICALE. Le mélanome est un cancer de la peau grave. "
            "Consultez un dermatologue ou un oncologue en URGENCE. "
            "Ne tardez pas : un diagnostic et traitement précoces sont essentiels à la survie."
        ),
        "urgency": "critique",
    },
    "Dermatite Atopique": {
        "severity": "modéré",
        "color": "yellow",
        "advice": (
            "Appliquez des émollients quotidiennement. "
            "Évitez les produits irritants et les vêtements synthétiques. "
            "Consultez un dermatologue pour un suivi et un traitement personnalisé."
        ),
        "urgency": "modéré",
    },
    "Carcinome Basocellulaire": {
        "severity": "urgent",
        "color": "orange",
        "advice": (
            "Le carcinome basocellulaire est le cancer de la peau le plus courant. "
            "Consultez rapidement un dermatologue pour une biopsie et un traitement. "
            "Une intervention précoce est généralement curative."
        ),
        "urgency": "urgent",
    },
    "Nævus Mélanocytaire (Grain de beauté)": {
        "severity": "léger",
        "color": "green",
        "advice": (
            "La plupart des grains de beauté sont bénins. "
            "Surveillez tout changement de taille, forme ou couleur (règle ABCDE). "
            "Consultez un dermatologue en cas de doute ou d'évolution."
        ),
        "urgency": "léger",
    },
    "Kératose Bénigne": {
        "severity": "léger",
        "color": "green",
        "advice": (
            "Les kératoses bénignes ne nécessitent généralement pas de traitement. "
            "Consultez un médecin si elles deviennent gênantes ou si vous avez un doute diagnostique."
        ),
        "urgency": "léger",
    },
    "Psoriasis / Lichen Plan": {
        "severity": "modéré",
        "color": "yellow",
        "advice": (
            "Le psoriasis est une maladie chronique inflammatoire. "
            "Consultez un dermatologue pour un traitement adapté (topiques, photothérapie, biologiques). "
            "Évitez les facteurs déclenchants (stress, alcool, certains médicaments)."
        ),
        "urgency": "modéré",
    },
    "Kératose Séborrhéique": {
        "severity": "léger",
        "color": "green",
        "advice": (
            "Les kératoses séborrhéiques sont des lésions bénignes très fréquentes. "
            "Aucun traitement n'est nécessaire sauf raison esthétique ou inconfort. "
            "Consultez un dermatologue en cas de doute avec un mélanome."
        ),
        "urgency": "léger",
    },
    "Mycose / Teigne": {
        "severity": "modéré",
        "color": "yellow",
        "advice": (
            "Les infections fongiques sont traitables avec des antifongiques. "
            "Consultez un médecin pour obtenir un traitement approprié (crème ou comprimés). "
            "Évitez de partager serviettes ou vêtements pour prévenir la contagion."
        ),
        "urgency": "modéré",
    },
}

URGENT_ATTENTION_THRESHOLD = 0.15
"""Confidence threshold above which an urgent/critical disease triggers needs_urgent_attention."""

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class SkinDiagnosisCandidate:
    disease_name: str
    readable_name: str
    confidence: float
    severity: str
    color: str
    advice: str
    urgency: str


@dataclass
class SkinDiagnosisResult:
    candidates: List[SkinDiagnosisCandidate] = field(default_factory=list)
    top_diagnosis: Optional[SkinDiagnosisCandidate] = None
    needs_urgent_attention: bool = False
    disclaimer: str = (
        "⚠️ Avertissement : Ce diagnostic est généré par une intelligence artificielle "
        "et ne remplace en aucun cas l'avis d'un médecin ou d'un dermatologue qualifié. "
        "Consultez toujours un professionnel de santé pour tout problème cutané."
    )


class SkinDiseaseModel(nn.Module):
    """MobileNetV2 adapté pour la classification de maladies cutanées"""

    def __init__(self, num_classes: int = 10, freeze_base: bool = True):
        super().__init__()
        self.base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


class SkinDiseaseClassifier:
    """
    Interface haut niveau pour la classification de maladies cutanées.
    Charge le modèle à la demande et effectue l'inférence sur des images PIL.
    """

    CLASS_NAMES = list(FOLDER_TO_DISEASE.keys())

    def __init__(self, model_path: str = "models/skin_disease_model.pth"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SkinDiseaseModel] = None

    def _load_model(self) -> None:
        """Charge le modèle depuis le disque"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {self.model_path}. "
                "Entraînez-le d'abord avec scripts/train_skin_classifier.py"
            )
        model = SkinDiseaseModel(num_classes=len(self.CLASS_NAMES), freeze_base=False)
        state = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._model = model

    @property
    def model(self) -> SkinDiseaseModel:
        if self._model is None:
            self._load_model()
        return self._model

    def predict(self, image, top_n: int = 5) -> SkinDiagnosisResult:
        """
        Prédit les maladies cutanées à partir d'une image PIL.

        Args:
            image: Image PIL (RGB)
            top_n: Nombre de candidats à retourner

        Returns:
            SkinDiagnosisResult avec les candidats triés par confiance
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_n = min(top_n, len(self.CLASS_NAMES))
        top_probs, top_indices = torch.topk(probabilities, top_n)

        candidates: List[SkinDiagnosisCandidate] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            folder_name = self.CLASS_NAMES[idx]
            readable_name = FOLDER_TO_DISEASE[folder_name]
            info = DISEASE_INFO.get(readable_name, {})
            candidates.append(
                SkinDiagnosisCandidate(
                    disease_name=folder_name,
                    readable_name=readable_name,
                    confidence=round(prob, 4),
                    severity=info.get("severity", "inconnu"),
                    color=info.get("color", "gray"),
                    advice=info.get("advice", "Consultez un médecin."),
                    urgency=info.get("urgency", "inconnu"),
                )
            )

        top_diagnosis = candidates[0] if candidates else None

        needs_urgent_attention = any(
            c.urgency in ("critique", "urgent") and c.confidence > URGENT_ATTENTION_THRESHOLD
            for c in candidates
        )

        return SkinDiagnosisResult(
            candidates=candidates,
            top_diagnosis=top_diagnosis,
            needs_urgent_attention=needs_urgent_attention,
        )
