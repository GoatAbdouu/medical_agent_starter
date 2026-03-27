"""
VLM-based Explanation Module for Medical Vision AI Agent.

Provides natural-language explanations for skin diagnosis results.

Two backends are supported:
1. **BLIP** (``transformers`` library) — if available, uses the
   ``Salesforce/blip-image-captioning-base`` model to describe the image,
   then combines that caption with the diagnosis data into a full explanation.
2. **Template fallback** — if ``transformers`` is not installed or the model
   cannot be loaded, a high-quality multi-paragraph explanation is generated
   from the structured diagnosis data alone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from PIL import Image

if TYPE_CHECKING:
    from medical_agent.core.skin_disease_classifier import SkinDiagnosisResult

# ---------------------------------------------------------------------------
# Optional BLIP import
# ---------------------------------------------------------------------------
try:
    from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_DESCRIPTION: dict[str, str] = {
    "léger": (
        "The condition appears mild. It is generally benign and does not require "
        "emergency care, though monitoring and a routine medical check-up are advisable."
    ),
    "modéré": (
        "The condition is of moderate severity. While it is not immediately life-threatening, "
        "you should schedule an appointment with a dermatologist or general practitioner "
        "in the near future."
    ),
    "urgent": (
        "The condition has been flagged as potentially serious. Please seek medical attention "
        "as soon as possible — ideally within the next few days."
    ),
    "critique": (
        "⚠️ URGENT — The analysis indicates a potentially critical condition. "
        "You are strongly advised to consult a dermatologist or oncologist without delay. "
        "Early diagnosis and treatment are essential."
    ),
}

_NEXT_STEPS: dict[str, str] = {
    "léger": (
        "You can monitor the lesion at home using the ABCDE rule "
        "(Asymmetry, Border, Colour, Diameter, Evolution). "
        "Schedule a routine check-up with your doctor if you notice any changes."
    ),
    "modéré": (
        "Book an appointment with a dermatologist for a clinical examination. "
        "Avoid scratching or irritating the affected area in the meantime."
    ),
    "urgent": (
        "Contact your doctor or a dermatologist promptly to arrange a consultation. "
        "Do not attempt self-treatment."
    ),
    "critique": (
        "Seek specialist medical care immediately. "
        "A dermatologist or oncologist should perform a thorough clinical examination, "
        "and a biopsy may be required for a definitive diagnosis."
    ),
}


class VLMExplainer:
    """
    Generates natural-language explanations for skin diagnosis results.

    Parameters
    ----------
    use_blip : bool
        If ``True`` (default), attempt to load the BLIP model from HuggingFace.
        If ``False`` or if BLIP cannot be loaded, the template-based fallback
        is used automatically.
    blip_model_name : str
        HuggingFace model identifier for the BLIP image-captioning model.
    """

    def __init__(
        self,
        use_blip: bool = True,
        blip_model_name: str = _BLIP_MODEL_NAME,
    ) -> None:
        self._blip_processor: Optional[object] = None
        self._blip_model: Optional[object] = None
        self._blip_ready = False

        if use_blip and _TRANSFORMERS_AVAILABLE:
            self._try_load_blip(blip_model_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_load_blip(self, model_name: str) -> None:
        """Attempt to load the BLIP processor and model (silently fails)."""
        try:
            self._blip_processor = BlipProcessor.from_pretrained(model_name)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._blip_model.to(device)  # type: ignore[union-attr]
            self._blip_model.eval()  # type: ignore[union-attr]
            self._blip_device = device
            self._blip_ready = True
        except Exception:
            self._blip_ready = False

    def _blip_caption(self, image: Image.Image) -> str:
        """Generate a short BLIP caption for the image."""
        inputs = self._blip_processor(  # type: ignore[call-arg]
            images=image, return_tensors="pt"
        ).to(self._blip_device)
        with torch.no_grad():
            out = self._blip_model.generate(**inputs, max_new_tokens=50)  # type: ignore[union-attr]
        caption: str = self._blip_processor.decode(  # type: ignore[union-attr]
            out[0], skip_special_tokens=True
        )
        return caption

    @staticmethod
    def _template_explanation(
        image: Image.Image,  # noqa: ARG004  (kept for API symmetry)
        diagnosis_result: "SkinDiagnosisResult",
    ) -> str:
        """
        Generate a high-quality multi-paragraph explanation from structured data.

        This is the fallback used when BLIP is unavailable.
        """
        top = diagnosis_result.top_diagnosis
        if top is None:
            return (
                "The analysis could not identify a specific skin condition with "
                "sufficient confidence. Please consult a dermatologist for a "
                "professional examination.\n\n"
                "⚠️ This is not medical advice. Consult a healthcare professional."
            )

        confidence_pct = top.confidence * 100
        if confidence_pct >= 80:
            conf_desc = "high confidence"
        elif confidence_pct >= 50:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "low confidence"

        severity = top.severity
        severity_text = _SEVERITY_DESCRIPTION.get(
            severity,
            "The severity could not be determined from the available information.",
        )
        next_steps_text = _NEXT_STEPS.get(
            severity,
            "Please consult a healthcare professional for appropriate guidance.",
        )

        # Build the list of alternative diagnoses (excluding the top one)
        alternatives = diagnosis_result.candidates[1:4]
        alt_lines = "\n".join(
            f"  • {c.readable_name} ({c.confidence * 100:.1f}%)"
            for c in alternatives
        )
        alt_section = (
            f"\n\nAlternative possibilities considered by the model:\n{alt_lines}"
            if alt_lines
            else ""
        )

        explanation = (
            f"## Skin Analysis Report\n\n"
            f"**Detected condition:** {top.readable_name}\n"
            f"**Confidence:** {confidence_pct:.1f}% ({conf_desc})\n"
            f"**Severity assessment:** {severity}\n\n"
            f"### What was detected\n"
            f"The ensemble of three deep learning models (EfficientNet-B3, MobileNetV2, "
            f"and ResNet50) analysed the provided skin image and identified the most likely "
            f"condition as **{top.readable_name}** with a confidence of "
            f"**{confidence_pct:.1f}%**.\n\n"
            f"### Severity\n"
            f"{severity_text}\n\n"
            f"### Clinical advice\n"
            f"{top.advice}\n\n"
            f"### Recommended next steps\n"
            f"{next_steps_text}"
            f"{alt_section}\n\n"
            f"---\n"
            f"⚠️ **Medical Disclaimer**: This analysis is generated by an artificial "
            f"intelligence system and **does not constitute a medical diagnosis**. "
            f"It is intended for informational purposes only. Always consult a licensed "
            f"dermatologist or healthcare professional for any skin-related concerns.\n"
            f"⚠️ This is not medical advice. Consult a healthcare professional."
        )
        return explanation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        image: Image.Image,
        diagnosis_result: "SkinDiagnosisResult",
    ) -> str:
        """
        Generate a detailed natural-language explanation for a diagnosis result.

        Parameters
        ----------
        image : PIL.Image.Image
            The original skin image that was analysed.
        diagnosis_result : SkinDiagnosisResult
            The ensemble classification result.

        Returns
        -------
        str
            Multi-paragraph explanation including what was detected, confidence
            interpretation, severity assessment, recommended next steps, and
            the mandatory medical disclaimer.
        """
        if self._blip_ready:
            try:
                caption = self._blip_caption(image)
                top = diagnosis_result.top_diagnosis
                confidence_pct = (top.confidence * 100) if top else 0.0
                condition = top.readable_name if top else "unknown condition"
                severity = top.severity if top else "unknown"

                severity_text = _SEVERITY_DESCRIPTION.get(severity, "")
                next_steps_text = _NEXT_STEPS.get(severity, "")
                advice = top.advice if top else ""

                explanation = (
                    f"## Skin Analysis Report (VLM-enhanced)\n\n"
                    f"**Visual description:** {caption}\n\n"
                    f"**Detected condition:** {condition}\n"
                    f"**Confidence:** {confidence_pct:.1f}%\n"
                    f"**Severity:** {severity}\n\n"
                    f"### Severity assessment\n{severity_text}\n\n"
                    f"### Clinical advice\n{advice}\n\n"
                    f"### Recommended next steps\n{next_steps_text}\n\n"
                    f"---\n"
                    f"⚠️ **Medical Disclaimer**: This analysis is generated by an "
                    f"artificial intelligence system and **does not constitute a medical "
                    f"diagnosis**. Always consult a licensed healthcare professional.\n"
                    f"⚠️ This is not medical advice. Consult a healthcare professional."
                )
                return explanation
            except Exception:
                pass  # Fall through to template

        return self._template_explanation(image, diagnosis_result)
