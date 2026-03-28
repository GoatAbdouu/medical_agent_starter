"""Core modules"""
from medical_agent.core.agent import MedicalAgent
from medical_agent.core.symptom_extractor import SymptomExtractor
from medical_agent.core.disease_predictor import DiseasePredictor
from medical_agent.core.triage_system import TriageSystem
from medical_agent.core.system_prompt import SYSTEM_PROMPT

# Deep Learning predictor (optionnel)
try:
    from medical_agent.core.deep_learning_predictor import DeepLearningPredictor
except ImportError:
    DeepLearningPredictor = None

# Ensemble classifier (optionnel — nécessite torchvision)
try:
    from medical_agent.core.ensemble_classifier import EnsembleClassifier
except ImportError:
    EnsembleClassifier = None

# VLM explainer (optionnel)
try:
    from medical_agent.core.vlm_explainer import VLMExplainer
except ImportError:
    VLMExplainer = None

# Image pipeline (optionnel — nécessite torchvision)
try:
    from medical_agent.core.image_pipeline import ImagePipeline
except ImportError:
    ImagePipeline = None

__all__ = [
    'MedicalAgent',
    'SymptomExtractor',
    'DiseasePredictor',
    'TriageSystem',
    'DeepLearningPredictor',
    'EnsembleClassifier',
    'VLMExplainer',
    'ImagePipeline',
    'SYSTEM_PROMPT',
]
