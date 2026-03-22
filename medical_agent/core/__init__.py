"""Core modules"""
from medical_agent.core.agent import MedicalAgent
from medical_agent.core.symptom_extractor import SymptomExtractor
from medical_agent.core.disease_predictor import DiseasePredictor
from medical_agent.core.triage_system import TriageSystem

# Deep Learning predictor (optionnel)
try:
    from medical_agent.core.deep_learning_predictor import DeepLearningPredictor
except ImportError:
    DeepLearningPredictor = None

__all__ = [
    'MedicalAgent',
    'SymptomExtractor', 
    'DiseasePredictor',
    'TriageSystem',
    'DeepLearningPredictor'
]
