"""
Agent Médical - Système de diagnostic médical intelligent
"""

__version__ = "2.0.0"
__author__ = "Medical Agent Team"

from medical_agent.core.agent import MedicalAgent
from medical_agent.models.data_models import PatientInput, DiagnosisResult

__all__ = ["MedicalAgent", "PatientInput", "DiagnosisResult"]
