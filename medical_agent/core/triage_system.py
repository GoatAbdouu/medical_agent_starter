"""
Système de triage - Évalue l'urgence et les drapeaux rouges
"""
import re
from typing import List, Optional
from pathlib import Path

from medical_agent.models.data_models import PatientInput, TriageResult
from medical_agent.config.settings import settings


class TriageSystem:
    """Système de triage médical pour évaluer l'urgence"""
    
    def __init__(self, red_flags_path: Optional[Path] = None):
        """
        Args:
            red_flags_path: Chemin vers le fichier de configuration des drapeaux rouges
        """
        self.red_flags_path = red_flags_path or settings.RED_FLAGS_PATH
        self.patterns, self.rules = self._load_red_flags()
    
    def _load_red_flags(self):
        """Charge les patterns de drapeaux rouges"""
        config = settings.load_yaml_config(self.red_flags_path)
        
        patterns = []
        for item in config.get("patterns", []):
            patterns.append({
                "label": item["label"],
                "regex": re.compile(item["regex"], re.IGNORECASE)
            })
        
        rules = config.get("rules", [])
        
        return patterns, rules
    
    def _check_pattern_red_flags(self, text: str) -> List[str]:
        """Vérifie les drapeaux rouges basés sur les patterns"""
        red_flags = []
        
        for pattern in self.patterns:
            if pattern["regex"].search(text):
                red_flags.append(pattern["label"])
        
        return red_flags
    
    def _check_symptom_red_flags(self, patient: PatientInput) -> List[str]:
        """Vérifie les drapeaux rouges basés sur les symptômes"""
        red_flags = []
        
        # Température très élevée
        temp = patient.temperature
        if temp and temp >= settings.TEMP_HIGH_FEVER_THRESHOLD:
            red_flags.append(f"Hyperthermie ≥ {settings.TEMP_HIGH_FEVER_THRESHOLD}°C")
        
        # Difficultés respiratoires
        respiratory_symptoms = ["dyspnea", "essoufflement", "difficulté respiratoire"]
        if any(symp in patient.symptoms for symp in respiratory_symptoms):
            red_flags.append("Difficulté respiratoire")
        
        # Douleur thoracique
        if "douleur thoracique" in patient.symptoms:
            red_flags.append("Douleur thoracique")
        
        # Perte de conscience
        consciousness_symptoms = ["perte de conscience", "évanouissement", "syncope"]
        if any(symp in patient.symptoms for symp in consciousness_symptoms):
            red_flags.append("Trouble de la conscience")
        
        # Saignement important
        bleeding_symptoms = ["hémorragie", "saignement", "sang"]
        if any(symp in patient.symptoms for symp in bleeding_symptoms):
            if patient.intensity == "severe":
                red_flags.append("Saignement important")
        
        return red_flags
    
    def _determine_triage_level(self, red_flags: List[str], patient: PatientInput) -> str:
        """Détermine le niveau de triage"""
        
        # Critique: drapeaux rouges majeurs
        critical_flags = [
            "Hyperthermie",
            "Difficulté respiratoire",
            "Douleur thoracique",
            "Trouble de la conscience",
            "Hémorragie",
            "AVC",
            "Infarctus"
        ]
        
        if any(flag in " ".join(red_flags) for flag in critical_flags):
            return "critique"
        
        # Urgent: symptômes sérieux
        urgent_symptoms = ["dyspnea", "douleur thoracique", "paralysie", "confusion"]
        if any(symp in patient.symptoms for symp in urgent_symptoms):
            return "urgent"
        
        # Urgent: fièvre élevée avec symptômes multiples
        if patient.temperature and patient.temperature >= 39.5 and len(patient.symptoms) >= 3:
            return "urgent"
        
        # Normal: symptômes modérés
        if len(patient.symptoms) >= 2 or patient.intensity in ["severe", "moderate"]:
            return "normal"
        
        # Léger: symptômes mineurs
        return "léger"
    
    def evaluate(self, patient: PatientInput) -> TriageResult:
        """
        Évalue l'urgence du cas patient
        
        Args:
            patient: Informations du patient
            
        Returns:
            Résultat du triage avec niveau d'urgence et recommandations
        """
        # Collecter tous les drapeaux rouges
        red_flags = []
        red_flags.extend(self._check_pattern_red_flags(patient.raw_text))
        red_flags.extend(self._check_symptom_red_flags(patient))
        
        # Dédupliquer
        red_flags = list(set(red_flags))
        
        # Déterminer le niveau de triage
        level = self._determine_triage_level(red_flags, patient)
        
        # Obtenir les infos du niveau
        level_info = settings.TRIAGE_LEVELS.get(level, settings.TRIAGE_LEVELS["normal"])
        
        # Construire la raison
        reason_parts = []
        if red_flags:
            reason_parts.append(f"Drapeaux rouges détectés: {', '.join(red_flags)}")
        if patient.temperature and patient.temperature >= settings.TEMP_FEVER_THRESHOLD:
            reason_parts.append(f"Fièvre à {patient.temperature}°C")
        if patient.intensity == "severe":
            reason_parts.append("Symptômes sévères")
        
        reason = " | ".join(reason_parts) if reason_parts else "Évaluation standard"
        
        return TriageResult(
            level=level,
            priority=level_info["priority"],
            reason=reason,
            red_flags=red_flags,
            recommended_action=level_info["action"]
        )
