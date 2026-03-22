"""
Modèles de données pour l'agent médical
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class PatientInput:
    """Entrée du patient"""
    raw_text: str
    symptoms: List[str] = field(default_factory=list)
    onset: Optional[str] = None
    intensity: Optional[str] = None
    measured_values: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def temperature(self) -> Optional[float]:
        """Retourne la température si disponible"""
        return self.measured_values.get("temperature_c")
    
    @property
    def has_fever(self) -> bool:
        """Vérifie si le patient a de la fièvre"""
        temp = self.temperature
        return temp is not None and temp >= 38.0


@dataclass
class Disease:
    """Représentation d'une maladie"""
    name: str
    symptoms: List[str] = field(default_factory=list)
    specialty: str = "general"
    severity: str = "normal"
    description: Optional[str] = None


@dataclass
class DiagnosisCandidate:
    """Candidat de diagnostic"""
    disease_name: str
    confidence: float
    matched_symptoms: List[str] = field(default_factory=list)
    specialty: str = "general"
    
    def __post_init__(self):
        """Valide les données après initialisation"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("La confiance doit être entre 0 et 1")


@dataclass
class TriageResult:
    """Résultat du triage médical"""
    level: str  # critique, urgent, normal, léger
    priority: int
    reason: str
    red_flags: List[str] = field(default_factory=list)
    recommended_action: str = ""
    
    @property
    def color(self) -> str:
        """Retourne la couleur associée au niveau"""
        colors = {
            "critique": "red",
            "urgent": "orange",
            "normal": "yellow",
            "léger": "green"
        }
        return colors.get(self.level, "gray")


@dataclass
class DiagnosisResult:
    """Résultat complet du diagnostic"""
    patient_input: PatientInput
    candidates: List[DiagnosisCandidate] = field(default_factory=list)
    triage: Optional[TriageResult] = None
    specialty: str = "general"
    questions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Nouveaux champs pour la désambiguïsation
    needs_disambiguation: bool = False
    disambiguation_reason: str = ""
    symptom_specificity_score: float = 1.0
    conversation_turn: int = 1
    
    @property
    def top_diagnosis(self) -> Optional[DiagnosisCandidate]:
        """Retourne le diagnostic le plus probable"""
        return self.candidates[0] if self.candidates else None
    
    @property
    def is_urgent(self) -> bool:
        """Vérifie si le cas est urgent"""
        return self.triage is not None and self.triage.level in ["critique", "urgent"]
    
    @property
    def is_conclusive(self) -> bool:
        """Vérifie si le diagnostic est suffisamment conclusif"""
        return not self.needs_disambiguation and self.top_diagnosis is not None


@dataclass
class ConversationState:
    """État de la conversation pour le suivi multi-tours"""
    conversation_id: str
    patient_input: PatientInput
    all_symptoms: List[str] = field(default_factory=list)
    diagnosis_history: List[DiagnosisResult] = field(default_factory=list)
    asked_questions: List[str] = field(default_factory=list)
    current_turn: int = 1
    max_turns: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_symptoms(self, new_symptoms: List[str]):
        """Ajoute de nouveaux symptômes à la liste"""
        for symptom in new_symptoms:
            if symptom not in self.all_symptoms:
                self.all_symptoms.append(symptom)
        self.patient_input.symptoms = self.all_symptoms
        self.last_updated = datetime.now()
    
    def add_diagnosis_result(self, result: DiagnosisResult):
        """Ajoute un résultat de diagnostic à l'historique"""
        self.diagnosis_history.append(result)
        self.current_turn += 1
        self.last_updated = datetime.now()
    
    def add_asked_question(self, question: str):
        """Marque une question comme posée"""
        if question not in self.asked_questions:
            self.asked_questions.append(question)
    
    def should_continue(self) -> bool:
        """Vérifie si la conversation devrait continuer"""
        return self.current_turn <= self.max_turns
    
    @property
    def latest_diagnosis(self) -> Optional[DiagnosisResult]:
        """Retourne le dernier résultat de diagnostic"""
        return self.diagnosis_history[-1] if self.diagnosis_history else None
