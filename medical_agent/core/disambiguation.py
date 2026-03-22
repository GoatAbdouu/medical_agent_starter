"""
Module de désambiguïsation - Détecte les symptômes génériques et les prédictions ambiguës
"""
from typing import List, Set, Dict, Optional
from medical_agent.models.data_models import PatientInput, DiagnosisCandidate, Disease


# Symptômes génériques très communs à de nombreuses maladies
GENERIC_SYMPTOMS = {
    'fièvre', 'fever', 'température', 'température élevée',
    'fatigue', 'fatigue excessive', 'épuisement', 'tired', 'tiredness',
    'mal de tête', 'céphalée', 'céphalées', 'headache', 'maux de tête',
    'douleur', 'douleurs', 'pain', 'ache',
    'nausée', 'nausées', 'nausea',
    'vomissement', 'vomissements', 'vomiting',
    'toux', 'cough', 'toussing',
    'étourdissement', 'étourdissements', 'vertiges', 'dizziness', 'dizzy',
    'faiblesse', 'weakness',
    'perte d\'appétit', 'loss of appetite',
    'malaise', 'malaise général',
    'transpiration', 'sweating', 'sueurs',
    'frissons', 'chills',
    'douleur musculaire', 'courbatures', 'muscle pain', 'myalgie',
    'essoufflement', 'shortness of breath', 'dyspnée',
    'insomnie', 'trouble du sommeil', 'insomnia',
    'anxiété', 'anxiety', 'stress',
    'dépression', 'depression',
}


class DisambiguationDetector:
    """
    Détecte quand un diagnostic nécessite une désambiguïsation
    en raison de symptômes trop génériques ou de prédictions ambiguës
    """
    
    def __init__(
        self,
        generic_symptoms: Optional[Set[str]] = None,
        max_confidence_for_generic: float = 0.50,
        ambiguity_threshold: float = 0.15
    ):
        """
        Args:
            generic_symptoms: Ensemble de symptômes considérés comme génériques
            max_confidence_for_generic: Confiance max acceptable avec symptômes génériques
            ambiguity_threshold: Écart minimal entre top 2 diagnostics pour être non-ambigu
        """
        self.generic_symptoms = generic_symptoms or GENERIC_SYMPTOMS
        self.max_confidence_for_generic = max_confidence_for_generic
        self.ambiguity_threshold = ambiguity_threshold
    
    def is_generic_symptom(self, symptom: str) -> bool:
        """Vérifie si un symptôme est considéré comme générique"""
        symptom_lower = symptom.lower().strip()
        
        # Vérification exacte
        if symptom_lower in self.generic_symptoms:
            return True
        
        # Vérification par sous-chaîne (pour attraper les variations)
        for generic in self.generic_symptoms:
            if generic in symptom_lower or symptom_lower in generic:
                return True
        
        return False
    
    def calculate_symptom_specificity(self, symptoms: List[str]) -> float:
        """
        Calcule la spécificité des symptômes (0 = tous génériques, 1 = tous spécifiques)
        
        Returns:
            Score de spécificité entre 0 et 1
        """
        if not symptoms:
            return 0.0
        
        specific_count = sum(1 for s in symptoms if not self.is_generic_symptom(s))
        return specific_count / len(symptoms)
    
    def has_only_generic_symptoms(self, patient: PatientInput) -> bool:
        """Vérifie si le patient n'a que des symptômes génériques"""
        if not patient.symptoms:
            return True
        
        return all(self.is_generic_symptom(s) for s in patient.symptoms)
    
    def get_generic_symptoms_from_input(self, patient: PatientInput) -> List[str]:
        """Retourne la liste des symptômes génériques du patient"""
        return [s for s in patient.symptoms if self.is_generic_symptom(s)]
    
    def get_specific_symptoms_from_input(self, patient: PatientInput) -> List[str]:
        """Retourne la liste des symptômes spécifiques du patient"""
        return [s for s in patient.symptoms if not self.is_generic_symptom(s)]
    
    def is_prediction_ambiguous(
        self,
        candidates: List[DiagnosisCandidate],
        min_candidates_for_ambiguity: int = 2
    ) -> bool:
        """
        Détermine si les prédictions sont ambiguës
        
        Critères d'ambiguïté:
        - Plusieurs candidats avec des confiances similaires
        - Pas de diagnostic clairement dominant
        
        Returns:
            True si les prédictions nécessitent une désambiguïsation
        """
        if len(candidates) < min_candidates_for_ambiguity:
            return False
        
        # Vérifier l'écart entre le top 1 et le top 2
        top1_confidence = candidates[0].confidence
        top2_confidence = candidates[1].confidence
        
        confidence_gap = top1_confidence - top2_confidence
        
        # Si l'écart est trop faible, c'est ambigu
        return confidence_gap < self.ambiguity_threshold
    
    def needs_disambiguation(
        self,
        patient: PatientInput,
        candidates: List[DiagnosisCandidate]
    ) -> Dict[str, any]:
        """
        Détermine si une désambiguïsation est nécessaire
        
        Returns:
            Dictionnaire avec:
            - 'needs_disambiguation': bool
            - 'reason': str (raison de la désambiguïsation)
            - 'specificity_score': float
            - 'is_ambiguous': bool
            - 'has_generic_only': bool
            - 'top_confidence': float
        """
        if not candidates:
            return {
                'needs_disambiguation': True,
                'reason': 'Aucun diagnostic candidat trouvé',
                'specificity_score': 0.0,
                'is_ambiguous': False,
                'has_generic_only': True,
                'top_confidence': 0.0
            }
        
        specificity_score = self.calculate_symptom_specificity(patient.symptoms)
        has_generic_only = self.has_only_generic_symptoms(patient)
        is_ambiguous = self.is_prediction_ambiguous(candidates)
        top_confidence = candidates[0].confidence if candidates else 0.0
        
        # Cas 1: Symptômes uniquement génériques
        if has_generic_only:
            return {
                'needs_disambiguation': True,
                'reason': 'Symptômes trop génériques - besoin de plus d\'informations spécifiques',
                'specificity_score': specificity_score,
                'is_ambiguous': is_ambiguous,
                'has_generic_only': True,
                'top_confidence': top_confidence
            }
        
        # Cas 2: Faible spécificité avec confiance élevée (suspect)
        if specificity_score < 0.3 and top_confidence > self.max_confidence_for_generic:
            return {
                'needs_disambiguation': True,
                'reason': f'Confiance trop élevée ({top_confidence:.0%}) pour des symptômes génériques',
                'specificity_score': specificity_score,
                'is_ambiguous': is_ambiguous,
                'has_generic_only': False,
                'top_confidence': top_confidence
            }
        
        # Cas 3: Prédictions ambiguës (scores similaires)
        if is_ambiguous and top_confidence < 0.8:
            gap = candidates[0].confidence - candidates[1].confidence
            return {
                'needs_disambiguation': True,
                'reason': f'Diagnostics multiples possibles (écart de {gap:.0%} seulement)',
                'specificity_score': specificity_score,
                'is_ambiguous': True,
                'has_generic_only': has_generic_only,
                'top_confidence': top_confidence
            }
        
        # Cas 4: Peu de symptômes spécifiques
        if len(patient.symptoms) <= 2:
            return {
                'needs_disambiguation': True,
                'reason': 'Informations insuffisantes pour un diagnostic précis',
                'specificity_score': specificity_score,
                'is_ambiguous': is_ambiguous,
                'has_generic_only': has_generic_only,
                'top_confidence': top_confidence
            }
        
        # Pas besoin de désambiguïsation
        return {
            'needs_disambiguation': False,
            'reason': 'Symptômes suffisamment spécifiques et diagnostic clair',
            'specificity_score': specificity_score,
            'is_ambiguous': is_ambiguous,
            'has_generic_only': has_generic_only,
            'top_confidence': top_confidence
        }
    
    def adjust_confidence_for_genericity(
        self,
        candidates: List[DiagnosisCandidate],
        specificity_score: float
    ) -> List[DiagnosisCandidate]:
        """
        Ajuste les scores de confiance en fonction de la spécificité des symptômes
        
        Plus les symptômes sont génériques, plus on pénalise la confiance
        
        Args:
            candidates: Liste des candidats de diagnostic
            specificity_score: Score de spécificité (0 = génériques, 1 = spécifiques)
            
        Returns:
            Liste des candidats avec confiances ajustées
        """
        adjusted_candidates = []
        
        for candidate in candidates:
            # Facteur de pénalité basé sur la spécificité
            # Si spécificité = 0 (que des génériques), pénalité maximale
            # Si spécificité = 1 (que des spécifiques), pas de pénalité
            penalty_factor = 0.3 + (0.7 * specificity_score)
            
            adjusted_confidence = candidate.confidence * penalty_factor
            
            # S'assurer que la confiance ne dépasse jamais le seuil pour les symptômes génériques
            if specificity_score < 0.5:
                adjusted_confidence = min(adjusted_confidence, self.max_confidence_for_generic)
            
            adjusted_candidates.append(DiagnosisCandidate(
                disease_name=candidate.disease_name,
                confidence=adjusted_confidence,
                matched_symptoms=candidate.matched_symptoms,
                specialty=candidate.specialty
            ))
        
        return adjusted_candidates
