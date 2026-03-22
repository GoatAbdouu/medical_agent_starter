"""
Générateur de questions de suivi intelligentes pour désambiguïsation
"""
from typing import List, Set, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

from medical_agent.models.data_models import PatientInput, DiagnosisCandidate, Disease


class FollowUpQuestionGenerator:
    """
    Génère des questions de suivi ciblées pour différencier entre
    des diagnostics candidats similaires
    """
    
    def __init__(self, dataset: Optional[pd.DataFrame] = None):
        """
        Args:
            dataset: DataFrame avec colonnes 'disease' et 'symptom'
        """
        self.dataset = dataset
    
    def get_disease_symptoms(self, disease_name: str) -> Set[str]:
        """Récupère tous les symptômes associés à une maladie"""
        if self.dataset is None:
            return set()
        
        disease_data = self.dataset[
            self.dataset['disease'].str.lower() == disease_name.lower()
        ]
        return set(disease_data['symptom'].unique())
    
    def find_discriminating_symptoms(
        self,
        candidates: List[DiagnosisCandidate],
        patient_symptoms: List[str],
        top_n: int = 3
    ) -> List[Dict[str, any]]:
        """
        Trouve les symptômes qui permettent de différencier entre les candidats
        
        Args:
            candidates: Liste des diagnostics candidats
            patient_symptoms: Symptômes déjà mentionnés par le patient
            top_n: Nombre de candidats à considérer
            
        Returns:
            Liste de dictionnaires avec:
            - 'symptom': nom du symptôme
            - 'diseases': liste des maladies associées
            - 'discriminative_power': score de pouvoir discriminant (0-1)
        """
        if not candidates or self.dataset is None:
            return []
        
        # Prendre seulement les top N candidats
        top_candidates = candidates[:min(top_n, len(candidates))]
        patient_symptoms_lower = [s.lower() for s in patient_symptoms]
        
        # Récupérer tous les symptômes de chaque candidat
        candidate_symptoms = {}
        for candidate in top_candidates:
            symptoms = self.get_disease_symptoms(candidate.disease_name)
            candidate_symptoms[candidate.disease_name] = symptoms
        
        # Trouver les symptômes discriminants
        discriminating = {}
        
        for disease, symptoms in candidate_symptoms.items():
            for symptom in symptoms:
                # Ignorer les symptômes déjà mentionnés
                if symptom.lower() in patient_symptoms_lower:
                    continue
                
                # Compter dans combien de maladies candidates ce symptôme apparaît
                disease_count = sum(
                    1 for d_symptoms in candidate_symptoms.values()
                    if symptom in d_symptoms
                )
                
                # Un bon symptôme discriminant apparaît dans certaines maladies mais pas toutes
                # Score: 1 = apparaît dans une seule maladie (très discriminant)
                # Score: 0 = apparaît dans toutes les maladies (pas discriminant)
                total_diseases = len(top_candidates)
                discriminative_power = 1 - (disease_count - 1) / max(total_diseases - 1, 1)
                
                if symptom not in discriminating:
                    discriminating[symptom] = {
                        'symptom': symptom,
                        'diseases': [],
                        'discriminative_power': discriminative_power,
                        'appears_in_count': disease_count
                    }
                
                discriminating[symptom]['diseases'].append(disease)
        
        # Trier par pouvoir discriminant (favoriser les symptômes spécifiques)
        sorted_discriminating = sorted(
            discriminating.values(),
            key=lambda x: (x['discriminative_power'], -x['appears_in_count']),
            reverse=True
        )
        
        return sorted_discriminating
    
    def generate_symptom_questions(
        self,
        discriminating_symptoms: List[Dict[str, any]],
        max_questions: int = 5
    ) -> List[str]:
        """
        Génère des questions à partir des symptômes discriminants
        
        Args:
            discriminating_symptoms: Liste de symptômes discriminants
            max_questions: Nombre maximum de questions à générer
            
        Returns:
            Liste de questions
        """
        questions = []
        
        for symptom_info in discriminating_symptoms[:max_questions]:
            symptom = symptom_info['symptom']
            
            # Formater la question selon le symptôme
            question = self._format_symptom_question(symptom)
            questions.append(question)
        
        return questions
    
    def _format_symptom_question(self, symptom: str) -> str:
        """
        Formate une question pour un symptôme spécifique
        
        Args:
            symptom: Nom du symptôme
            
        Returns:
            Question formatée
        """
        symptom_lower = symptom.lower()
        
        # Questions oui/non
        yes_no_patterns = [
            'douleur', 'fièvre', 'toux', 'nausée', 'vomissement',
            'diarrhée', 'constipation', 'éruption', 'gonflement',
            'saignement', 'démangeaison', 'brûlure', 'crampe'
        ]
        
        for pattern in yes_no_patterns:
            if pattern in symptom_lower:
                return f"Avez-vous {symptom} ?"
        
        # Questions avec "ressentez-vous"
        feeling_patterns = ['douleur', 'sensation', 'picotement', 'engourdissement']
        for pattern in feeling_patterns:
            if pattern in symptom_lower:
                return f"Ressentez-vous {symptom} ?"
        
        # Questions avec "observez-vous"
        observation_patterns = ['rougeur', 'gonflement', 'écoulement', 'éruption']
        for pattern in observation_patterns:
            if pattern in symptom_lower:
                return f"Observez-vous {symptom} ?"
        
        # Question par défaut
        return f"Présentez-vous {symptom} ?"
    
    def generate_contextual_questions(
        self,
        candidates: List[DiagnosisCandidate],
        patient: PatientInput
    ) -> List[str]:
        """
        Génère des questions contextuelles basées sur les candidats
        
        Args:
            candidates: Diagnostics candidats
            patient: Informations patient actuelles
            
        Returns:
            Liste de questions contextuelles
        """
        questions = []
        
        # Questions sur l'évolution temporelle
        if not patient.onset:
            questions.append("Depuis combien de temps avez-vous ces symptômes ?")
        
        # Questions sur l'intensité
        if not patient.intensity:
            questions.append("Comment évaluez-vous l'intensité de vos symptômes ? (léger/modéré/sévère)")
        
        # Questions sur la température si fièvre mentionnée
        if patient.has_fever and not patient.temperature:
            questions.append("Avez-vous mesuré votre température ? Si oui, quelle est-elle ?")
        elif any('fièvre' in s.lower() or 'fever' in s.lower() for s in patient.symptoms):
            questions.append("Avez-vous de la fièvre ? Avez-vous mesuré votre température ?")
        
        # Questions sur les facteurs aggravants/atténuants
        questions.append("Y a-t-il quelque chose qui aggrave ou soulage vos symptômes ?")
        
        # Questions sur les antécédents
        if not patient.risk_factors:
            questions.append("Avez-vous des antécédents médicaux ou prenez-vous des médicaments ?")
        
        return questions
    
    def generate_all_follow_up_questions(
        self,
        patient: PatientInput,
        candidates: List[DiagnosisCandidate],
        max_questions: int = 5,
        prioritize_discriminating: bool = True
    ) -> List[Dict[str, any]]:
        """
        Génère toutes les questions de suivi nécessaires
        
        Args:
            patient: Informations patient
            candidates: Diagnostics candidats
            max_questions: Nombre maximum de questions à retourner
            prioritize_discriminating: Prioriser les questions discriminantes
            
        Returns:
            Liste de dictionnaires avec:
            - 'question': texte de la question
            - 'type': 'discriminating' ou 'contextual'
            - 'priority': score de priorité
            - 'related_diseases': maladies liées (pour questions discriminantes)
        """
        all_questions = []
        
        # 1. Questions discriminantes (basées sur les symptômes)
        discriminating_symptoms = self.find_discriminating_symptoms(
            candidates,
            patient.symptoms,
            top_n=min(3, len(candidates))
        )
        
        for symptom_info in discriminating_symptoms[:max_questions]:
            question_text = self._format_symptom_question(symptom_info['symptom'])
            all_questions.append({
                'question': question_text,
                'type': 'discriminating',
                'priority': symptom_info['discriminative_power'],
                'related_diseases': symptom_info['diseases'],
                'symptom': symptom_info['symptom']
            })
        
        # 2. Questions contextuelles
        contextual = self.generate_contextual_questions(candidates, patient)
        for i, question_text in enumerate(contextual):
            # Priorité plus faible pour les questions contextuelles
            priority = 0.5 - (i * 0.1)
            all_questions.append({
                'question': question_text,
                'type': 'contextual',
                'priority': priority,
                'related_diseases': [],
                'symptom': None
            })
        
        # Trier par priorité
        if prioritize_discriminating:
            all_questions.sort(key=lambda x: x['priority'], reverse=True)
        
        # Limiter au nombre max
        return all_questions[:max_questions]
    
    def generate_questions_for_disambiguation(
        self,
        patient: PatientInput,
        candidates: List[DiagnosisCandidate],
        max_questions: int = 3
    ) -> List[str]:
        """
        Génère des questions optimisées pour la désambiguïsation
        (version simplifiée pour l'API principale)
        
        Args:
            patient: Informations patient
            candidates: Diagnostics candidats
            max_questions: Nombre maximum de questions
            
        Returns:
            Liste de questions (strings)
        """
        detailed_questions = self.generate_all_follow_up_questions(
            patient,
            candidates,
            max_questions=max_questions * 2,  # Générer plus pour filtrer
            prioritize_discriminating=True
        )
        
        # Extraire seulement le texte des questions
        questions = [q['question'] for q in detailed_questions[:max_questions]]
        
        return questions
