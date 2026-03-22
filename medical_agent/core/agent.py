"""
Agent Médical Principal - Orchestre tous les composants
"""
from typing import Optional, List
from pathlib import Path

from medical_agent.models.data_models import PatientInput, DiagnosisResult, ConversationState
from medical_agent.core.symptom_extractor import SymptomExtractor
from medical_agent.core.disease_predictor import DiseasePredictor
from medical_agent.core.triage_system import TriageSystem
from medical_agent.core.disambiguation import DisambiguationDetector
from medical_agent.core.question_generator import FollowUpQuestionGenerator
from medical_agent.config.settings import settings

# Import du prédicteur Deep Learning (optionnel)
try:
    from medical_agent.core.deep_learning_predictor import DeepLearningPredictor
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False


class MedicalAgent:
    """
    Agent médical principal qui coordonne:
    - Extraction de symptômes
    - Prédiction de maladies (Deep Learning ou règles)
    - Triage d'urgence
    - Génération de recommandations
    """
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        synonyms_path: Optional[Path] = None,
        red_flags_path: Optional[Path] = None,
        use_deep_learning: bool = True
    ):
        """
        Initialise l'agent médical avec tous ses composants
        
        Args:
            dataset_path: Chemin vers le dataset de maladies
            model_path: Chemin vers le modèle ML
            synonyms_path: Chemin vers les synonymes de symptômes
            red_flags_path: Chemin vers les drapeaux rouges
            use_deep_learning: Utiliser le Deep Learning si disponible
        """
        # Initialiser les composants
        self.symptom_extractor = SymptomExtractor(synonyms_path)
        
        # Choisir le prédicteur (Deep Learning ou classique)
        self.use_deep_learning = use_deep_learning and DL_AVAILABLE
        
        if self.use_deep_learning:
            try:
                self.disease_predictor = DeepLearningPredictor(dataset_path)
                print("✅ Mode Deep Learning activé")
            except Exception as e:
                print(f"⚠️ Deep Learning non disponible ({e}), utilisation du mode classique")
                self.disease_predictor = DiseasePredictor(dataset_path, model_path)
                self.use_deep_learning = False
        else:
            self.disease_predictor = DiseasePredictor(dataset_path, model_path)
            
        self.triage_system = TriageSystem(red_flags_path)
        
        # Initialiser les systèmes de désambiguïsation
        self.disambiguation_detector = DisambiguationDetector()
        self.question_generator = FollowUpQuestionGenerator(
            dataset=self.disease_predictor.df if hasattr(self.disease_predictor, 'df') else None
        )
        
        # Charger les questions de suivi
        self.questions = self._load_questions()
    
    def _load_questions(self) -> dict:
        """Charge les questions de suivi par spécialité"""
        return settings.load_yaml_config(settings.QUESTIONS_PATH)
    
    def _generate_follow_up_questions(self, patient: PatientInput, diagnosis_result: DiagnosisResult) -> List[str]:
        """Génère des questions de suivi pertinentes et discriminantes"""
        
        # Si disambiguation nécessaire, utiliser le générateur intelligent
        if diagnosis_result.needs_disambiguation:
            # Générer des questions discriminantes basées sur les candidats
            discriminating_questions = self.question_generator.generate_questions_for_disambiguation(
                patient,
                diagnosis_result.candidates,
                max_questions=3
            )
            
            # Ajouter quelques questions contextuelles si nécessaire
            contextual_questions = []
            
            if not patient.temperature:
                contextual_questions.append("Avez-vous pris votre température ?")
            
            if not patient.onset:
                contextual_questions.append("Depuis combien de temps avez-vous ces symptômes ?")
            
            if not patient.intensity:
                contextual_questions.append("Comment évaluez-vous l'intensité de vos symptômes (léger/modéré/sévère) ?")
            
            # Combiner (max 5 questions)
            all_questions = discriminating_questions + contextual_questions
            return all_questions[:5]
        
        # Sinon, questions standard basées sur les symptômes manquants
        questions = []
        
        if diagnosis_result.top_diagnosis:
            disease_info = self.disease_predictor.get_disease_info(
                diagnosis_result.top_diagnosis.disease_name
            )
            
            if disease_info:
                # Trouver les symptômes de la maladie non mentionnés
                missing_symptoms = [
                    s for s in disease_info.symptoms
                    if s not in patient.symptoms
                ]
                
                # Ajouter des questions sur ces symptômes
                if missing_symptoms[:3]:  # Max 3 questions
                    for symptom in missing_symptoms[:3]:
                        questions.append(f"Avez-vous également {symptom} ?")
        
        # Questions standard
        if not patient.temperature:
            questions.append("Avez-vous pris votre température ?")
        
        if not patient.onset:
            questions.append("Depuis combien de temps avez-vous ces symptômes ?")
        
        if not patient.intensity:
            questions.append("Comment évaluez-vous l'intensité de vos symptômes (léger/modéré/sévère) ?")
        
        return questions[:5]  # Maximum 5 questions
    
    def _generate_recommendations(self, diagnosis_result: DiagnosisResult) -> List[str]:
        """Génère des recommandations personnalisées"""
        recommendations = []
        
        # Recommandations basées sur le triage
        if diagnosis_result.triage:
            recommendations.append(f"🚨 {diagnosis_result.triage.recommended_action}")
            
            if diagnosis_result.triage.level == "critique":
                recommendations.append("⚠️ Appelez le 15 (SAMU) immédiatement")
            elif diagnosis_result.triage.level == "urgent":
                recommendations.append("⚠️ Rendez-vous aux urgences rapidement")
        
        # Recommandations basées sur la température
        patient = diagnosis_result.patient_input
        if patient.has_fever:
            recommendations.append(f"🌡️ Surveillez votre température (actuellement {patient.temperature}°C)")
            recommendations.append("💊 Prenez du paracétamol si nécessaire")
            recommendations.append("💧 Hydratez-vous bien")
        
        # Recommandations générales
        if not diagnosis_result.is_urgent:
            recommendations.append("📞 Consultez votre médecin traitant si les symptômes persistent")
            recommendations.append("📝 Notez l'évolution de vos symptômes")
        
        return recommendations
    
    def diagnose(self, text: str, top_n: int = 5) -> DiagnosisResult:
        """
        Effectue un diagnostic complet à partir du texte patient
        
        Args:
            text: Description des symptômes par le patient
            top_n: Nombre de diagnostics candidats à retourner
            
        Returns:
            Résultat complet du diagnostic avec triage et recommandations
        """
        # 1. Extraire les symptômes et informations
        patient_input = self.symptom_extractor.extract(text)
        
        # 2. Évaluer l'urgence (triage)
        triage_result = self.triage_system.evaluate(patient_input)
        
        # 3. Prédire les maladies possibles (avec ajustement de confiance intégré)
        candidates = self.disease_predictor.predict(patient_input, top_n)
        
        # 4. Détecter si une désambiguïsation est nécessaire
        disambiguation_info = self.disambiguation_detector.needs_disambiguation(
            patient_input,
            candidates
        )
        
        # 5. Créer le résultat du diagnostic
        diagnosis_result = DiagnosisResult(
            patient_input=patient_input,
            candidates=candidates,
            triage=triage_result,
            specialty="general",  # À améliorer avec un router
            needs_disambiguation=disambiguation_info['needs_disambiguation'],
            disambiguation_reason=disambiguation_info['reason'],
            symptom_specificity_score=disambiguation_info['specificity_score']
        )
        
        # 6. Générer les questions de suivi
        diagnosis_result.questions = self._generate_follow_up_questions(
            patient_input, diagnosis_result
        )
        
        # 7. Générer les recommandations
        diagnosis_result.recommendations = self._generate_recommendations(
            diagnosis_result
        )
        
        return diagnosis_result
    
    def _init_skin_classifier(self) -> None:
        """Initialise le classificateur de maladies cutanées (chargement paresseux)"""
        if hasattr(self, '_skin_classifier'):
            return
        try:
            from medical_agent.core.skin_disease_classifier import SkinDiseaseClassifier
            self._skin_classifier = SkinDiseaseClassifier()
        except Exception as e:
            print(f"⚠️ Classificateur de peau non disponible ({e})")
            self._skin_classifier = None

    def diagnose_skin_image(self, image, top_n: int = 5):
        """
        Effectue un diagnostic de maladie cutanée à partir d'une image PIL.

        Args:
            image: Image PIL
            top_n: Nombre de diagnostics candidats à retourner

        Returns:
            SkinDiagnosisResult ou lève une exception si le modèle n'est pas disponible
        """
        self._init_skin_classifier()
        if self._skin_classifier is None:
            raise RuntimeError(
                "Le classificateur de maladies cutanées n'est pas disponible. "
                "Vérifiez que les dépendances sont installées et que le modèle existe."
            )
        return self._skin_classifier.predict(image, top_n=top_n)

    def get_disease_details(self, disease_name: str):
        """Récupère les détails d'une maladie spécifique"""
        return self.disease_predictor.get_disease_info(disease_name)
    
    def continue_conversation(
        self,
        conversation_state: ConversationState,
        new_text: str
    ) -> DiagnosisResult:
        """
        Continue une conversation multi-tours en ajoutant de nouvelles informations
        
        Args:
            conversation_state: État actuel de la conversation
            new_text: Nouvelles informations du patient (réponses aux questions)
            
        Returns:
            Résultat de diagnostic mis à jour
        """
        # Extraire les nouveaux symptômes de la réponse
        new_patient_input = self.symptom_extractor.extract(new_text)
        
        # Mettre à jour l'état de la conversation
        conversation_state.add_symptoms(new_patient_input.symptoms)
        
        # Mettre à jour les valeurs mesurées si présentes
        if new_patient_input.temperature:
            conversation_state.patient_input.measured_values['temperature_c'] = new_patient_input.temperature
        
        if new_patient_input.onset:
            conversation_state.patient_input.onset = new_patient_input.onset
        
        if new_patient_input.intensity:
            conversation_state.patient_input.intensity = new_patient_input.intensity
        
        # Effectuer un nouveau diagnostic avec les informations cumulées
        combined_text = conversation_state.patient_input.raw_text + " " + new_text
        diagnosis_result = self.diagnose(combined_text, top_n=5)
        
        # Mettre à jour le tour de conversation
        diagnosis_result.conversation_turn = conversation_state.current_turn + 1
        
        # Ajouter à l'historique
        conversation_state.add_diagnosis_result(diagnosis_result)
        
        return diagnosis_result
