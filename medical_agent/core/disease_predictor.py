"""
Prédicteur de maladies - Combine ML et règles métier
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import joblib

from medical_agent.models.data_models import PatientInput, DiagnosisCandidate, Disease
from medical_agent.config.settings import settings
from medical_agent.core.disambiguation import DisambiguationDetector


class DiseasePredictor:
    """Prédit les maladies possibles à partir des symptômes"""
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        model_path: Optional[Path] = None
    ):
        """
        Args:
            dataset_path: Chemin vers le dataset CSV
            model_path: Chemin vers le modèle ML pré-entraîné
        """
        self.dataset_path = dataset_path or settings.DATASET_PATH
        self.model_path = model_path or settings.DISEASE_MODEL_PATH
        
        # Charger les données
        self.df = self._load_dataset()
        
        # Charger le modèle ML si disponible
        self.ml_model = None
        self.vectorizer = None
        if self.model_path.exists():
            self._load_ml_model()
        
        # Initialiser le détecteur de désambiguïsation
        self.disambiguation_detector = DisambiguationDetector()
    
    def _load_dataset(self) -> pd.DataFrame:
        """Charge et nettoie le dataset"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        
        # Nettoyer les colonnes
        if len(df.columns) >= 2:
            df.columns = ['disease', 'symptom'] + list(df.columns[2:])
        
        # Normaliser
        df['disease'] = df['disease'].astype(str).str.lower().str.strip()
        df['symptom'] = df['symptom'].astype(str).str.lower().str.strip()
        
        # Supprimer les valeurs invalides
        df = df.dropna(subset=['disease', 'symptom'])
        df = df[df['disease'] != 'nan']
        df = df[df['symptom'] != 'nan']
        df = df[df['disease'] != '']
        df = df[df['symptom'] != '']
        
        return df
    
    def _load_ml_model(self):
        """Charge le modèle ML pré-entraîné"""
        try:
            artifact = joblib.load(self.model_path)
            self.ml_model = artifact.get("model")
            self.vectorizer = artifact.get("vectorizer")
        except Exception as e:
            print(f"Erreur chargement modèle ML: {e}")
            self.ml_model = None
            self.vectorizer = None
    
    def _predict_with_rules(self, symptoms: List[str], top_n: int = 5) -> List[DiagnosisCandidate]:
        """Prédiction basée sur les règles (matching de symptômes)"""
        if not symptoms:
            return []
        
        # Compter les correspondances par maladie
        disease_scores = {}
        
        for disease in self.df['disease'].unique():
            disease_symptoms = self.df[self.df['disease'] == disease]['symptom'].tolist()
            
            # Calculer le score
            matches = 0
            matched_symptoms = []
            
            for user_symptom in symptoms:
                for db_symptom in disease_symptoms:
                    if user_symptom in db_symptom or db_symptom in user_symptom:
                        matches += 1
                        matched_symptoms.append(user_symptom)
                        break
            
            if matches > 0:
                # Score = (symptômes correspondants / total symptômes utilisateur)
                confidence = matches / len(symptoms)
                disease_scores[disease] = {
                    'confidence': confidence,
                    'matches': matches,
                    'matched_symptoms': list(set(matched_symptoms))
                }
        
        # Trier par score
        sorted_diseases = sorted(
            disease_scores.items(),
            key=lambda x: (x[1]['confidence'], x[1]['matches']),
            reverse=True
        )
        
        # Créer les candidats
        candidates = []
        for disease_name, info in sorted_diseases[:top_n]:
            candidates.append(DiagnosisCandidate(
                disease_name=disease_name,
                confidence=info['confidence'],
                matched_symptoms=info['matched_symptoms']
            ))
        
        return candidates
    
    def _predict_with_ml(self, symptoms: List[str], top_n: int = 5) -> List[DiagnosisCandidate]:
        """Prédiction avec le modèle ML"""
        if not self.ml_model or not self.vectorizer:
            return []
        
        try:
            # Transformer les symptômes en texte
            symptoms_text = " ".join(symptoms)
            
            # Vectoriser
            X = self.vectorizer.transform([symptoms_text])
            
            # Prédire les probabilités
            probas = self.ml_model.predict_proba(X)[0]
            
            # Obtenir les top N
            top_indices = probas.argsort()[-top_n:][::-1]
            
            candidates = []
            for idx in top_indices:
                disease_name = self.ml_model.classes_[idx]
                confidence = float(probas[idx])
                
                if confidence >= settings.CONFIDENCE_THRESHOLD:
                    candidates.append(DiagnosisCandidate(
                        disease_name=disease_name,
                        confidence=confidence,
                        matched_symptoms=symptoms
                    ))
            
            return candidates
        
        except Exception as e:
            print(f"Erreur prédiction ML: {e}")
            return []
    
    def _combine_predictions(
        self,
        rule_candidates: List[DiagnosisCandidate],
        ml_candidates: List[DiagnosisCandidate],
        top_n: int = 5
    ) -> List[DiagnosisCandidate]:
        """Combine les prédictions des règles et du ML"""
        
        # Si pas de ML, retourner les règles
        if not ml_candidates:
            return rule_candidates[:top_n]
        
        # Si pas de règles, retourner le ML
        if not rule_candidates:
            return ml_candidates[:top_n]
        
        # Combiner les scores
        combined_scores = {}
        
        # Ajouter les scores des règles
        for candidate in rule_candidates:
            combined_scores[candidate.disease_name] = {
                'rule_confidence': candidate.confidence,
                'ml_confidence': 0.0,
                'matched_symptoms': candidate.matched_symptoms
            }
        
        # Ajouter les scores ML
        for candidate in ml_candidates:
            if candidate.disease_name in combined_scores:
                combined_scores[candidate.disease_name]['ml_confidence'] = candidate.confidence
            else:
                combined_scores[candidate.disease_name] = {
                    'rule_confidence': 0.0,
                    'ml_confidence': candidate.confidence,
                    'matched_symptoms': candidate.matched_symptoms
                }
        
        # Calculer le score combiné (moyenne pondérée)
        final_candidates = []
        for disease_name, scores in combined_scores.items():
            # 60% ML, 40% règles si les deux disponibles
            if scores['ml_confidence'] > 0 and scores['rule_confidence'] > 0:
                combined_confidence = 0.6 * scores['ml_confidence'] + 0.4 * scores['rule_confidence']
            else:
                combined_confidence = max(scores['ml_confidence'], scores['rule_confidence'])
            
            final_candidates.append(DiagnosisCandidate(
                disease_name=disease_name,
                confidence=combined_confidence,
                matched_symptoms=scores['matched_symptoms']
            ))
        
        # Trier par confiance
        final_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return final_candidates[:top_n]
    
    def predict(self, patient: PatientInput, top_n: int = 5) -> List[DiagnosisCandidate]:
        """
        Prédit les maladies possibles
        
        Args:
            patient: Informations du patient
            top_n: Nombre de diagnostics à retourner
            
        Returns:
            Liste des candidats de diagnostic triés par confiance
        """
        if not patient.symptoms:
            return []
        
        # Prédictions avec règles
        rule_candidates = self._predict_with_rules(patient.symptoms, top_n)
        
        # Prédictions avec ML
        ml_candidates = self._predict_with_ml(patient.symptoms, top_n)
        
        # Combiner les prédictions
        final_candidates = self._combine_predictions(rule_candidates, ml_candidates, top_n)
        
        # Ajuster les confiances selon la spécificité des symptômes
        specificity_score = self.disambiguation_detector.calculate_symptom_specificity(
            patient.symptoms
        )
        adjusted_candidates = self.disambiguation_detector.adjust_confidence_for_genericity(
            final_candidates,
            specificity_score
        )
        
        return adjusted_candidates
    
    def get_disease_info(self, disease_name: str) -> Optional[Disease]:
        """Récupère les informations d'une maladie"""
        disease_data = self.df[self.df['disease'] == disease_name.lower()]
        
        if disease_data.empty:
            return None
        
        symptoms = disease_data['symptom'].unique().tolist()
        
        return Disease(
            name=disease_name,
            symptoms=symptoms
        )
