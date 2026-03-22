"""
Deep Learning Predictor - Réseau de neurones pour la prédiction de maladies
Remplace la régression logistique par un MLP (Multi-Layer Perceptron)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import joblib
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from medical_agent.models.data_models import PatientInput, DiagnosisCandidate
from medical_agent.config.settings import settings


class SymptomVectorizer:
    """Convertit les symptômes en vecteurs pour le réseau de neurones"""
    
    def __init__(self):
        self.symptom_to_idx: Dict[str, int] = {}
        self.idx_to_symptom: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        # Mapping des termes courants vers les termes du dataset
        self.symptom_aliases = {
            "fievre": "fièvre",
            "fièvre": "fièvre",
            "temperature": "fièvre",
            "gorge": "mal de gorge",
            "mal à la gorge": "mal de gorge",
            "mal a la gorge": "mal de gorge",
            "toux": "toux",
            "tousse": "toux",
            "fatigue": "fatigue",
            "fatigué": "fatigue",
            "mal de tête": "céphalées",
            "mal à la tête": "céphalées",
            "mal a la tete": "céphalées",
            "maux de tête": "céphalées",
            "nausée": "nausée",
            "nausee": "nausée",
            "vomissement": "vomissements",
            "essoufflement": "essoufflement",
            "diarrhée": "diarrhée",
            "douleur abdominale": "douleur abdominale",
            "mal au ventre": "douleur abdominale",
            "frissons": "frissons",
            "courbatures": "douleurs musculaires",
            "vertiges": "étourdissements",
            "étourdissements": "étourdissements",
        }
        
    def fit(self, symptoms_list: List[List[str]]):
        """Crée le vocabulaire à partir de la liste des symptômes"""
        all_symptoms = set()
        for symptoms in symptoms_list:
            all_symptoms.update(symptoms)
        
        self.symptom_to_idx = {s: i for i, s in enumerate(sorted(all_symptoms))}
        self.idx_to_symptom = {i: s for s, i in self.symptom_to_idx.items()}
        self.vocab_size = len(self.symptom_to_idx)
        
    def transform(self, symptoms: List[str]) -> np.ndarray:
        """Convertit une liste de symptômes en vecteur binaire avec matching intelligent"""
        vector = np.zeros(self.vocab_size, dtype=np.float32)
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()
            
            # 1. Essayer le match exact
            if symptom_lower in self.symptom_to_idx:
                vector[self.symptom_to_idx[symptom_lower]] = 1.0
                continue
            
            # 2. Essayer via les alias
            if symptom_lower in self.symptom_aliases:
                alias = self.symptom_aliases[symptom_lower]
                if alias in self.symptom_to_idx:
                    vector[self.symptom_to_idx[alias]] = 1.0
                    continue
            
            # 3. Matching partiel (le symptom est contenu dans un terme du vocab)
            for vocab_symptom in self.symptom_to_idx:
                if symptom_lower in vocab_symptom or vocab_symptom in symptom_lower:
                    vector[self.symptom_to_idx[vocab_symptom]] = 1.0
                    break
        
        return vector
    
    def save(self, path: Path):
        """Sauvegarde le vectorizer"""
        data = {
            'symptom_to_idx': self.symptom_to_idx,
            'vocab_size': self.vocab_size
        }
        joblib.dump(data, path)
        
    def load(self, path: Path):
        """Charge le vectorizer"""
        data = joblib.load(path)
        self.symptom_to_idx = data['symptom_to_idx']
        self.vocab_size = data['vocab_size']
        self.idx_to_symptom = {i: s for s, i in self.symptom_to_idx.items()}


class DiseaseClassifierMLP(nn.Module):
    """
    Réseau de neurones Multi-Layer Perceptron pour la classification de maladies
    Architecture: Input -> Dense(256) -> ReLU -> Dropout -> Dense(128) -> ReLU -> Dropout -> Output
    """
    
    def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.3):
        super(DiseaseClassifierMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, num_classes)
        )
        
        # Utiliser softmax pour obtenir des probabilités calibrées
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        """Retourne les probabilités (après softmax)"""
        with torch.no_grad():
            logits = self.forward(x)
            return self.softmax(logits)


class DeepLearningPredictor:
    """
    Prédicteur de maladies utilisant le Deep Learning
    Fournit des probabilités mieux calibrées que la régression logistique
    """
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        model_path: Optional[Path] = None
    ):
        self.dataset_path = dataset_path or settings.DATASET_PATH
        self.model_path = model_path or (settings.MODELS_DIR / "deep_learning_model.pt")
        self.vectorizer_path = settings.MODELS_DIR / "dl_vectorizer.joblib"
        self.classes_path = settings.MODELS_DIR / "dl_classes.joblib"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        self.model: Optional[DiseaseClassifierMLP] = None
        self.vectorizer = SymptomVectorizer()
        self.classes: List[str] = []
        self.is_trained = False
        
        # Charger le dataset
        self.df = self._load_dataset()
        
        # Charger le modèle si disponible
        if self.model_path.exists():
            self._load_model()
        else:
            print("Modèle Deep Learning non trouvé. Entraînement nécessaire.")
    
    def _load_dataset(self) -> pd.DataFrame:
        """Charge et prépare le dataset"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        
        if len(df.columns) >= 2:
            df.columns = ['disease', 'symptom'] + list(df.columns[2:])
        
        df['disease'] = df['disease'].astype(str).str.lower().str.strip()
        df['symptom'] = df['symptom'].astype(str).str.lower().str.strip()
        
        df = df.dropna(subset=['disease', 'symptom'])
        df = df[df['disease'] != 'nan']
        df = df[df['symptom'] != 'nan']
        
        return df
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement"""
        # Grouper les symptômes par maladie
        disease_symptoms = self.df.groupby('disease')['symptom'].apply(list).reset_index()
        
        # Créer le vocabulaire des symptômes
        all_symptoms = [symptoms for symptoms in disease_symptoms['symptom'].values]
        self.vectorizer.fit(all_symptoms)
        
        # Créer les classes
        self.classes = disease_symptoms['disease'].tolist()
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Créer les vecteurs d'entrée et les labels
        X_list = []
        y_list = []
        
        for _, row in disease_symptoms.iterrows():
            disease = row['disease']
            symptoms = row['symptom']
            
            # Vecteur de symptômes
            X_list.append(self.vectorizer.transform(symptoms))
            y_list.append(class_to_idx[disease])
        
        # Data augmentation: créer des sous-ensembles de symptômes
        augmented_X = []
        augmented_y = []
        
        for i, (x, y) in enumerate(zip(X_list, y_list)):
            # Ajouter l'exemple original
            augmented_X.append(x)
            augmented_y.append(y)
            
            # Créer des variations (sous-ensembles de symptômes)
            symptom_indices = np.where(x > 0)[0]
            if len(symptom_indices) > 2:
                for _ in range(min(5, len(symptom_indices))):  # Max 5 variations
                    # Sélectionner aléatoirement un sous-ensemble
                    n_select = np.random.randint(2, len(symptom_indices) + 1)
                    selected = np.random.choice(symptom_indices, n_select, replace=False)
                    
                    aug_x = np.zeros_like(x)
                    aug_x[selected] = 1.0
                    augmented_X.append(aug_x)
                    augmented_y.append(y)
        
        X = np.array(augmented_X, dtype=np.float32)
        y = np.array(augmented_y, dtype=np.int64)
        
        return X, y
    
    def train(self, epochs: int = 30, batch_size: int = 128, learning_rate: float = 0.001):
        """Entraîne le modèle de deep learning"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch n'est pas installé. Installez-le avec: pip install torch")
        
        print("Préparation des données d'entraînement...")
        X, y = self._prepare_training_data()
        
        print(f"Données: {X.shape[0]} exemples, {X.shape[1]} features, {len(self.classes)} classes")
        
        # Créer les tensors PyTorch
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialiser le modèle
        self.model = DiseaseClassifierMLP(
            input_size=self.vectorizer.vocab_size,
            num_classes=len(self.classes),
            dropout_rate=0.3
        ).to(self.device)
        
        # Poids des classes pour gérer le déséquilibre
        class_counts = np.bincount(y, minlength=len(self.classes))
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * len(self.classes)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Entraînement
        print("Début de l'entraînement...")
        print(f"Nombre de batches par epoch: {len(dataloader)}")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            scheduler.step()
            
            # Afficher à chaque epoch pour voir la progression
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        self.is_trained = True
        self._save_model()
        print("Entraînement terminé et modèle sauvegardé!")
    
    def _save_model(self):
        """Sauvegarde le modèle et les métadonnées"""
        if self.model is None:
            return
            
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle PyTorch
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.vectorizer.vocab_size,
            'num_classes': len(self.classes)
        }, self.model_path)
        
        # Sauvegarder le vectorizer
        self.vectorizer.save(self.vectorizer_path)
        
        # Sauvegarder les classes
        joblib.dump(self.classes, self.classes_path)
    
    def _load_model(self):
        """Charge le modèle pré-entraîné"""
        if not TORCH_AVAILABLE:
            print("PyTorch non disponible, impossible de charger le modèle")
            return
            
        try:
            # Charger les métadonnées
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if self.vectorizer_path.exists():
                self.vectorizer.load(self.vectorizer_path)
            
            if self.classes_path.exists():
                self.classes = joblib.load(self.classes_path)
            
            # Reconstruire le modèle
            self.model = DiseaseClassifierMLP(
                input_size=checkpoint['input_size'],
                num_classes=checkpoint['num_classes']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_trained = True
            
            print("Modèle Deep Learning chargé avec succès!")
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.model = None
            self.is_trained = False
    
    def predict(self, patient: PatientInput, top_n: int = 5) -> List[DiagnosisCandidate]:
        """
        Prédit les maladies à partir des symptômes du patient
        
        Args:
            patient: Objet PatientInput contenant les symptômes
            top_n: Nombre de prédictions à retourner
            
        Returns:
            Liste des candidats de diagnostic avec probabilités calibrées
        """
        symptoms = patient.symptoms if hasattr(patient, 'symptoms') else patient
        
        if not symptoms:
            return []
            
        if not self.is_trained or self.model is None:
            print("Modèle non entraîné. Utilisation du fallback règles.")
            return self._predict_with_rules(symptoms, top_n)
        
        try:
            # Vectoriser les symptômes
            X = self.vectorizer.transform(symptoms)
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            
            # Prédiction
            self.model.eval()
            with torch.no_grad():
                probas = self.model.predict_proba(X_tensor)[0].cpu().numpy()
            
            # Obtenir les top N
            top_indices = probas.argsort()[-top_n:][::-1]
            
            candidates = []
            for idx in top_indices:
                disease_name = self.classes[idx]
                confidence = float(probas[idx])
                
                # Ne garder que si la confiance est significative
                if confidence >= 0.01:  # Seuil minimal de 1%
                    # Trouver les symptômes correspondants
                    disease_symptoms = self.df[self.df['disease'] == disease_name]['symptom'].tolist()
                    matched = [s for s in symptoms if any(s.lower() in ds or ds in s.lower() for ds in disease_symptoms)]
                    
                    candidates.append(DiagnosisCandidate(
                        disease_name=disease_name,
                        confidence=confidence,
                        matched_symptoms=matched if matched else symptoms
                    ))
            
            return candidates
            
        except Exception as e:
            print(f"Erreur de prédiction Deep Learning: {e}")
            return self._predict_with_rules(symptoms, top_n)
    
    def _predict_with_rules(self, symptoms: List[str], top_n: int = 5) -> List[DiagnosisCandidate]:
        """Fallback: prédiction basée sur les règles avec scoring amélioré"""
        if not symptoms:
            return []
        
        disease_scores = {}
        
        for disease in self.df['disease'].unique():
            disease_symptoms = set(self.df[self.df['disease'] == disease]['symptom'].tolist())
            total_disease_symptoms = len(disease_symptoms)
            
            matches = 0
            matched_symptoms = []
            
            for user_symptom in symptoms:
                user_symptom_lower = user_symptom.lower()
                for db_symptom in disease_symptoms:
                    if user_symptom_lower in db_symptom or db_symptom in user_symptom_lower:
                        matches += 1
                        matched_symptoms.append(user_symptom)
                        break
            
            if matches > 0:
                # Score amélioré: Jaccard-like similarity
                # Prend en compte à la fois la couverture des symptômes utilisateur
                # et la proportion des symptômes de la maladie couverts
                user_coverage = matches / len(symptoms)
                disease_coverage = matches / total_disease_symptoms if total_disease_symptoms > 0 else 0
                
                # Score harmonique (F1-like)
                if user_coverage + disease_coverage > 0:
                    confidence = 2 * (user_coverage * disease_coverage) / (user_coverage + disease_coverage)
                else:
                    confidence = 0
                
                # Ajuster pour éviter les 100%
                confidence = min(confidence * 0.95, 0.95)
                
                disease_scores[disease] = {
                    'confidence': confidence,
                    'matches': matches,
                    'matched_symptoms': list(set(matched_symptoms))
                }
        
        sorted_diseases = sorted(
            disease_scores.items(),
            key=lambda x: (x[1]['confidence'], x[1]['matches']),
            reverse=True
        )
        
        candidates = []
        for disease_name, info in sorted_diseases[:top_n]:
            candidates.append(DiagnosisCandidate(
                disease_name=disease_name,
                confidence=info['confidence'],
                matched_symptoms=info['matched_symptoms']
            ))
        
        return candidates
    
    def get_disease_info(self, disease_name: str):
        """Récupère les informations d'une maladie"""
        from medical_agent.models.data_models import Disease
        
        disease_data = self.df[self.df['disease'] == disease_name.lower()]
        
        if disease_data.empty:
            return None
        
        symptoms = disease_data['symptom'].unique().tolist()
        
        return Disease(
            name=disease_name,
            symptoms=symptoms
        )
