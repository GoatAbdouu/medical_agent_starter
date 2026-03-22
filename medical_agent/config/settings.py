"""
Configuration centralisée du projet
"""
from pathlib import Path
from typing import List, Dict
import yaml

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "medical_agent" / "config"

class Settings:
    """Configuration globale de l'application"""
    
    # Chemins des dossiers (exposés dans la classe)
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    MODELS_DIR = MODELS_DIR
    CONFIG_DIR = CONFIG_DIR
    
    # Chemins des fichiers
    DATASET_PATH = DATA_DIR / "cleaned_data.csv"
    DISEASE_MODEL_PATH = MODELS_DIR / "disease_predictor.joblib"
    SYMPTOM_VOCAB_PATH = MODELS_DIR / "symptom_vocab.joblib"
    ROUTER_MODEL_PATH = MODELS_DIR / "router.joblib"
    
    # Fichiers de configuration
    SYNONYMS_PATH = CONFIG_DIR / "synonyms.yaml"
    RED_FLAGS_PATH = CONFIG_DIR / "red_flags.yaml"
    QUESTIONS_PATH = CONFIG_DIR / "questions.yaml"
    
    # Paramètres de l'agent
    TOP_DIAGNOSES = 5
    CONFIDENCE_THRESHOLD = 0.3
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    
    # Spécialités médicales
    SPECIALTIES = [
        "general", "orl", "respiratory", "digestive",
        "neuro", "dermato", "uro", "cardio", "osteo"
    ]
    
    # Niveaux de triage
    TRIAGE_LEVELS = {
        "critique": {"color": "red", "priority": 1, "action": "Appel SAMU immédiat"},
        "urgent": {"color": "orange", "priority": 2, "action": "Consultation aux urgences"},
        "normal": {"color": "yellow", "priority": 3, "action": "Consultation médecin traitant"},
        "léger": {"color": "green", "priority": 4, "action": "Surveillance à domicile"}
    }
    
    # Seuils de température
    TEMP_FEVER_THRESHOLD = 38.0
    TEMP_HIGH_FEVER_THRESHOLD = 40.0
    
    @staticmethod
    def load_yaml_config(file_path: Path) -> Dict:
        """Charge un fichier de configuration YAML"""
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def ensure_directories():
        """Crée les dossiers nécessaires s'ils n'existent pas"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
