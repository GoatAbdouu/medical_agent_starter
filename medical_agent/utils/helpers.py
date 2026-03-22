"""
Utilitaires divers pour l'agent médical
"""
import re
from typing import List, Dict
import pandas as pd


def normalize_disease_name(name: str) -> str:
    """Normalise le nom d'une maladie"""
    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)
    return name


def normalize_symptom(symptom: str) -> str:
    """Normalise un symptôme"""
    symptom = symptom.lower().strip()
    symptom = re.sub(r'\s+', ' ', symptom)
    return symptom


def format_temperature(temp: float) -> str:
    """Formate une température pour l'affichage"""
    return f"{temp:.1f}°C"


def get_severity_color(confidence: float) -> str:
    """Retourne une couleur basée sur la confiance"""
    if confidence >= 0.7:
        return "green"
    elif confidence >= 0.4:
        return "orange"
    else:
        return "red"


def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    """Calcule la similarité de Jaccard entre deux ensembles"""
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def load_csv_safely(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """Charge un CSV de manière sécurisée"""
    try:
        return pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        # Essayer avec un autre encodage
        return pd.read_csv(filepath, encoding='latin-1')
    except Exception as e:
        raise Exception(f"Erreur lors du chargement de {filepath}: {e}")


def validate_symptoms(symptoms: List[str]) -> bool:
    """Valide une liste de symptômes"""
    if not symptoms:
        return False
    
    if not isinstance(symptoms, list):
        return False
    
    # Vérifier que tous les éléments sont des chaînes non vides
    return all(isinstance(s, str) and s.strip() for s in symptoms)


def extract_numbers(text: str) -> List[float]:
    """Extrait tous les nombres d'un texte"""
    pattern = r'\d+(?:[.,]\d+)?'
    matches = re.findall(pattern, text)
    return [float(m.replace(',', '.')) for m in matches]
