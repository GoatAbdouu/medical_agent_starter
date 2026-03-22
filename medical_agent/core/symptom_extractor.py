"""
Extracteur de symptômes - Parse le texte patient et extrait les symptômes
Supporte le texte naturel et la reconnaissance vocale
"""
import re
from typing import List, Dict, Optional, Set
from pathlib import Path

from medical_agent.models.data_models import PatientInput
from medical_agent.config.settings import settings


class SymptomExtractor:
    """Extrait les symptômes du texte naturel du patient"""
    
    def __init__(self, synonyms_path: Optional[Path] = None):
        """
        Args:
            synonyms_path: Chemin vers le fichier de synonymes YAML
        """
        self.synonyms_path = synonyms_path or settings.SYNONYMS_PATH
        self.synonyms = self._load_synonyms()
        
        # Patterns regex améliorés pour la reconnaissance vocale
        # Supporte: "39°C", "39 degrés", "39.5 celsius", "trente-neuf", etc.
        self.temp_pattern = re.compile(
            r'(\d{2}(?:[.,]\d)?)\s*(?:°\s*[cC]|degr[ée]s?(?:\s+celsius)?|celsius)',
            re.IGNORECASE
        )
        
        # Pattern alternatif pour température sans unité (ex: "fièvre à 39")
        self.temp_alt_pattern = re.compile(
            r'(?:fièvre|fievre|température|temperature)\s+(?:à|de|a)?\s*(\d{2}(?:[.,]\d)?)',
            re.IGNORECASE
        )
        
        self.duration_pattern = re.compile(
            r"(depuis|il y a|ça fait|ca fait)\s+([0-9]+|quelques|deux|trois|quatre|cinq)\s*(minute|heure|jour|semaine|mois)s?",
            re.IGNORECASE
        )
        
        # Mots à ignorer (bruit de la reconnaissance vocale)
        self.stop_words = {
            "et", "je", "j'ai", "j ai", "un", "une", "de", "du", "la", "le", "les",
            "à", "a", "au", "aux", "en", "avec", "sans", "pour", "sur", "dans",
            "très", "tres", "beaucoup", "peu", "aussi", "vraiment", "suis", "ai"
        }
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Charge les synonymes de symptômes"""
        return settings.load_yaml_config(self.synonyms_path)
    
    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour la reconnaissance vocale"""
        text = text.lower().strip()
        text = text.replace(",", ".")
        
        # Normaliser les variations courantes de la reconnaissance vocale
        replacements = {
            "j'ai": "j ai",
            "d'un": "d un", 
            "l'ai": "l ai",
            "c'est": "c est",
            "qu'est": "qu est",
            "degrés celsius": "degrés",
            "degré celsius": "degrés",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _extract_temperature(self, text: str) -> Optional[float]:
        """Extrait la température du texte (supporte voix et texte)"""
        # Essayer le pattern principal
        match = self.temp_pattern.search(text)
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                pass
        
        # Essayer le pattern alternatif (fièvre à 39)
        match = self.temp_alt_pattern.search(text)
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                pass
        
        return None
    
    def _extract_duration(self, text: str) -> Optional[str]:
        """Extrait la durée des symptômes"""
        match = self.duration_pattern.search(text)
        if match:
            return " ".join(match.groups())
        return None
    
    def _extract_intensity(self, text: str) -> Optional[str]:
        """Extrait l'intensité des symptômes"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["fort", "intense", "violente", "sévère", "insupportable", "terrible", "horrible"]):
            return "severe"
        elif any(word in text_lower for word in ["léger", "légère", "faible", "un peu"]):
            return "mild"
        elif any(word in text_lower for word in ["moyen", "moyenne", "modéré", "modérée"]):
            return "moderate"
        return None
    
    def _fuzzy_match(self, text: str, pattern: str, threshold: float = 0.8) -> bool:
        """
        Matching flou pour gérer les erreurs de reconnaissance vocale
        Retourne True si le pattern est trouvé dans le texte avec une certaine tolérance
        """
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        
        # Match exact
        if pattern_lower in text_lower:
            return True
        
        # Match partiel (pour les mots coupés ou mal reconnus)
        pattern_words = pattern_lower.split()
        if len(pattern_words) > 1:
            # Pour les expressions multi-mots, vérifier si tous les mots sont présents
            matches = sum(1 for word in pattern_words if word in text_lower)
            if matches / len(pattern_words) >= threshold:
                return True
        
        return False
    
    def _extract_symptoms_from_synonyms(self, text: str) -> List[str]:
        """Extrait les symptômes en utilisant les synonymes avec matching amélioré"""
        text_lower = text.lower()
        symptoms = []
        matched_canonical = set()
        
        for canonical, words in self.synonyms.items():
            if canonical in matched_canonical:
                continue
                
            for word in words:
                word_lower = word.lower()
                
                # Matching exact
                if word_lower in text_lower:
                    if word not in symptoms:
                        symptoms.append(word)
                        matched_canonical.add(canonical)
                    break
                
                # Matching flou pour les expressions multi-mots
                if self._fuzzy_match(text_lower, word_lower):
                    if word not in symptoms:
                        symptoms.append(word)
                        matched_canonical.add(canonical)
                    break
        
        # Détection supplémentaire par mots-clés courants (voix)
        # Ne pas ajouter si le symptôme canonique est déjà détecté
        additional_symptoms = self._detect_voice_symptoms(text_lower, matched_canonical)
        for symptom in additional_symptoms:
            if symptom not in symptoms:
                symptoms.append(symptom)
        
        return symptoms
    
    def _detect_voice_symptoms(self, text: str, already_matched: Set[str] = None) -> List[str]:
        """
        Détecte les symptômes additionnels courants dans la reconnaissance vocale
        Gère les variations et erreurs de transcription
        """
        if already_matched is None:
            already_matched = set()
            
        symptoms = []
        
        # Mapping des patterns vers les clés canoniques
        voice_patterns = {
            "fièvre": ("fever", [r"fi[eè]vre", r"fievre", r"de la fi[eè]vre"]),
            "mal à la gorge": ("sore_throat", [r"mal\s+[àa]\s+la\s+gorge", r"gorge\s+(?:qui\s+)?(?:fait\s+)?mal", r"angine"]),
            "toux": ("cough", [r"tousse", r"toux", r"je\s+tousse"]),
            "mal de tête": ("headache", [r"mal\s+(?:de|à|a)\s+(?:la\s+)?t[êe]te", r"c[ée]phal[ée]e", r"migraine", r"maux\s+de\s+t[êe]te"]),
            "fatigue": ("fatigue", [r"fatigu[ée]", r"[ée]puis[ée]", r"crev[ée]"]),
            "nausée": ("nausea", [r"naus[ée]e", r"envie\s+de\s+vomir", r"mal\s+au\s+coeur"]),
            "vomissement": ("vomiting", [r"vomi", r"vomiss"]),
            "diarrhée": ("diarrhea", [r"diarrh[ée]e", r"selles\s+liquides"]),
            "essoufflement": ("dyspnea", [r"essouffl[ée]", r"du\s+mal\s+[àa]\s+respirer", r"respir(?:e|ation)\s+difficile"]),
            "douleur abdominale": ("abdominal_pain", [r"mal\s+au\s+ventre", r"douleur\s+(?:au\s+)?ventre", r"ventre\s+(?:qui\s+)?fait\s+mal"]),
            "frissons": ("fever", [r"frisson", r"je\s+tremble", r"tremblements"]),
            "courbatures": ("muscle_pain", [r"courbatur", r"douleurs?\s+musculaires?", r"muscles?\s+(?:qui\s+)?(?:font\s+)?mal"]),
        }
        
        for symptom, (canonical, patterns) in voice_patterns.items():
            # Skip si déjà détecté via synonymes
            if canonical in already_matched:
                continue
                
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if symptom not in symptoms:
                        symptoms.append(symptom)
                    break
        
        return symptoms
    
    def _extract_risk_factors(self, text: str) -> List[str]:
        """Extrait les facteurs de risque"""
        text_lower = text.lower()
        risk_factors = []
        
        # Diabète
        if any(word in text_lower for word in ["diabète", "diabete", "diabétique"]):
            risk_factors.append("diabetes")
        
        # Hypertension
        if any(word in text_lower for word in ["hypertension", "tension", "hypertendu"]):
            risk_factors.append("hypertension")
        
        # Grossesse
        if any(word in text_lower for word in ["enceinte", "grossesse", "femme enceinte"]):
            risk_factors.append("pregnancy")
        
        # Âge avancé
        if any(word in text_lower for word in ["âgé", "age", "senior", "vieux"]):
            risk_factors.append("elderly")
        
        # Immunodépression
        if any(word in text_lower for word in ["immunodéprim", "défenses immunitaires"]):
            risk_factors.append("immunocompromised")
        
        return risk_factors
    
    def extract(self, text: str) -> PatientInput:
        """
        Extrait toutes les informations du texte patient
        
        Args:
            text: Texte décrivant les symptômes du patient
            
        Returns:
            PatientInput avec toutes les informations extraites
        """
        normalized_text = self._normalize_text(text)
        
        # Créer l'objet PatientInput
        patient_input = PatientInput(raw_text=text)
        
        # Extraire les symptômes
        patient_input.symptoms = self._extract_symptoms_from_synonyms(normalized_text)
        
        # Extraire la température
        temp = self._extract_temperature(normalized_text)
        if temp:
            patient_input.measured_values["temperature_c"] = temp
        
        # Extraire la durée
        duration = self._extract_duration(normalized_text)
        if duration:
            patient_input.onset = duration
            patient_input.metadata["duration_raw"] = duration
        
        # Extraire l'intensité
        patient_input.intensity = self._extract_intensity(text)
        
        # Extraire les facteurs de risque
        patient_input.risk_factors = self._extract_risk_factors(normalized_text)
        
        return patient_input
