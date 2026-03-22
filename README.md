# Agent Médical v2.0

Système de diagnostic médical utilisant l'apprentissage automatique et des règles métier pour le triage et le diagnostic préliminaire.

## Fonctionnalités

- Extraction de symptômes depuis le texte naturel
- Diagnostic combinant ML et règles métier
- Système de triage avec évaluation de l'urgence
- Génération de recommandations
- Questions de suivi automatiques
- Interface Streamlit

## Installation

### Prérequis

- Python 3.8+
- pip

### Installation

```bash
cd medical_agent_starter
pip install -r requirements.txt
```

### Structure

```
medical_agent_starter/
├── medical_agent/
│   ├── core/              # Modules principaux
│   ├── models/            # Modèles de données
│   ├── config/            # Configuration
│   └── utils/             # Utilitaires
├── data/                  # Données CSV
├── models/                # Modèles ML
├── scripts/               # Scripts
├── tests/                 # Tests
└── app.py                 # Application
```

## Démarrage

### Lancer l'application web

```bash
streamlit run app.py
```

L'application sera accessible à `http://localhost:8501`

### Utilisation en Python

```python
from medical_agent import MedicalAgent

# Initialiser l'agent
agent = MedicalAgent()

```bash
streamlit run app.py
```

Application accessible sur http://localhost:8501

### Utilisation Python

```python
from medical_agent import MedicalAgent

agent = MedicalAgent()
### Composants

**SymptomExtractor**: Extraction de symptômes depuis le texte

**DiseasePredictor**: Prédiction par ML et règles métier

**TriageSystem**: Évaluation de l'urgence

**MedicalAgent**: Orchestration des composants

##crivez vos symptômes
3. Cliquez sur "Analyser"
4. Consultez les résultats

### Via l'API Python

```python
from medical_agent import MedicalAgent
Interface web

```bash
streamlit run app.py
```

### API Python

```python
from medical_agent import MedicalAgent

agent = MedicalAgent()
result = agent.diagnose("Mal de tête et fièvre")

print(f"Urgence: {result.triage.level}")
for candidate in result.candidates:
    print(f"{candidate.disease_name}: {candidate.confidence:.2%}")
```
Configuration dans `medical_agent/config/settings.py`:
- Seuils de confiance
- Niveaux de triage
- Chemins des fichiers

## API

**MedicalAgent**
```python
agent = MedicalAgent()
result = agent.diagnose(text, top_n=5)
```

**SymptomExtractor**
```python
extractor = SymptomExtractor()
patient = extractor.extract(text)
```

**DiseasePredictor**
```python
predictor = DiseasePredictor()
candidates = predictor.predict(patient, top_n=5)
```

**TriageSystem**
```python
triage = TriageSystem()
result = triage.evaluate(patient)
```

## Tests

```bash
python3 scripts/test_system.py
```

## Format des données

`data/cleaned_data.csv`:
```csv
disease,symptom
grippe,fièvre
grippe,toux
```

## Avertissement

Outil informatif uniquement. Ne remplace pas une consultation médicale. En cas d'urgence: 15 (SAMU).

