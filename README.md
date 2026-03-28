# 🏥 Medical Vision AI Agent

A **complete Medical Vision AI Agent** for dermatological image analysis using an ensemble of three deep learning models combined with a Vision-Language Model (VLM) for natural-language explanations.

> ⚠️ **Medical Disclaimer**: This system is for research and informational purposes only. It does **NOT** provide medical diagnoses. Always consult a licensed healthcare professional.
> ⚠️ **This is not medical advice. Consult a healthcare professional.**

---

## Architecture Overview

```
Image Input
    │
    ├──► EfficientNet-B3 ──► probabilities
    ├──► MobileNetV2     ──► probabilities  ──► Ensemble Logic ──► SkinDiagnosisResult
    └──► ResNet50        ──► probabilities                              │
                                                                        ▼
                                                                 VLM Explainer
                                                                 (BLIP / template)
                                                                        │
                                                                        ▼
                                                              Natural Language Explanation
```

### Ensemble Logic
- **All 3 models agree** → Average their confidence scores
- **Models disagree** → Take the prediction with the highest individual confidence

### 10 Skin Disease Classes
| Class | Severity |
|-------|----------|
| Eczéma | Moderate |
| Verrues / Molluscum | Mild |
| Mélanome | **Critical** |
| Dermatite Atopique | Moderate |
| Carcinome Basocellulaire | **Urgent** |
| Nævus Mélanocytaire | Mild |
| Kératose Bénigne | Mild |
| Psoriasis / Lichen Plan | Moderate |
| Kératose Séborrhéique | Mild |
| Mycose / Teigne | Moderate |

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Launch the Streamlit App
```bash
streamlit run app.py
```
The app will be accessible at `http://localhost:8501`

### App Features
- **🩺 Diagnostic Symptômes** — symptom-based diagnosis with Deep Learning / rules engine
- **📷 Diagnostic Peau** — skin image analysis with ensemble models + VLM explanation
  - Upload or capture a skin image
  - See ensemble result with confidence score
  - See individual model predictions (expandable)
  - Read natural language VLM explanation

---

## How to Train

### Train the full ensemble (all 3 models)
```bash
python scripts/train_ensemble.py \
    --data_dir "path/to/IMG_CLASSES" \
    --output_dir models \
    --epochs 40 \
    --batch_size 32 \
    --lr 1e-3
```
Each model trains in 2 phases:
- **Phase 1** (10 epochs): classifier head only
- **Phase 2** (up to 30 epochs): full fine-tuning + early stopping

Outputs:
- `models/efficientnet_skin.pth`
- `models/mobilenet_skin.pth`
- `models/resnet_skin.pth`
- `models/ensemble_training_history.json`

### Train single MobileNetV2 (legacy)
```bash
python scripts/train_skin_classifier.py \
    --data_dir "path/to/IMG_CLASSES"
```

### Visualise training history
```bash
python scripts/plot_training_history.py \
    --history models/ensemble_training_history.json \
    --output_dir outputs
```

---

## Google Colab Notebook

Open `notebooks/Medical_Vision_AI_Training.ipynb` in Google Colab for GPU-accelerated training:

1. Upload to [Google Colab](https://colab.research.google.com)
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run all cells

The notebook covers:
- Dataset exploration
- Image augmentation demo
- Training all 3 models with progress bars
- Loss/accuracy graphs per epoch
- Ensemble inference demo
- VLM explanation demo
- Saving models to Google Drive

---

## Project Structure

```
medical_agent_starter/
├── app.py                          # Streamlit app (ensemble + VLM UI)
├── requirements.txt
├── README.md
├── medical_agent/
│   ├── core/
│   │   ├── agent.py                # Main orchestrator
│   │   ├── system_prompt.py        # SYSTEM_PROMPT constant (NEW)
│   │   ├── ensemble_classifier.py  # EfficientNet + MobileNet + ResNet ensemble (NEW)
│   │   ├── vlm_explainer.py        # VLM explanation module (NEW)
│   │   ├── image_pipeline.py       # Image preprocessing & augmentation (NEW)
│   │   ├── skin_disease_classifier.py  # Single MobileNetV2 (legacy)
│   │   ├── disease_predictor.py
│   │   ├── symptom_extractor.py
│   │   ├── triage_system.py
│   │   └── ...
│   ├── models/
│   │   └── data_models.py
│   ├── config/
│   └── services/
├── scripts/
│   ├── train_ensemble.py           # Ensemble training script (NEW)
│   ├── plot_training_history.py    # Training visualisation (NEW)
│   └── train_skin_classifier.py    # Single model training (legacy)
├── notebooks/
│   └── Medical_Vision_AI_Training.ipynb  # Google Colab notebook (NEW)
├── models/                         # Saved model weights
│   ├── efficientnet_skin.pth       # (after training)
│   ├── mobilenet_skin.pth          # (after training)
│   ├── resnet_skin.pth             # (after training)
│   └── ensemble_training_history.json
└── outputs/                        # Training plots
```

---

## Python API

### Ensemble Classifier
```python
from medical_agent.core.ensemble_classifier import EnsembleClassifier
from PIL import Image

classifier = EnsembleClassifier(
    efficientnet_path="models/efficientnet_skin.pth",
    mobilenet_path="models/mobilenet_skin.pth",
    resnet_path="models/resnet_skin.pth",
)

image = Image.open("skin_photo.jpg")
result = classifier.predict(image, top_n=5)
individual = classifier.get_individual_predictions(image, top_n=3)

print(result.top_diagnosis.readable_name, result.top_diagnosis.confidence)
```

### VLM Explainer
```python
from medical_agent.core.vlm_explainer import VLMExplainer

explainer = VLMExplainer(use_blip=False)  # template fallback
explanation = explainer.explain(image, result)
print(explanation)
```

### Medical Agent (full pipeline)
```python
from medical_agent import MedicalAgent

agent = MedicalAgent()

# Symptom diagnosis
result = agent.diagnose("J'ai de la fièvre à 39°C et mal à la gorge")

# Skin image diagnosis (ensemble if available, single model fallback)
skin_result = agent.diagnose_skin_image(image, top_n=5)

# Ensemble with individual predictions
ensemble_result, individual_preds = agent.diagnose_skin_image_ensemble(image, top_n=5)
```

### System Prompt
```python
from medical_agent.core.system_prompt import SYSTEM_PROMPT
print(SYSTEM_PROMPT)
```

---

## Configuration

`medical_agent/config/settings.py` — thresholds, paths, triage levels

---

## Medical Disclaimer

This tool is **informational only** and does **NOT** replace a consultation with a licensed medical professional. In case of emergency, call **15 (SAMU)**.

⚠️ **This is not medical advice. Consult a healthcare professional.**

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

