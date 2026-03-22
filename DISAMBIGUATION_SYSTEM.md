# Système de Désambiguïsation - Documentation

## 🎯 Problème Résolu

### Problème Initial
Lorsque les utilisateurs entraient des symptômes génériques (ex: fièvre, fatigue, mal de tête), le modèle produisait des prédictions sur-confiantes (ex: Lyme disease à 78%), ce qui est médicalement irréaliste et causé par:
- **Biais du dataset**: Certaines maladies sont sur-représentées
- **Manque d'information discriminante**: Les symptômes génériques sont communs à de nombreuses maladies

### Solution Implémentée
Un système de désambiguïsation intelligent qui:
1. **Détecte** quand les symptômes sont trop génériques
2. **Ajuste** les confiances à la baisse pour refléter l'incertitude médicale
3. **Génère** des questions ciblées pour différencier les diagnostics
4. **Supporte** des conversations multi-tours pour affiner progressivement le diagnostic

---

## 📦 Nouveaux Composants

### 1. `disambiguation.py` - Détecteur de Désambiguïsation

**Classe: `DisambiguationDetector`**

#### Fonctionnalités Principales

```python
# Liste des symptômes génériques
GENERIC_SYMPTOMS = {
    'fièvre', 'fatigue', 'mal de tête', 'douleur', 
    'nausée', 'toux', 'étourdissement', 'faiblesse', ...
}
```

#### Méthodes Clés

**`is_generic_symptom(symptom: str) -> bool`**
- Vérifie si un symptôme est considéré comme générique

**`calculate_symptom_specificity(symptoms: List[str]) -> float`**
- Retourne un score de spécificité (0 = tous génériques, 1 = tous spécifiques)

**`needs_disambiguation(patient, candidates) -> Dict`**
- Détermine si une désambiguïsation est nécessaire
- Critères:
  - Symptômes uniquement génériques
  - Faible spécificité avec confiance élevée (suspect)
  - Prédictions ambiguës (scores similaires)
  - Informations insuffisantes (≤2 symptômes)

**`adjust_confidence_for_genericity(candidates, specificity_score) -> List`**
- Ajuste les scores de confiance basés sur la spécificité
- Formule: `adjusted = original × (0.3 + 0.7 × specificity)`
- Plafonné à 50% si spécificité < 0.5

#### Exemple d'Utilisation

```python
from medical_agent.core.disambiguation import DisambiguationDetector

detector = DisambiguationDetector()

# Vérifier si un symptôme est générique
is_generic = detector.is_generic_symptom("fièvre")  # True

# Calculer la spécificité
specificity = detector.calculate_symptom_specificity(
    ["fièvre", "toux", "éruption cutanée"]
)  # ~0.33 (1 spécifique sur 3)

# Vérifier besoin de désambiguïsation
disambiguation_info = detector.needs_disambiguation(patient, candidates)
# Returns:
# {
#     'needs_disambiguation': True,
#     'reason': 'Symptômes trop génériques...',
#     'specificity_score': 0.0,
#     'top_confidence': 0.78
# }

# Ajuster les confiances
adjusted = detector.adjust_confidence_for_genericity(candidates, 0.2)
```

---

### 2. `question_generator.py` - Générateur de Questions

**Classe: `FollowUpQuestionGenerator`**

#### Fonctionnalités Principales

Génère des questions **discriminantes** qui permettent de différencier entre des diagnostics similaires.

#### Méthodes Clés

**`find_discriminating_symptoms(candidates, patient_symptoms, top_n=3)`**
- Trouve les symptômes qui distinguent les maladies candidates
- Calcule un score de pouvoir discriminant:
  - Score = 1 : symptôme unique à une maladie (très discriminant)
  - Score = 0 : symptôme présent dans toutes les maladies (pas discriminant)

**`generate_questions_for_disambiguation(patient, candidates, max_questions=3)`**
- Génère des questions optimisées pour la désambiguïsation
- Priorise les questions discriminantes
- Format adapté au symptôme

**`generate_all_follow_up_questions(patient, candidates, max_questions=5)`**
- Génère tous types de questions (discriminantes + contextuelles)
- Retourne des objets détaillés avec priorités et métadonnées

#### Exemple d'Utilisation

```python
from medical_agent.core.question_generator import FollowUpQuestionGenerator

generator = FollowUpQuestionGenerator(dataset=df)

# Générer des questions discriminantes
questions = generator.generate_questions_for_disambiguation(
    patient=patient_input,
    candidates=diagnosis_candidates,
    max_questions=3
)

# Résultat:
# [
#     "Avez-vous des éruptions cutanées ?",
#     "Ressentez-vous des douleurs articulaires ?",
#     "Depuis combien de temps avez-vous ces symptômes ?"
# ]

# Version détaillée avec métadonnées
detailed = generator.generate_all_follow_up_questions(
    patient=patient_input,
    candidates=candidates,
    max_questions=5,
    prioritize_discriminating=True
)

# Résultat:
# [
#     {
#         'question': 'Avez-vous des éruptions cutanées ?',
#         'type': 'discriminating',
#         'priority': 0.85,
#         'related_diseases': ['lyme disease', 'measles'],
#         'symptom': 'éruption cutanée'
#     },
#     ...
# ]
```

---

### 3. Modèles de Données Améliorés

#### `DiagnosisResult` - Nouveaux Champs

```python
@dataclass
class DiagnosisResult:
    # ... champs existants ...
    
    # Nouveaux champs pour la désambiguïsation
    needs_disambiguation: bool = False
    disambiguation_reason: str = ""
    symptom_specificity_score: float = 1.0
    conversation_turn: int = 1
    
    @property
    def is_conclusive(self) -> bool:
        """Vérifie si le diagnostic est suffisamment conclusif"""
        return not self.needs_disambiguation and self.top_diagnosis is not None
```

#### `ConversationState` - Nouveau Modèle

```python
@dataclass
class ConversationState:
    """État de la conversation pour le suivi multi-tours"""
    conversation_id: str
    patient_input: PatientInput
    all_symptoms: List[str]
    diagnosis_history: List[DiagnosisResult]
    asked_questions: List[str]
    current_turn: int = 1
    max_turns: int = 5
    
    def add_symptoms(self, new_symptoms: List[str])
    def add_diagnosis_result(self, result: DiagnosisResult)
    def should_continue(self) -> bool
```

---

## 🔄 Flux de Travail

### 1. Diagnostic Initial

```python
agent = MedicalAgent()
result = agent.diagnose("J'ai de la fièvre et de la fatigue", top_n=5)

# Le système automatiquement:
# 1. Extrait les symptômes
# 2. Calcule la spécificité (fièvre=générique, fatigue=générique)
# 3. Fait les prédictions
# 4. Ajuste les confiances à la baisse (spécificité = 0%)
# 5. Détecte le besoin de désambiguïsation
# 6. Génère des questions discriminantes
```

### 2. Ajustement des Confiances

**Avant:**
```
Lyme disease: 78%
Grippe: 75%
Mononucléose: 72%
```

**Après (avec symptômes 100% génériques):**
```
Lyme disease: 23%  (78% × 0.3)
Grippe: 23%        (75% × 0.3)
Mononucléose: 22%  (72% × 0.3)
```

**Explication:**
- Facteur de pénalité = `0.3 + (0.7 × specificity_score)`
- Avec spécificité = 0: facteur = 0.3 (70% de réduction)
- Avec spécificité = 0.5: facteur = 0.65 (35% de réduction)
- Avec spécificité = 1.0: facteur = 1.0 (pas de réduction)

### 3. Conversation Multi-Tours

```python
# Tour 1: Symptômes initiaux
result1 = agent.diagnose("J'ai de la fièvre et de la fatigue")

if result1.needs_disambiguation:
    print(f"Questions: {result1.questions}")
    # ["Avez-vous des éruptions cutanées ?", ...]
    
    # Tour 2: Réponse aux questions
    conversation = ConversationState(
        conversation_id="123",
        patient_input=result1.patient_input
    )
    
    result2 = agent.continue_conversation(
        conversation,
        "Oui, j'ai des éruptions cutanées rouges sur les bras"
    )
    
    # Maintenant avec plus d'informations:
    # - Spécificité augmentée
    # - Confiances plus élevées pour maladies avec éruptions
    # - Diagnostic plus précis
```

---

## 📊 Métriques et Seuils

### Seuils de Désambiguïsation

```python
# Dans DisambiguationDetector
max_confidence_for_generic = 0.50   # 50% max pour symptômes génériques
ambiguity_threshold = 0.15          # Écart minimal entre top 2 diagnostics
```

### Critères de Désambiguïsation

| Critère | Condition | Action |
|---------|-----------|--------|
| Symptômes génériques uniquement | 100% génériques | ✅ Désambiguïsation requise |
| Faible spécificité + haute confiance | Spécificité < 30% ET confiance > 50% | ✅ Désambiguïsation requise |
| Prédictions ambiguës | Écart top1-top2 < 15% | ✅ Désambiguïsation requise |
| Informations insuffisantes | ≤ 2 symptômes | ✅ Désambiguïsation requise |
| Spécificité haute + diagnostic clair | Spécificité > 50% ET écart > 15% | ❌ Pas de désambiguïsation |

---

## 🧪 Tests et Validation

### Exécuter les Tests

```bash
cd medical_agent_starter
python scripts/test_disambiguation.py
```

### Cas de Test

**Test 1: Symptômes Génériques**
```
Input: "J'ai de la fièvre, de la fatigue et mal à la tête"
Attendu: Confiances faibles (≤30%), désambiguïsation requise
```

**Test 2: Symptômes Spécifiques**
```
Input: "J'ai des douleurs thoraciques oppressantes, essoufflement au repos, et sueurs froides"
Attendu: Confiances normales, pas de désambiguïsation
```

**Test 3: Symptômes Mixtes**
```
Input: "J'ai de la fièvre à 38.5°C, des éruptions cutanées rouges sur le torse, et de la fatigue"
Attendu: Confiances modérées (~50-60%)
```

---

## 🎨 Interface Utilisateur

### Affichage de la Désambiguïsation

L'application Streamlit affiche maintenant:

```
⚠️ DÉSAMBIGUÏSATION NÉCESSAIRE

Symptômes trop génériques - besoin de plus d'informations spécifiques

Spécificité des symptômes: 0%

ℹ️ Les confiances affichées ont été ajustées pour refléter l'incertitude.
Veuillez répondre aux questions ci-dessous pour un diagnostic plus précis.
```

### Questions de Suivi

```
❓ QUESTIONS DE SUIVI:
1. Avez-vous des éruptions cutanées ?
2. Ressentez-vous des douleurs articulaires ?
3. Depuis combien de temps avez-vous ces symptômes ?
```

---

## 📈 Impact et Bénéfices

### Avant le Système

❌ **Problèmes:**
- Confiances irréalistes (78% pour Lyme disease avec juste "fièvre, fatigue")
- Pas de différenciation entre diagnostics similaires
- Aucune indication d'incertitude
- Risque de faux sentiment de certitude

### Après le Système

✅ **Améliorations:**
- Confiances ajustées reflétant l'incertitude réelle (23%)
- Questions discriminantes générées automatiquement
- Support pour affiner progressivement le diagnostic
- Communication transparente de l'incertitude
- Meilleure sécurité médicale

---

## 🔧 Configuration

### Personnaliser les Symptômes Génériques

```python
from medical_agent.core.disambiguation import DisambiguationDetector

# Ajouter des symptômes génériques personnalisés
custom_generics = {
    'symptôme_custom_1',
    'symptôme_custom_2',
}

detector = DisambiguationDetector(
    generic_symptoms=custom_generics,
    max_confidence_for_generic=0.45,  # Ajuster le seuil
    ambiguity_threshold=0.20           # Ajuster l'écart d'ambiguïté
)
```

### Ajuster les Facteurs de Pénalité

Dans `disambiguation.py`, méthode `adjust_confidence_for_genericity`:

```python
# Formule actuelle: 0.3 + 0.7 × specificity
# Pour être plus strict:
penalty_factor = 0.2 + (0.8 * specificity_score)

# Pour être plus permissif:
penalty_factor = 0.5 + (0.5 * specificity_score)
```

---

## 🚀 Utilisation Avancée

### API Programmatique

```python
from medical_agent import MedicalAgent
from medical_agent.models.data_models import ConversationState
import uuid

# Initialiser l'agent
agent = MedicalAgent()

# Diagnostic initial
result = agent.diagnose("J'ai de la fièvre et de la fatigue")

# Vérifier si désambiguïsation nécessaire
if result.needs_disambiguation:
    print(f"Raison: {result.disambiguation_reason}")
    print(f"Spécificité: {result.symptom_specificity_score:.0%}")
    
    # Créer un état de conversation
    conversation = ConversationState(
        conversation_id=str(uuid.uuid4()),
        patient_input=result.patient_input
    )
    
    # Afficher les questions
    for i, question in enumerate(result.questions, 1):
        print(f"{i}. {question}")
    
    # Simuler réponse utilisateur
    user_response = "Oui, j'ai aussi des éruptions cutanées et des douleurs articulaires"
    
    # Continuer la conversation
    result2 = agent.continue_conversation(conversation, user_response)
    
    # Vérifier amélioration
    print(f"Nouvelle spécificité: {result2.symptom_specificity_score:.0%}")
    print(f"Top diagnostic: {result2.top_diagnosis.disease_name}")
    print(f"Confiance: {result2.top_diagnosis.confidence:.1%}")
```

### Intégration dans un Chatbot

```python
class MedicalChatbot:
    def __init__(self):
        self.agent = MedicalAgent()
        self.conversations = {}
    
    def start_conversation(self, user_id, initial_message):
        result = self.agent.diagnose(initial_message)
        
        if result.needs_disambiguation:
            # Créer état de conversation
            self.conversations[user_id] = ConversationState(
                conversation_id=user_id,
                patient_input=result.patient_input
            )
            
            # Retourner les questions
            return {
                'type': 'questions',
                'questions': result.questions,
                'message': result.disambiguation_reason
            }
        else:
            # Diagnostic conclusif
            return {
                'type': 'diagnosis',
                'results': result.candidates
            }
    
    def continue_conversation(self, user_id, response):
        if user_id not in self.conversations:
            return {'error': 'Conversation not found'}
        
        conversation = self.conversations[user_id]
        result = self.agent.continue_conversation(conversation, response)
        
        if result.is_conclusive:
            # Conversation terminée
            del self.conversations[user_id]
            return {
                'type': 'diagnosis',
                'results': result.candidates
            }
        else:
            # Plus de questions
            return {
                'type': 'questions',
                'questions': result.questions
            }
```

---

## 📝 Notes Importantes

### Limitations

1. **Liste de symptômes génériques**: Basée sur une analyse médicale mais peut nécessiter des ajustements selon le domaine
2. **Formule de pénalité**: Linéaire, pourrait être affinée avec des courbes plus sophistiquées
3. **Questions discriminantes**: Dépendent de la qualité du dataset

### Recommandations

1. **Toujours vérifier** le flag `needs_disambiguation` avant de présenter les résultats
2. **Afficher clairement** l'incertitude aux utilisateurs
3. **Encourager** les utilisateurs à fournir plus de détails
4. **Ne jamais** présenter un diagnostic comme définitif sans validation médicale

### Sécurité Médicale

⚠️ **IMPORTANT**: Ce système est un outil d'aide à la décision, pas un substitut au jugement médical professionnel.

- Toujours inclure des disclaimers appropriés
- En cas d'urgence, diriger vers le 15 (SAMU)
- Ne pas utiliser pour des décisions médicales critiques sans supervision

---

## 🤝 Contribution

Pour améliorer le système:

1. **Enrichir la liste de symptômes génériques**: Analyser plus de cas réels
2. **Affiner les seuils**: Basé sur des retours d'utilisateurs médicaux
3. **Améliorer les questions**: Intégrer des arbres de décision médicaux
4. **Ajouter des métriques**: Tracer l'amélioration de la précision avec la désambiguïsation

---

## 📚 Références

- `medical_agent/core/disambiguation.py` - Détecteur de désambiguïsation
- `medical_agent/core/question_generator.py` - Générateur de questions
- `medical_agent/core/disease_predictor.py` - Prédicteur avec ajustement de confiance
- `medical_agent/core/agent.py` - Agent principal avec intégration
- `scripts/test_disambiguation.py` - Suite de tests

---

## ✅ Résumé

Le système de désambiguïsation:

1. ✅ Détecte automatiquement les symptômes génériques
2. ✅ Ajuste les confiances pour refléter l'incertitude médicale
3. ✅ Génère des questions discriminantes intelligentes
4. ✅ Supporte des conversations multi-tours
5. ✅ Améliore progressivement la précision diagnostique
6. ✅ Communique clairement l'incertitude aux utilisateurs

**Résultat**: Un système plus sûr, plus transparent et médicalement plus réaliste.
