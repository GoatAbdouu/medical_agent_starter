# Architecture du Système de Désambiguïsation

## 📊 Flux de Traitement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT UTILISATEUR                                │
│              "J'ai de la fièvre, de la fatigue et mal de tête"          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SymptomExtractor                                     │
│  Extrait: ["fièvre", "fatigue", "mal de tête"]                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DisambiguationDetector                                │
│  • Analyse la spécificité des symptômes                                 │
│  • Identifie: 100% symptômes génériques                                 │
│  • Score de spécificité: 0.0                                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DiseasePredictor                                      │
│  • Prédictions initiales (ML + Règles)                                  │
│  • Lyme disease: 78%                                                     │
│  • Grippe: 75%                                                           │
│  • Mononucléose: 72%                                                     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           DisambiguationDetector.adjust_confidence()                     │
│  Ajustement avec facteur de pénalité: 0.3 + (0.7 × 0.0) = 0.3          │
│  • Lyme disease: 78% × 0.3 = 23%                                        │
│  • Grippe: 75% × 0.3 = 23%                                              │
│  • Mononucléose: 72% × 0.3 = 22%                                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DisambiguationDetector.needs_disambiguation()               │
│  Détection: ✅ OUI                                                       │
│  Raison: "Symptômes trop génériques"                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  FollowUpQuestionGenerator                               │
│  • Analyse les symptômes discriminants                                  │
│  • Trouve symptômes spécifiques à certaines maladies                    │
│  • Génère questions ciblées                                             │
│  Résultat:                                                               │
│    1. "Avez-vous des éruptions cutanées ?"                              │
│    2. "Ressentez-vous des douleurs articulaires ?"                      │
│    3. "Depuis combien de temps avez-vous ces symptômes ?"              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DiagnosisResult                                   │
│  candidates: [Lyme: 23%, Grippe: 23%, Mono: 22%]                       │
│  needs_disambiguation: True                                              │
│  disambiguation_reason: "Symptômes trop génériques..."                  │
│  symptom_specificity_score: 0.0                                         │
│  questions: [...3 questions discriminantes...]                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT UTILISATEUR                               │
│  ⚠️  DÉSAMBIGUÏSATION NÉCESSAIRE                                        │
│      Symptômes trop génériques - besoin d'informations spécifiques      │
│                                                                          │
│  Diagnostics possibles (confiances ajustées):                           │
│    1. Lyme disease: 23%                                                  │
│    2. Grippe: 23%                                                        │
│    3. Mononucléose: 22%                                                  │
│                                                                          │
│  Questions recommandées:                                                 │
│    1. Avez-vous des éruptions cutanées ?                                │
│    2. Ressentez-vous des douleurs articulaires ?                        │
│    3. Depuis combien de temps avez-vous ces symptômes ?                │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Conversation Multi-Tours

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TOUR 1: Input Initial                                                   │
│  "J'ai de la fièvre et de la fatigue"                                   │
│                                                                          │
│  → Spécificité: 0%                                                       │
│  → Confiances ajustées: ~23%                                            │
│  → Questions générées ✅                                                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               │ "Oui, j'ai des éruptions cutanées
                               │  rouges et des douleurs articulaires"
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TOUR 2: Informations Supplémentaires                                   │
│  ConversationState.add_symptoms()                                        │
│  Symptômes cumulés: ["fièvre", "fatigue", "éruptions cutanées",        │
│                      "douleurs articulaires"]                           │
│                                                                          │
│  → Spécificité: 50% (2 génériques, 2 spécifiques)                      │
│  → Confiances: ~50-65%                                                   │
│  → Lyme disease devient dominant (éruptions + articulaires)            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               │ "L'éruption est en forme d'anneau"
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TOUR 3: Détail Spécifique                                              │
│  Symptômes: + ["éruption en anneau"]                                    │
│                                                                          │
│  → Spécificité: 60%                                                      │
│  → Lyme disease: 75-85% (signe pathognomonique)                         │
│  → Diagnostic conclusif ✅                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture des Composants

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MedicalAgent                                   │
│                      (Orchestrateur Principal)                           │
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐ │
│  │ SymptomExtractor   │  │  TriageSystem      │  │  DiseasePredictor│ │
│  │                    │  │                    │  │                  │ │
│  │ Extrait symptômes  │  │ Évalue urgence     │  │ Prédit maladies  │ │
│  └────────────────────┘  └────────────────────┘  └──────────────────┘ │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │            NOUVEAU: Système de Désambiguïsation                   │ │
│  │                                                                   │ │
│  │  ┌────────────────────────────────────────────────────────────┐ │ │
│  │  │  DisambiguationDetector                                     │ │ │
│  │  │  • is_generic_symptom()                                     │ │ │
│  │  │  • calculate_symptom_specificity()                          │ │ │
│  │  │  • needs_disambiguation()                                   │ │ │
│  │  │  • adjust_confidence_for_genericity()                       │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │  ┌────────────────────────────────────────────────────────────┐ │ │
│  │  │  FollowUpQuestionGenerator                                  │ │ │
│  │  │  • find_discriminating_symptoms()                           │ │ │
│  │  │  • generate_questions_for_disambiguation()                  │ │ │
│  │  │  • generate_all_follow_up_questions()                       │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Logique de Décision

```
Symptômes détectés
        │
        ▼
┌───────────────────┐
│ Calculer          │
│ spécificité       │◄─── Liste de symptômes génériques
└─────────┬─────────┘     (fièvre, fatigue, etc.)
          │
          ▼
    ┌─────────────────────────────┐
    │ Spécificité = 0% ?          │
    └──────┬──────────────┬───────┘
           │ OUI          │ NON
           ▼              ▼
    ┌──────────────┐  ┌────────────────────┐
    │ Tous         │  │ Spécificité < 30% │
    │ génériques   │  │ ET confiance > 50%?│
    └──────┬───────┘  └──────┬─────────────┘
           │                 │
           │ OUI             │ OUI
           ▼                 ▼
    ┌──────────────────────────────────┐
    │  DÉSAMBIGUÏSATION NÉCESSAIRE     │
    │                                  │
    │  Actions:                        │
    │  1. Ajuster confiances ↓         │
    │  2. Générer questions            │
    │  3. Marquer besoin désambig.     │
    └──────────────────────────────────┘
```

## 📊 Calcul de la Confiance Ajustée

```
Formule: adjusted_confidence = original × penalty_factor

Où: penalty_factor = 0.3 + (0.7 × specificity_score)

┌─────────────────┬──────────────────┬─────────────────────────┐
│  Spécificité    │  Facteur Pén.    │  Exemple (original 80%) │
├─────────────────┼──────────────────┼─────────────────────────┤
│  0% (tous gén.) │  0.3             │  80% × 0.3 = 24%       │
│  25%            │  0.475           │  80% × 0.475 = 38%     │
│  50%            │  0.65            │  80% × 0.65 = 52%      │
│  75%            │  0.825           │  80% × 0.825 = 66%     │
│  100% (tous sp.)│  1.0             │  80% × 1.0 = 80%       │
└─────────────────┴──────────────────┴─────────────────────────┘

Graphique:
Confiance │
Ajustée   │
    100%  │                              ────────
          │                        ──────
     75%  │                  ──────
          │            ──────
     50%  │      ──────
          │──────
     25%  │
          │
      0%  └──────────────────────────────────────
          0%    25%    50%    75%   100%
                Spécificité des Symptômes
```

## 🔍 Exemple Concret: Lyme Disease

```
Symptômes de Lyme Disease dans le dataset:
┌────────────────────────────────────────┐
│ • fièvre (générique)                   │
│ • fatigue (générique)                  │
│ • mal de tête (générique)              │
│ • douleurs articulaires (spécifique)   │
│ • éruption cutanée (spécifique)        │
│ • éruption en anneau (très spécifique) │
└────────────────────────────────────────┘

Scénario 1: Input générique
Input: "fièvre, fatigue, mal de tête"
→ Spécificité: 0% (3/3 génériques)
→ Lyme: 23% (78% → ajusté)
→ Désambiguïsation: OUI
→ Questions: "Avez-vous des éruptions cutanées?"

Scénario 2: Ajout symptôme spécifique
Input: "fièvre, fatigue, éruptions cutanées"
→ Spécificité: 33% (1/3 spécifique)
→ Lyme: 42% (78% × 0.54)
→ Lyme devient plus probable
→ Question: "Les éruptions sont-elles en forme d'anneau?"

Scénario 3: Symptôme pathognomonique
Input: "éruption en anneau, douleurs articulaires"
→ Spécificité: 100% (2/2 spécifiques)
→ Lyme: 85% (confiance haute justifiée)
→ Désambiguïsation: NON
→ Diagnostic conclusif ✅
```

## 🎪 Questions Discriminantes

Comment le système choisit les questions:

```
Candidats:
1. Lyme disease    - Symptômes: [fièvre, fatigue, éruption*, articulaires*]
2. Grippe          - Symptômes: [fièvre, fatigue, toux, courbatures]
3. Mononucléose    - Symptômes: [fièvre, fatigue, gorge*, ganglions*]

* = symptôme non mentionné par le patient

Analyse discriminante:
┌──────────────────────┬──────────────┬─────────────────┐
│ Symptôme             │ Présent dans │ Pouvoir Discr.  │
├──────────────────────┼──────────────┼─────────────────┤
│ éruption cutanée     │ Lyme only    │ 1.0 (parfait)   │
│ douleurs articulaires│ Lyme only    │ 1.0 (parfait)   │
│ mal de gorge         │ Mono only    │ 1.0 (parfait)   │
│ toux                 │ Grippe only  │ 1.0 (parfait)   │
│ courbatures          │ Grippe, Lyme │ 0.5 (moyen)     │
│ fièvre               │ Tous         │ 0.0 (aucun)     │
└──────────────────────┴──────────────┴─────────────────┘

Questions générées (triées par pouvoir discriminant):
1. "Avez-vous des éruptions cutanées ?" (discrimine Lyme)
2. "Avez-vous mal à la gorge ?" (discrimine Mono)
3. "Avez-vous de la toux ?" (discrimine Grippe)
```

## 🔄 Boucle de Raffinement

```
    ┌──────────────────────────────────────────┐
    │  État initial: Symptômes génériques      │
    │  Confiances basses, ambiguïté élevée     │
    └───────────────┬──────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │  Générer questions discriminantes        │
    └───────────────┬──────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │  Utilisateur répond                      │
    └───────────────┬──────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │  Ajouter nouveaux symptômes              │
    │  Recalculer spécificité                  │
    └───────────────┬──────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │  Réajuster confiances                    │
    │  (pénalité réduite si spécificité ↑)     │
    └───────────────┬──────────────────────────┘
                    │
                    ▼
           ┌────────┴──────────┐
           │                   │
           ▼                   ▼
    ┌──────────┐      ┌──────────────┐
    │ Conclusif│      │ Encore ambigu│
    │    ✅    │      │      ↻       │
    └──────────┘      └──────┬───────┘
                             │
                             └──► Nouvelles questions
```

Cette architecture garantit un diagnostic progressivement plus précis et médicalement responsable!
