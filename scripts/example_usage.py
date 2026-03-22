"""
Exemple d'utilisation du système de désambiguïsation
"""
import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medical_agent import MedicalAgent


def example_basic():
    """Exemple basique avec symptômes génériques"""
    print("\n" + "="*80)
    print("EXEMPLE 1: Symptômes génériques")
    print("="*80 + "\n")
    
    agent = MedicalAgent()
    
    # Cas avec symptômes très génériques
    text = "J'ai de la fièvre, de la fatigue et mal à la tête"
    print(f"Patient dit: \"{text}\"\n")
    
    result = agent.diagnose(text, top_n=5)
    
    # Vérifier si désambiguïsation nécessaire
    if result.needs_disambiguation:
        print("⚠️  DÉSAMBIGUÏSATION NÉCESSAIRE")
        print(f"    Raison: {result.disambiguation_reason}")
        print(f"    Spécificité des symptômes: {result.symptom_specificity_score:.0%}\n")
    
    # Afficher les diagnostics avec confiances ajustées
    print("Diagnostics possibles (confiances ajustées):")
    for i, candidate in enumerate(result.candidates[:3], 1):
        print(f"  {i}. {candidate.disease_name}: {candidate.confidence:.1%}")
    
    # Afficher les questions de suivi
    if result.questions:
        print("\nQuestions recommandées pour affiner le diagnostic:")
        for i, question in enumerate(result.questions[:3], 1):
            print(f"  {i}. {question}")


def example_specific():
    """Exemple avec symptômes spécifiques"""
    print("\n" + "="*80)
    print("EXEMPLE 2: Symptômes spécifiques")
    print("="*80 + "\n")
    
    agent = MedicalAgent()
    
    # Cas avec symptômes spécifiques
    text = "J'ai des éruptions cutanées rouges en forme d'anneau, des douleurs articulaires et une fatigue"
    print(f"Patient dit: \"{text}\"\n")
    
    result = agent.diagnose(text, top_n=5)
    
    # Vérifier si désambiguïsation nécessaire
    if result.needs_disambiguation:
        print("⚠️  DÉSAMBIGUÏSATION NÉCESSAIRE")
        print(f"    Raison: {result.disambiguation_reason}")
    else:
        print("✅ DIAGNOSTIC CONCLUSIF")
        print(f"    Spécificité des symptômes: {result.symptom_specificity_score:.0%}\n")
    
    # Afficher les diagnostics
    print("Diagnostics possibles:")
    for i, candidate in enumerate(result.candidates[:3], 1):
        print(f"  {i}. {candidate.disease_name}: {candidate.confidence:.1%}")


def example_comparison():
    """Comparaison côte à côte: générique vs spécifique"""
    print("\n" + "="*80)
    print("EXEMPLE 3: Comparaison - Avant/Après ajout de détails")
    print("="*80 + "\n")
    
    agent = MedicalAgent()
    
    # Cas 1: Symptômes génériques
    text1 = "J'ai de la fièvre et de la fatigue"
    result1 = agent.diagnose(text1, top_n=3)
    
    print(f"📝 Phase 1: \"{text1}\"")
    print(f"   Spécificité: {result1.symptom_specificity_score:.0%}")
    print(f"   Désambiguïsation: {'OUI' if result1.needs_disambiguation else 'NON'}")
    print("\n   Top 3 diagnostics:")
    for i, candidate in enumerate(result1.candidates[:3], 1):
        print(f"     {i}. {candidate.disease_name}: {candidate.confidence:.1%}")
    
    print("\n" + "-"*80 + "\n")
    
    # Cas 2: Ajout de symptômes spécifiques
    text2 = "J'ai de la fièvre et de la fatigue, plus des éruptions cutanées rouges et des douleurs articulaires"
    result2 = agent.diagnose(text2, top_n=3)
    
    print(f"📝 Phase 2: \"{text2}\"")
    print(f"   Spécificité: {result2.symptom_specificity_score:.0%}")
    print(f"   Désambiguïsation: {'OUI' if result2.needs_disambiguation else 'NON'}")
    print("\n   Top 3 diagnostics:")
    for i, candidate in enumerate(result2.candidates[:3], 1):
        print(f"     {i}. {candidate.disease_name}: {candidate.confidence:.1%}")
    
    # Analyse de l'amélioration
    print("\n📊 Amélioration:")
    print(f"   Spécificité: {result1.symptom_specificity_score:.0%} → {result2.symptom_specificity_score:.0%}")
    if result1.candidates and result2.candidates:
        conf_improvement = (result2.candidates[0].confidence - result1.candidates[0].confidence) * 100
        print(f"   Confiance top 1: +{conf_improvement:.1f} points de pourcentage")


def example_interactive():
    """Exemple interactif"""
    print("\n" + "="*80)
    print("EXEMPLE 4: Session interactive")
    print("="*80 + "\n")
    
    agent = MedicalAgent()
    
    print("Décrivez vos symptômes (ou tapez 'exit' pour quitter):")
    print("Exemples:")
    print("  - J'ai de la fièvre et de la fatigue")
    print("  - J'ai mal à la gorge, difficultés à avaler, et des ganglions gonflés")
    print()
    
    while True:
        user_input = input("Vos symptômes: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nAu revoir!")
            break
        
        if not user_input:
            continue
        
        print()
        result = agent.diagnose(user_input, top_n=5)
        
        # Afficher le statut
        if result.needs_disambiguation:
            print(f"⚠️  {result.disambiguation_reason}")
            print(f"    Spécificité: {result.symptom_specificity_score:.0%}\n")
        else:
            print(f"✅ Diagnostic basé sur des symptômes suffisamment spécifiques")
            print(f"    Spécificité: {result.symptom_specificity_score:.0%}\n")
        
        # Afficher les symptômes détectés
        print("Symptômes détectés:")
        for symptom in result.patient_input.symptoms:
            print(f"  • {symptom}")
        print()
        
        # Afficher les diagnostics
        print("Top 5 diagnostics possibles:")
        for i, candidate in enumerate(result.candidates[:5], 1):
            print(f"  {i}. {candidate.disease_name}: {candidate.confidence:.1%}")
        print()
        
        # Afficher les questions de suivi
        if result.questions:
            print("Questions recommandées:")
            for i, question in enumerate(result.questions[:3], 1):
                print(f"  {i}. {question}")
            print()
        
        # Afficher le triage si urgent
        if result.is_urgent:
            print("⚠️  SITUATION URGENTE DÉTECTÉE")
            print(f"    Niveau: {result.triage.level.upper()}")
            print(f"    Action: {result.triage.recommended_action}")
            print()
        
        print("-"*80 + "\n")


def main():
    """Fonction principale"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║               EXEMPLES D'UTILISATION - SYSTÈME DE DÉSAMBIGUÏSATION       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Exécuter les exemples
        example_basic()
        input("\n⏸️  Appuyez sur Entrée pour l'exemple suivant...")
        
        example_specific()
        input("\n⏸️  Appuyez sur Entrée pour l'exemple suivant...")
        
        example_comparison()
        input("\n⏸️  Appuyez sur Entrée pour la session interactive...")
        
        example_interactive()
        
    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
