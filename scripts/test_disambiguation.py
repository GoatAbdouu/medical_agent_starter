"""
Script de test pour le système de désambiguïsation
Démontre comment le système gère les symptômes génériques
"""
import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medical_agent import MedicalAgent
from medical_agent.config.settings import settings


def print_separator(title=""):
    """Affiche un séparateur visuel"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    print()


def print_diagnosis_result(result, show_all=True):
    """Affiche les résultats du diagnostic de manière formatée"""
    
    print_separator("RÉSULTATS DU DIAGNOSTIC")
    
    # Informations sur la désambiguïsation
    if result.needs_disambiguation:
        print("🔴 DÉSAMBIGUÏSATION NÉCESSAIRE")
        print(f"   Raison: {result.disambiguation_reason}")
        print(f"   Spécificité des symptômes: {result.symptom_specificity_score:.1%}")
    else:
        print("🟢 DIAGNOSTIC CONCLUSIF")
        print(f"   Spécificité des symptômes: {result.symptom_specificity_score:.1%}")
    
    print()
    
    # Symptômes détectés
    print("📋 SYMPTÔMES DÉTECTÉS:")
    for symptom in result.patient_input.symptoms:
        print(f"   • {symptom}")
    
    if result.patient_input.temperature:
        print(f"   🌡️  Température: {result.patient_input.temperature}°C")
    
    print()
    
    # Diagnostics candidats
    if result.candidates:
        print("🏥 DIAGNOSTICS POSSIBLES:")
        print()
        
        for i, candidate in enumerate(result.candidates, 1):
            confidence_percent = candidate.confidence * 100
            
            # Barre de confiance visuelle
            bar_length = int(confidence_percent / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            print(f"   {i}. {candidate.disease_name.upper()}")
            print(f"      Confiance: [{bar}] {confidence_percent:.1f}%")
            
            if show_all and candidate.matched_symptoms:
                print(f"      Symptômes correspondants: {', '.join(candidate.matched_symptoms[:3])}")
            print()
    else:
        print("⚠️  Aucun diagnostic trouvé")
        print()
    
    # Triage
    if result.triage:
        print(f"🚨 TRIAGE: {result.triage.level.upper()}")
        print(f"   {result.triage.recommended_action}")
        print()
    
    # Questions de suivi
    if result.questions:
        print("❓ QUESTIONS DE SUIVI:")
        for i, question in enumerate(result.questions, 1):
            print(f"   {i}. {question}")
        print()
    
    # Recommandations
    if result.recommendations and show_all:
        print("💡 RECOMMANDATIONS:")
        for rec in result.recommendations:
            print(f"   • {rec}")
        print()


def test_case_1_generic_symptoms():
    """
    Test 1: Symptômes très génériques
    Attendu: Confiances faibles + besoin de désambiguïsation
    """
    print_separator("TEST 1: Symptômes Génériques (Fièvre, Fatigue, Mal de tête)")
    
    agent = MedicalAgent()
    
    # Cas avec symptômes très génériques
    text = "J'ai de la fièvre, de la fatigue et mal à la tête"
    
    print(f"💬 Input patient: \"{text}\"")
    print()
    
    result = agent.diagnose(text, top_n=5)
    print_diagnosis_result(result)
    
    return result


def test_case_2_specific_symptoms():
    """
    Test 2: Symptômes spécifiques
    Attendu: Confiances normales + pas de désambiguïsation
    """
    print_separator("TEST 2: Symptômes Spécifiques")
    
    agent = MedicalAgent()
    
    # Cas avec symptômes spécifiques
    text = "J'ai des douleurs thoraciques oppressantes, essoufflement au repos, et sueurs froides"
    
    print(f"💬 Input patient: \"{text}\"")
    print()
    
    result = agent.diagnose(text, top_n=5)
    print_diagnosis_result(result)
    
    return result


def test_case_3_mixed_symptoms():
    """
    Test 3: Mélange de symptômes génériques et spécifiques
    Attendu: Confiances modérées
    """
    print_separator("TEST 3: Symptômes Mixtes")
    
    agent = MedicalAgent()
    
    # Cas avec mélange
    text = "J'ai de la fièvre à 38.5°C, des éruptions cutanées rouges sur le torse, et de la fatigue"
    
    print(f"💬 Input patient: \"{text}\"")
    print()
    
    result = agent.diagnose(text, top_n=5)
    print_diagnosis_result(result)
    
    return result


def test_case_4_ambiguous_diseases():
    """
    Test 4: Symptômes menant à plusieurs diagnostics similaires
    Attendu: Détection d'ambiguïté + questions discriminantes
    """
    print_separator("TEST 4: Diagnostics Ambigus")
    
    agent = MedicalAgent()
    
    # Cas ambigu (symptômes respiratoires génériques)
    text = "J'ai de la toux, de la fièvre et du mal à respirer"
    
    print(f"💬 Input patient: \"{text}\"")
    print()
    
    result = agent.diagnose(text, top_n=5)
    print_diagnosis_result(result)
    
    return result


def compare_before_after():
    """
    Compare le comportement avant/après pour le même cas
    """
    print_separator("COMPARAISON: Impact du Système de Désambiguïsation")
    
    agent = MedicalAgent()
    
    text = "J'ai de la fièvre, de la fatigue et mal à la tête"
    
    print(f"💬 Input patient: \"{text}\"")
    print()
    
    result = agent.diagnose(text, top_n=5)
    
    print("📊 ANALYSE DU SYSTÈME:")
    print()
    
    # Analyser les symptômes
    generic_symptoms = agent.disambiguation_detector.get_generic_symptoms_from_input(
        result.patient_input
    )
    specific_symptoms = agent.disambiguation_detector.get_specific_symptoms_from_input(
        result.patient_input
    )
    
    print(f"   Symptômes génériques: {generic_symptoms}")
    print(f"   Symptômes spécifiques: {specific_symptoms}")
    print(f"   Score de spécificité: {result.symptom_specificity_score:.1%}")
    print()
    
    print("   Interprétation:")
    if result.needs_disambiguation:
        print(f"   ✅ Le système a détecté le problème: {result.disambiguation_reason}")
        print(f"   ✅ Les confiances ont été ajustées à la baisse")
        print(f"   ✅ Des questions discriminantes ont été générées")
    else:
        print(f"   ℹ️  Pas de désambiguïsation nécessaire")
    
    print()
    print_diagnosis_result(result, show_all=False)


def main():
    """Fonction principale"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║          SYSTÈME DE DÉSAMBIGUÏSATION - TESTS DE VALIDATION               ║
    ║                                                                           ║
    ║  Ce script teste le nouveau système qui détecte les symptômes génériques ║
    ║  et ajuste les confiances pour éviter les prédictions sur-confiantes.   ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Vérifier que les fichiers existent
        if not settings.DATASET_PATH.exists():
            print(f"❌ Erreur: Dataset non trouvé à {settings.DATASET_PATH}")
            return
        
        print(f"✅ Dataset trouvé: {settings.DATASET_PATH}")
        print()
        
        # Exécuter les tests
        test_case_1_generic_symptoms()
        input("\n⏸️  Appuyez sur Entrée pour continuer...")
        
        test_case_2_specific_symptoms()
        input("\n⏸️  Appuyez sur Entrée pour continuer...")
        
        test_case_3_mixed_symptoms()
        input("\n⏸️  Appuyez sur Entrée pour continuer...")
        
        test_case_4_ambiguous_diseases()
        input("\n⏸️  Appuyez sur Entrée pour continuer...")
        
        compare_before_after()
        
        print_separator("TESTS TERMINÉS")
        print("""
        ✅ Tous les tests ont été exécutés avec succès!
        
        Le système de désambiguïsation:
        - Détecte les symptômes génériques
        - Ajuste les confiances à la baisse quand nécessaire
        - Génère des questions discriminantes
        - Améliore la précision diagnostique
        """)
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
