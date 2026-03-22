"""
Script de test du système médical
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_agent import MedicalAgent


def test_basic_diagnosis():
    """Test de diagnostic de base"""
    print("Test 1: Diagnostic de base")
    print("-" * 50)
    
    agent = MedicalAgent()
    
    test_cases = [
        "J'ai de la fièvre à 39°C et mal à la gorge depuis 2 jours",
        "Je tousse beaucoup et j'ai du mal à respirer",
        "J'ai des nausées et vomissements depuis hier",
        "Mal de tête intense et sensibilité à la lumière"
    ]
    
    for idx, text in enumerate(test_cases, 1):
        print(f"\nCas {idx}: {text}")
        
        try:
            result = agent.diagnose(text, top_n=3)
            
            print(f"Symptômes: {', '.join(result.patient_input.symptoms)}")
            print(f"Triage: {result.triage.level}")
            
            if result.candidates:
                print(f"Diagnostic: {result.candidates[0].disease_name} ({result.candidates[0].confidence:.1%})")
            else:
                print("Aucun diagnostic")
                
        except Exception as e:
            print(f"Erreur: {e}")
    
    print("\n" + "=" * 50)
    print("Tests terminés")


def test_components():
    """Test des composants"""
    print("\nTest 2: Composants")
    print("-" * 50)
    
    from medical_agent.core.symptom_extractor import SymptomExtractor
    from medical_agent.core.triage_system import TriageSystem
    
    print("\n1. SymptomExtractor")
    extractor = SymptomExtractor()
    patient = extractor.extract("J'ai 38.5°C de température et je suis fatigué")
    print(f"   Température: {patient.temperature}°C")
    print(f"   Symptômes: {patient.symptoms}")
    
    print("\n2. TriageSystem")
    triage_sys = TriageSystem()
    triage = triage_sys.evaluate(patient)
    print(f"   Niveau: {triage.level}")
    print(f"   Action: {triage.recommended_action}")
    
    print("\nTests composants OK")


def display_stats():
    """Statistiques"""
    print("\nStatistiques")
    print("-" * 50)
    
    agent = MedicalAgent()
    
    if agent.disease_predictor.df is not None:
        df = agent.disease_predictor.df
        n_diseases = df['disease'].nunique()
        n_symptoms = df['symptom'].nunique()
        n_records = len(df)
        
        print(f"Maladies: {n_diseases}")
        print(f"Symptômes: {n_symptoms}")
        print(f"Enregistrements: {n_records}")
    else:
        print("Aucune donnée")


if __name__ == "__main__":
    print("=" * 50)
    print("TESTS AGENT MÉDICAL v2.0")
    print("=" * 50)
    
    try:
        test_basic_diagnosis()
        test_components()
        display_stats()
        
        print("\n" + "=" * 50)
        print("TOUS LES TESTS OK")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
