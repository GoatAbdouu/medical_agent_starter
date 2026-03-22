"""
Script d'entraînement du modèle Deep Learning
Exécutez ce script pour entraîner le réseau de neurones sur le dataset
"""
import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("🧠 Entraînement du modèle Deep Learning - Agent Médical")
    print("=" * 60)
    
    # Vérifier PyTorch
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        print("❌ PyTorch n'est pas installé!")
        print("   Installez-le avec: pip install torch")
        return
    
    # Importer le prédicteur
    try:
        from medical_agent.core.deep_learning_predictor import DeepLearningPredictor
        print("✅ Module DeepLearningPredictor importé")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return
    
    # Initialiser et entraîner
    print("\n📊 Chargement du dataset...")
    try:
        predictor = DeepLearningPredictor()
        print(f"   Dataset chargé: {len(predictor.df)} lignes")
        print(f"   Maladies uniques: {predictor.df['disease'].nunique()}")
        print(f"   Symptômes uniques: {predictor.df['symptom'].nunique()}")
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return
    
    # Entraînement
    print("\n🏋️ Début de l'entraînement...")
    print("   Cela peut prendre quelques minutes...\n")
    
    try:
        predictor.train(
            epochs=30,  # Réduit pour un entraînement plus rapide
            batch_size=128,
            learning_rate=0.001
        )
        print("\n✅ Entraînement terminé avec succès!")
        print(f"   Modèle sauvegardé: {predictor.model_path}")
    except Exception as e:
        print(f"\n❌ Erreur d'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test rapide
    print("\n🧪 Test du modèle...")
    test_symptoms = ["fièvre", "toux", "fatigue"]
    print(f"   Symptômes de test: {test_symptoms}")
    
    try:
        predictions = predictor.predict(test_symptoms, top_n=5)
        print("\n   Prédictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"   {i}. {pred.disease_name}: {pred.confidence*100:.1f}%")
    except Exception as e:
        print(f"   Erreur de test: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Terminé! Vous pouvez maintenant lancer l'application.")
    print("   streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
