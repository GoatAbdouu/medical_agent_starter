"""
Application Streamlit - Agent Médical Vision IA
Avec support Ensemble Deep Learning, VLM et reconnaissance vocale
"""
import streamlit as st
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from medical_agent import MedicalAgent
from medical_agent.config.settings import settings
from medical_agent.core.system_prompt import SYSTEM_PROMPT

# Import du classificateur de maladies cutanées (optionnel)
try:
    from medical_agent.core.skin_disease_classifier import SkinDiseaseClassifier
    SKIN_CLASSIFIER_AVAILABLE = True
except ImportError:
    SKIN_CLASSIFIER_AVAILABLE = False

# Import de l'ensemble classifier (optionnel)
try:
    from medical_agent.core.ensemble_classifier import EnsembleClassifier
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

# Import du VLM explainer (optionnel)
try:
    from medical_agent.core.vlm_explainer import VLMExplainer
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

# Import des services vocaux (optionnel)
try:
    from medical_agent.services.voice_recognition import (
        VoiceRecognitionService, 
        StreamlitVoiceRecorder,
        check_voice_dependencies
    )
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

st.set_page_config(
    page_title="Agent Médical",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def initialize_agent():
    """Initialise l'agent médical"""
    try:
        settings.ensure_directories()
        agent = MedicalAgent()
        return agent, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def initialize_voice_service():
    """Initialise le service de reconnaissance vocale"""
    if not VOICE_AVAILABLE:
        return None, "Service vocal non disponible"
    try:
        service = VoiceRecognitionService(backend="google", language="fr-FR")
        return service, None
    except Exception as e:
        return None, str(e)

def transcribe_audio(audio_bytes, voice_service):
    """Transcrit l'audio en texte"""
    if not voice_service:
        return None, "Service vocal non initialisé"
    
    # Sauvegarder temporairement le fichier audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = Path(tmp_file.name)
    
    try:
        text, error = voice_service.transcribe_audio_file(tmp_path)
        return text, error
    finally:
        tmp_path.unlink(missing_ok=True)

@st.cache_resource
def initialize_vlm_explainer():
    """Initialise le VLM explainer.

    Tries BLIP first; falls back to the template-based system if BLIP is not
    installed or cannot be loaded (e.g., no internet access or insufficient RAM).
    """
    if not VLM_AVAILABLE:
        return None
    try:
        explainer = VLMExplainer(use_blip=True)  # BLIP if available; template fallback otherwise
        return explainer
    except Exception:
        return None

def main():
    """Application principale"""

    st.title("🏥 Agent Médical Vision IA")
    st.caption("Ensemble Deep Learning (EfficientNet-B3 + MobileNetV2 + ResNet50) + VLM Explanation")

    agent, error = initialize_agent()
    voice_service, voice_error = initialize_voice_service()
    
    if error:
        st.error(f"Erreur: {error}")
        st.info("Vérifiez que le fichier data/cleaned_data.csv existe")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informations")

        # Afficher le mode actuel
        if hasattr(agent, 'use_deep_learning') and agent.use_deep_learning:
            st.success("🧠 Mode Deep Learning actif")
        else:
            st.info("📊 Mode règles classique")

        if ENSEMBLE_AVAILABLE:
            st.success("🤖 Ensemble IA disponible (EfficientNet-B3 + MobileNetV2 + ResNet50)")
        else:
            st.warning("🤖 Ensemble IA non disponible")

        if VLM_AVAILABLE:
            st.success("💬 Explication VLM disponible")
        else:
            st.info("💬 Explication VLM non disponible")

        if VOICE_AVAILABLE and voice_service:
            st.success("🎤 Reconnaissance vocale disponible")
        else:
            st.warning("🎤 Reconnaissance vocale non disponible")

        st.markdown("""
        ### Comment utiliser

        1. **Texte**: Décrivez vos symptômes
        2. **Voix**: Ou utilisez le microphone 🎤
        3. Cliquez sur **Analyser**
        4. Consultez les résultats

        ### Exemples
        - "J'ai de la fièvre à 39°C et mal à la gorge"
        - "Je tousse et j'ai du mal à respirer"
        - "J'ai des nausées et vomissements"

        ### ⚠️ Avertissement
        Outil informatif uniquement.
        Consultez un professionnel de santé.
        """)

        st.divider()

        # Show abbreviated system prompt context
        with st.expander("🤖 Contexte IA (System Prompt)", expanded=False):
            # Display first ~500 chars of the system prompt
            preview = SYSTEM_PROMPT[:500].strip()
            st.caption(preview + "…")

        st.divider()

        if agent and agent.disease_predictor.df is not None:
            st.markdown("### 📈 Statistiques")
            n_diseases = agent.disease_predictor.df['disease'].nunique()
            n_symptoms = agent.disease_predictor.df['symptom'].nunique()
            st.metric("Maladies", n_diseases)
            st.metric("Symptômes", n_symptoms)
    
    # Zone principale : deux onglets de premier niveau
    symptom_tab, skin_tab = st.tabs(["🩺 Diagnostic Symptômes", "📷 Diagnostic Peau"])

    # ------------------------------------------------------------------
    # Onglet 1 : Diagnostic par symptômes (logique existante inchangée)
    # ------------------------------------------------------------------
    with symptom_tab:
        col1, col2 = st.columns([2, 1])
    
        with col1:
            st.header("📝 Décrivez vos symptômes")
            
            # Onglets pour texte ou voix
            input_tab, voice_tab = st.tabs(["✍️ Texte", "🎤 Voix"])
            
            user_input = ""
            
            with input_tab:
                user_input = st.text_area(
                    "Symptômes",
                    height=150,
                    placeholder="Exemple: J'ai de la fièvre à 39°C, mal de tête et fatigue depuis 2 jours",
                    key="text_input"
                )
            
            with voice_tab:
                if VOICE_AVAILABLE:
                    st.info("🎤 Cliquez sur le bouton ci-dessous pour enregistrer votre voix")
                    
                    # Utiliser st.audio_input si disponible (Streamlit >= 1.33)
                    if hasattr(st, 'audio_input'):
                        audio_data = st.audio_input(
                            "Enregistrez vos symptômes",
                            key="voice_input"
                        )
                        
                        if audio_data is not None:
                            st.audio(audio_data)
                            
                            if st.button("🔄 Transcrire l'audio", key="transcribe_btn"):
                                with st.spinner("Transcription en cours..."):
                                    audio_bytes = audio_data.getvalue()
                                    text, error = transcribe_audio(audio_bytes, voice_service)
                                    
                                    if error:
                                        st.error(f"Erreur de transcription: {error}")
                                    elif text:
                                        st.success("Transcription réussie!")
                                        st.session_state['transcribed_text'] = text
                                        st.info(f"**Texte reconnu:** {text}")
                    else:
                        # Fallback pour versions plus anciennes de Streamlit
                        uploaded_audio = st.file_uploader(
                            "📁 Ou uploadez un fichier audio (WAV, MP3)",
                            type=["wav", "mp3", "ogg", "m4a"],
                            key="audio_upload"
                        )
                        
                        if uploaded_audio is not None:
                            st.audio(uploaded_audio)
                            
                            if st.button("🔄 Transcrire l'audio", key="transcribe_upload_btn"):
                                with st.spinner("Transcription en cours..."):
                                    audio_bytes = uploaded_audio.getvalue()
                                    text, error = transcribe_audio(audio_bytes, voice_service)
                                    
                                    if error:
                                        st.error(f"Erreur de transcription: {error}")
                                    elif text:
                                        st.success("Transcription réussie!")
                                        st.session_state['transcribed_text'] = text
                                        st.info(f"**Texte reconnu:** {text}")
                else:
                    st.warning("""
                    🎤 La reconnaissance vocale n'est pas disponible.
                    
                    Pour l'activer, installez les dépendances:
                    ```
                    pip install SpeechRecognition
                    ```
                    """)
            
            # Utiliser le texte transcrit s'il existe
            if 'transcribed_text' in st.session_state and st.session_state['transcribed_text']:
                if not user_input:
                    user_input = st.session_state['transcribed_text']
                    st.info(f"📝 Utilisation du texte transcrit: *{user_input}*")
            
            analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)
        
        with col2:
            st.header("Options")
            
            top_n = st.slider(
                "Nombre de diagnostics",
                min_value=1,
                max_value=10,
                value=5
            )
            
            show_details = st.checkbox("Détails techniques", value=False)
        
        # Traitement
        if analyze_button and user_input.strip():
            
            with st.spinner("Analyse en cours..."):
                try:
                    result = agent.diagnose(user_input, top_n=top_n)
                    
                    st.success("Analyse terminée")
                    
                    # Afficher l'avertissement de désambiguïsation si nécessaire
                    if result.needs_disambiguation:
                        st.warning(f"""
                        ⚠️ **Désambiguïsation nécessaire**
                        
                        {result.disambiguation_reason}
                        
                        **Spécificité des symptômes:** {result.symptom_specificity_score:.0%}
                        
                        ℹ️ Les confiances affichées ont été ajustées pour refléter l'incertitude.
                        Veuillez répondre aux questions ci-dessous pour un diagnostic plus précis.
                        """)
                    
                    # Triage
                    if result.triage:
                        st.header("Évaluation de l'urgence")
                        
                        level_colors = {
                            "critique": "red",
                            "urgent": "orange",
                            "normal": "yellow",
                            "léger": "green"
                        }
                        
                        color = level_colors.get(result.triage.level, "gray")
                        
                        st.markdown(f"""
                        **Niveau:** :{color}[{result.triage.level.upper()}]
                        
                        **Action:** {result.triage.recommended_action}
                        
                        **Raison:** {result.triage.reason}
                        """)
                        
                        if result.triage.red_flags:
                            st.warning("Drapeaux rouges détectés:")
                            for flag in result.triage.red_flags:
                                st.markdown(f"- {flag}")
                    
                    st.divider()
                    
                    # Symptômes
                    if result.patient_input.symptoms:
                        st.header("Symptômes détectés")
                        
                        cols = st.columns(3)
                        for idx, symptom in enumerate(result.patient_input.symptoms):
                            with cols[idx % 3]:
                                st.info(symptom)
                        
                        info_cols = st.columns(3)
                        
                        with info_cols[0]:
                            if result.patient_input.temperature:
                                st.metric(
                                    "Température",
                                    f"{result.patient_input.temperature}°C"
                                )
                        
                        with info_cols[1]:
                            if result.patient_input.intensity:
                                intensity_labels = {
                                    "mild": "Léger",
                                    "moderate": "Modéré",
                                    "severe": "Sévère"
                                }
                                st.metric(
                                    "Intensité",
                                    intensity_labels.get(result.patient_input.intensity, "N/A")
                                )
                        
                        with info_cols[2]:
                            if result.patient_input.onset:
                                st.metric("Durée", result.patient_input.onset)
                    
                    st.divider()
                    
                    # Diagnostics
                    if result.candidates:
                        st.header("Diagnostics possibles")
                        
                        for idx, candidate in enumerate(result.candidates[:top_n], 1):
                            confidence_percent = int(candidate.confidence * 100)
                            
                            with st.expander(
                                f"#{idx} - {candidate.disease_name.title()} ({confidence_percent}%)",
                                expanded=(idx == 1)
                            ):
                                st.progress(candidate.confidence)
                                
                                st.markdown(f"**Confiance:** {confidence_percent}%")
                                
                                if candidate.matched_symptoms:
                                    st.markdown("**Symptômes correspondants:**")
                                    for symp in candidate.matched_symptoms:
                                        st.markdown(f"- {symp}")
                    else:
                        st.warning("Aucun diagnostic trouvé")
                    
                    st.divider()
                    
                    # Recommandations
                    if result.recommendations:
                        st.header("Recommandations")
                        
                        for recommendation in result.recommendations:
                            st.info(recommendation)
                    
                    # Questions de suivi
                    if result.questions:
                        st.header("Questions complémentaires")
                        
                        for question in result.questions:
                            st.markdown(f"- {question}")
                    
                    # Détails techniques
                    if show_details:
                        st.divider()
                        st.header("Détails techniques")
                        
                        with st.expander("Voir détails"):
                            st.json({
                                "texte_brut": result.patient_input.raw_text,
                                "symptomes": result.patient_input.symptoms,
                                "facteurs_risque": result.patient_input.risk_factors,
                                "valeurs_mesurees": result.patient_input.measured_values,
                                "niveau_triage": result.triage.level if result.triage else None,
                                "nombre_candidats": len(result.candidates)
                            })
                    
                except Exception as e:
                    st.error(f"Erreur: {e}")
                    if show_details:
                        st.exception(e)
        
        elif analyze_button:
            st.warning("Veuillez décrire vos symptômes")

    # ------------------------------------------------------------------
    # Onglet 2 : Diagnostic par image de peau (Ensemble + VLM)
    # ------------------------------------------------------------------
    with skin_tab:
        if not SKIN_CLASSIFIER_AVAILABLE:
            st.error(
                "Le module de classification cutanée n'est pas disponible. "
                "Installez les dépendances : `pip install torchvision Pillow`"
            )
        else:
            # Check if at least one model (ensemble or single) is available
            ensemble_model_paths = [
                project_root / "models" / "efficientnet_skin.pth",
                project_root / "models" / "mobilenet_skin.pth",
                project_root / "models" / "resnet_skin.pth",
            ]
            single_model_path = project_root / "models" / "skin_disease_model.pth"
            ensemble_models_found = [p for p in ensemble_model_paths if p.exists()]
            any_model_exists = bool(ensemble_models_found) or single_model_path.exists()

            if not any_model_exists:
                st.warning(
                    "⚠️ Aucun modèle de classification cutanée trouvé. "
                    "Entraînez l'ensemble avec la commande :"
                )
                st.code(
                    'python scripts/train_ensemble.py '
                    '--data_dir "chemin/vers/IMG_CLASSES"',
                    language="bash",
                )
                st.info(
                    "Ou entraînez le modèle simple MobileNetV2 :\n"
                    '`python scripts/train_skin_classifier.py --data_dir "chemin/vers/IMG_CLASSES"`'
                )
            else:
                # Show which models are loaded
                if ensemble_models_found:
                    model_names = []
                    name_map = {
                        "efficientnet_skin.pth": "EfficientNet-B3",
                        "mobilenet_skin.pth": "MobileNetV2",
                        "resnet_skin.pth": "ResNet50",
                    }
                    for p in ensemble_models_found:
                        model_names.append(name_map.get(p.name, p.name))
                    st.success(f"✅ Modèles d'ensemble chargés : {', '.join(model_names)}")
                elif single_model_path.exists():
                    st.info("ℹ️ Modèle MobileNetV2 simple utilisé (aucun modèle d'ensemble trouvé)")

                skin_col1, skin_col2 = st.columns([2, 1])

                with skin_col1:
                    st.header("🖼️ Image à analyser")
                    upload_sub, camera_sub = st.tabs(["📁 Upload Image", "📸 Caméra"])

                    skin_image = None

                    with upload_sub:
                        uploaded_file = st.file_uploader(
                            "Choisissez une image",
                            type=["jpg", "jpeg", "png", "bmp", "webp"],
                            key="skin_upload",
                        )
                        if uploaded_file is not None:
                            from PIL import Image as PILImage
                            skin_image = PILImage.open(uploaded_file).convert("RGB")
                            st.image(skin_image, caption="Image uploadée", use_container_width=True)

                    with camera_sub:
                        if hasattr(st, "camera_input"):
                            camera_data = st.camera_input(
                                "Prenez une photo de la lésion cutanée",
                                key="skin_camera",
                            )
                            if camera_data is not None:
                                from PIL import Image as PILImage
                                import io
                                skin_image = PILImage.open(io.BytesIO(camera_data.getvalue())).convert("RGB")
                                st.image(skin_image, caption="Photo prise", use_container_width=True)
                        else:
                            st.info("La capture par caméra n'est pas disponible dans cette version de Streamlit.")

                with skin_col2:
                    st.header("Options")
                    skin_top_n = st.slider(
                        "Nombre de résultats",
                        min_value=1,
                        max_value=10,
                        value=5,
                        key="skin_top_n",
                    )
                    show_confidences = st.checkbox(
                        "Afficher les confidences détaillées",
                        value=True,
                        key="skin_show_conf",
                    )
                    show_individual = st.checkbox(
                        "Afficher les prédictions individuelles",
                        value=True,
                        key="skin_show_individual",
                    )
                    show_vlm = st.checkbox(
                        "Afficher l'explication VLM",
                        value=True,
                        key="skin_show_vlm",
                    )

                analyze_skin = st.button(
                    "🔬 Analyser l'image",
                    type="primary",
                    use_container_width=True,
                    disabled=(skin_image is None),
                )

                if analyze_skin and skin_image is not None:
                    with st.spinner("Analyse de l'image par l'ensemble IA..."):
                        try:
                            # Try ensemble first, fall back to single model
                            individual_preds = {}
                            agreement_status = None

                            if ensemble_models_found and ENSEMBLE_AVAILABLE:
                                try:
                                    skin_result, individual_preds = agent.diagnose_skin_image_ensemble(
                                        skin_image, top_n=skin_top_n
                                    )
                                    # Read the agreement flag set by the ensemble classifier
                                    agreement_status = getattr(skin_result, '_ensemble_agreement', None)
                                except Exception:
                                    skin_result = agent.diagnose_skin_image(skin_image, top_n=skin_top_n)
                            else:
                                skin_result = agent.diagnose_skin_image(skin_image, top_n=skin_top_n)

                            # Urgent attention banner
                            if skin_result.needs_urgent_attention:
                                st.error(
                                    "🚨 **ATTENTION URGENTE** — L'analyse détecte une possible lésion "
                                    "nécessitant une consultation médicale rapide. "
                                    "Consultez un dermatologue sans délai."
                                )

                            # Ensemble agreement note
                            if agreement_status is True:
                                st.success("✅ **Consensus** : Tous les modèles s'accordent sur le diagnostic.")
                            elif agreement_status is False:
                                st.info("ℹ️ **Divergence** : Les modèles ne s'accordent pas — le plus confiant a été retenu.")

                            # Main ensemble result
                            if skin_result.top_diagnosis:
                                top = skin_result.top_diagnosis
                                severity_colors = {
                                    "red": "red",
                                    "orange": "orange",
                                    "yellow": "orange",
                                    "green": "green",
                                }
                                color_key = severity_colors.get(top.color, "gray")

                                st.subheader("🏆 Résultat Ensemble")
                                st.markdown(
                                    f"**:{color_key}[{top.readable_name}]**  "
                                    f"— Sévérité : *{top.severity}*"
                                )
                                st.progress(top.confidence)
                                st.markdown(f"**Confiance ensemble :** {top.confidence:.1%}")
                                st.info(f"💡 **Conseil :** {top.advice}")

                            # Individual model predictions
                            if individual_preds and show_individual:
                                with st.expander("🔍 Prédictions individuelles par modèle", expanded=False):
                                    cols = st.columns(len(individual_preds))
                                    for col, (model_name, preds) in zip(cols, individual_preds.items()):
                                        with col:
                                            st.markdown(f"**{model_name}**")
                                            if preds:
                                                top_pred = preds[0]
                                                st.markdown(f"Top-1: **{top_pred.readable_name}**")
                                                st.progress(top_pred.confidence)
                                                st.caption(f"{top_pred.confidence:.1%}")
                                            for pred in preds[1:3]:
                                                st.caption(
                                                    f"• {pred.readable_name} ({pred.confidence:.1%})"
                                                )

                            # Other candidates
                            if skin_result.candidates and show_confidences:
                                st.subheader("Autres candidats")
                                for idx, cand in enumerate(skin_result.candidates[1:], 2):
                                    with st.expander(
                                        f"#{idx} — {cand.readable_name} "
                                        f"({cand.confidence:.1%})"
                                    ):
                                        st.progress(cand.confidence)
                                        st.markdown(f"**Sévérité :** {cand.severity}")
                                        st.markdown(f"**Conseil :** {cand.advice}")

                            # VLM Explanation
                            if show_vlm and VLM_AVAILABLE:
                                st.subheader("💬 Explication VLM")
                                with st.spinner("Génération de l'explication..."):
                                    try:
                                        vlm = initialize_vlm_explainer()
                                        if vlm is not None:
                                            explanation = vlm.explain(skin_image, skin_result)
                                            st.markdown(explanation)
                                        else:
                                            st.info("Le module VLM n'a pas pu être initialisé.")
                                    except Exception as vlm_err:
                                        st.warning(f"Explication VLM non disponible : {vlm_err}")

                            st.divider()
                            st.warning(skin_result.disclaimer)

                        except Exception as e:
                            st.error(f"Erreur lors de l'analyse : {e}")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Agent Médical Vision IA v3.0</p>
        <p>Ensemble EfficientNet-B3 + MobileNetV2 + ResNet50 | VLM Explanations</p>
        <p>Outil informatif - Ne remplace pas une consultation médicale</p>
        <p>En cas d'urgence: 15 (SAMU)</p>
        <p>⚠️ This is not medical advice. Consult a healthcare professional.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
