"""
Application Streamlit - Agent Médical
Avec support Deep Learning et reconnaissance vocale
"""
import streamlit as st
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from medical_agent import MedicalAgent
from medical_agent.config.settings import settings

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

def main():
    """Application principale"""
    
    st.title("🏥 Agent Médical")
    st.caption("Système de diagnostic avec Deep Learning et reconnaissance vocale")
    
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
        
        if agent and agent.disease_predictor.df is not None:
            st.markdown("### 📈 Statistiques")
            n_diseases = agent.disease_predictor.df['disease'].nunique()
            n_symptoms = agent.disease_predictor.df['symptom'].nunique()
            st.metric("Maladies", n_diseases)
            st.metric("Symptômes", n_symptoms)
    
    # Zone principale
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
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Agent Médical v2.0</p>
        <p>Outil informatif - Ne remplace pas une consultation médicale</p>
        <p>En cas d'urgence: 15 (SAMU)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
