"""
Service de reconnaissance vocale pour l'agent médical utilisant des modèles de deep learning avancés.
Ce module fournit une interface unifiée pour la transcription audio en texte en utilisant
plusieurs backends de reconnaissance vocale basés sur l'apprentissage profond.
Modèles de Deep Learning Utilisés:
----------------------------------
1. **OpenAI Whisper** (Modèle par défaut: "base")
    - Architecture: Transformer encoder-decoder
    - Type: Modèle Seq2Seq avec attention multi-têtes
    - Entraînement: 680,000 heures d'audio multilingue supervisé
    - Tailles disponibles: tiny, base, small, medium, large
    - Capacités: Transcription multilingue, détection de langue, traduction
    - Framework: PyTorch
    - Paramètres (base): ~74M paramètres
    - Performance: Fonctionne en local (CPU/GPU), pas besoin d'internet
2. **Google Speech Recognition API**
    - Architecture: Modèles neuronaux propriétaires de Google
    - Type: Réseaux de neurones récurrents (RNN) et Transformers
    - Basé sur: WaveNet, RNN-T (Recurrent Neural Network Transducer)
    - Entraînement: Données massives multilingues de Google
    - Performance: Nécessite une connexion internet, gratuit jusqu'à 60 minutes/mois
Bibliothèques Utilisées:
-----------------------
**Pour la Reconnaissance Vocale:**
- `openai-whisper`: Modèle Whisper d'OpenAI pour transcription locale
  * Utilise PyTorch sous le capot
  * Modèles pré-entraînés disponibles en plusieurs tailles
  * Support GPU (CUDA) et CPU
- `speech_recognition`: Interface Python pour plusieurs moteurs de reconnaissance
  * Google Speech Recognition API
  * Support de multiples backends
  * Gestion automatique de l'audio
**Pour le Traitement Audio:**
- `wave`: Lecture/écriture de fichiers WAV (bibliothèque standard Python)
- `pyaudio`: Capture audio depuis le microphone en temps réel
  * Wrapper Python pour PortAudio
  * Nécessaire pour `listen_from_microphone()`
**Pour l'Interface Streamlit:**
- `streamlit`: Framework web pour applications data science
  * `st.audio_input()`: Composant natif d'enregistrement audio (≥ v1.33)
  * Fallback vers upload de fichier pour versions antérieures
**Frameworks Deep Learning (dépendances indirectes):**
- `PyTorch`: Framework backend pour Whisper
  * Gestion des tenseurs et calculs GPU
  * Modèles de transformers pré-entraînés
  * Optimisations CUDA pour GPU NVIDIA
- `NumPy`: Calculs numériques et manipulation de tableaux audio
- `FFmpeg`: Conversion et traitement de formats audio (requis par Whisper)
Architecture du Pipeline de Transcription:
-----------------------------------------
1. **Capture Audio**: Microphone (PyAudio) ou fichier (Wave)
2. **Prétraitement**: Conversion en format WAV compatible
    - Rééchantillonnage à 16kHz
    - Conversion mono-canal
    - Normalisation du volume
3. **Transcription Deep Learning**:
    - **Whisper**: Encodeur audio → Décodeur texte avec attention croisée
    - **Google**: API REST vers modèles cloud
4. **Post-traitement**: Nettoyage du texte, capitalisation
Cas d'Usage:
-----------
- Consultation médicale vocale
- Prise de notes cliniques par dictée
- Transcription de symptômes patients
- Interface vocale pour dossiers médicaux
Installation des Dépendances:
----------------------------
Service de reconnaissance vocale pour l'agent médical
Supporte plusieurs backends: Google Speech Recognition, Whisper, etc.
"""
import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import wave

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class VoiceRecognitionService:
    """
    Service de reconnaissance vocale multi-backend
    
    Backends supportés:
    - Google Speech Recognition (gratuit, nécessite internet)
    - OpenAI Whisper (local, plus précis)
    """
    
    def __init__(self, backend: str = "google", language: str = "fr-FR"):
        """
        Initialise le service de reconnaissance vocale
        
        Args:
            backend: "google" ou "whisper"
            language: Code de langue (ex: "fr-FR" pour français)
        """
        self.backend = backend
        self.language = language
        
        if backend == "google" and not SR_AVAILABLE:
            raise ImportError(
                "speech_recognition n'est pas installé. "
                "Installez-le avec: pip install SpeechRecognition"
            )
        
        if backend == "whisper" and not WHISPER_AVAILABLE:
            raise ImportError(
                "whisper n'est pas installé. "
                "Installez-le avec: pip install openai-whisper"
            )
        
        if backend == "google":
            self.recognizer = sr.Recognizer()
        elif backend == "whisper":
            # Charger le modèle Whisper (base est un bon compromis vitesse/précision)
            self.whisper_model = whisper.load_model("base")
    
    def transcribe_audio_file(self, audio_path: Path) -> Tuple[str, Optional[str]]:
        """
        Transcrit un fichier audio en texte
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Tuple (texte transcrit, message d'erreur ou None)
        """
        if self.backend == "google":
            return self._transcribe_google(audio_path)
        elif self.backend == "whisper":
            return self._transcribe_whisper(audio_path)
        else:
            return "", f"Backend inconnu: {self.backend}"
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> Tuple[str, Optional[str]]:
        """
        Transcrit des données audio brutes en texte
        
        Args:
            audio_bytes: Données audio en bytes
            sample_rate: Taux d'échantillonnage
            
        Returns:
            Tuple (texte transcrit, message d'erreur ou None)
        """
        # Créer un fichier temporaire WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Écrire les données audio
            with wave.open(str(tmp_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)
        
        try:
            result = self.transcribe_audio_file(tmp_path)
        finally:
            # Nettoyer le fichier temporaire
            tmp_path.unlink(missing_ok=True)
        
        return result
    
    def _transcribe_google(self, audio_path: Path) -> Tuple[str, Optional[str]]:
        """Transcription avec Google Speech Recognition"""
        try:
            with sr.AudioFile(str(audio_path)) as source:
                audio = self.recognizer.record(source)
            
            # Reconnaître avec Google (gratuit)
            text = self.recognizer.recognize_google(
                audio, 
                language=self.language
            )
            return text, None
            
        except sr.UnknownValueError:
            return "", "Impossible de comprendre l'audio"
        except sr.RequestError as e:
            return "", f"Erreur de service Google: {e}"
        except Exception as e:
            return "", f"Erreur de transcription: {e}"
    
    def _transcribe_whisper(self, audio_path: Path) -> Tuple[str, Optional[str]]:
        """Transcription avec OpenAI Whisper (local)"""
        try:
            # Transcrire avec Whisper
            result = self.whisper_model.transcribe(
                str(audio_path),
                language=self.language.split("-")[0],  # "fr-FR" -> "fr"
                fp16=False  # Désactiver FP16 pour compatibilité CPU
            )
            return result["text"].strip(), None
            
        except Exception as e:
            return "", f"Erreur Whisper: {e}"
    
    def listen_from_microphone(self, timeout: int = 5, phrase_time_limit: int = 30) -> Tuple[str, Optional[str]]:
        """
        Écoute et transcrit depuis le microphone
        
        Args:
            timeout: Temps d'attente avant de commencer à parler (secondes)
            phrase_time_limit: Durée max d'enregistrement (secondes)
            
        Returns:
            Tuple (texte transcrit, message d'erreur ou None)
        """
        if not SR_AVAILABLE:
            return "", "speech_recognition n'est pas installé"
        
        recognizer = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                # Ajuster pour le bruit ambiant
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Écouter
                audio = recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            # Transcrire selon le backend
            if self.backend == "google":
                text = recognizer.recognize_google(audio, language=self.language)
                return text, None
            elif self.backend == "whisper":
                # Sauvegarder temporairement pour Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    with open(tmp_path, 'wb') as f:
                        f.write(audio.get_wav_data())
                
                try:
                    return self._transcribe_whisper(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
            
        except sr.WaitTimeoutError:
            return "", "Aucune parole détectée (timeout)"
        except sr.UnknownValueError:
            return "", "Impossible de comprendre l'audio"
        except sr.RequestError as e:
            return "", f"Erreur de service: {e}"
        except Exception as e:
            return "", f"Erreur: {e}"


class StreamlitVoiceRecorder:
    """
    Composant pour enregistrer de l'audio dans Streamlit
    Compatible avec st.audio_input (Streamlit >= 1.33) ou alternative
    """
    
    @staticmethod
    def get_audio_input_component():
        """
        Retourne le composant d'entrée audio approprié pour Streamlit
        """
        import streamlit as st
        
        # Vérifier si st.audio_input existe (Streamlit >= 1.33)
        if hasattr(st, 'audio_input'):
            return st.audio_input
        else:
            # Fallback: utiliser un file uploader
            return None
    
    @staticmethod
    def render_voice_input(key: str = "voice_input") -> Optional[bytes]:
        """
        Affiche le composant d'entrée vocale et retourne les données audio
        
        Args:
            key: Clé unique pour le composant Streamlit
            
        Returns:
            Données audio en bytes ou None
        """
        import streamlit as st
        
        # Essayer d'utiliser st.audio_input (Streamlit >= 1.33)
        if hasattr(st, 'audio_input'):
            audio_data = st.audio_input(
                "🎤 Cliquez pour enregistrer votre voix",
                key=key
            )
            if audio_data is not None:
                return audio_data.getvalue()
            return None
        else:
            # Fallback: upload de fichier audio
            st.info("💡 Votre version de Streamlit ne supporte pas l'enregistrement audio direct. "
                   "Vous pouvez uploader un fichier audio à la place.")
            
            uploaded_file = st.file_uploader(
                "📁 Uploader un fichier audio (WAV, MP3)",
                type=["wav", "mp3", "ogg", "m4a"],
                key=f"{key}_upload"
            )
            
            if uploaded_file is not None:
                return uploaded_file.getvalue()
            return None


def check_voice_dependencies() -> dict:
    """
    Vérifie les dépendances pour la reconnaissance vocale
    
    Returns:
        Dict avec le statut de chaque dépendance
    """
    deps = {
        "speech_recognition": SR_AVAILABLE,
        "whisper": WHISPER_AVAILABLE,
        "pyaudio": False
    }
    
    # Vérifier PyAudio (nécessaire pour le microphone)
    try:
        import pyaudio
        deps["pyaudio"] = True
    except ImportError:
        pass
    
    return deps
