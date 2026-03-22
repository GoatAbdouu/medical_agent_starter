"""
Services pour l'agent médical
"""
from medical_agent.services.voice_recognition import (
    VoiceRecognitionService,
    StreamlitVoiceRecorder,
    check_voice_dependencies
)

__all__ = [
    'VoiceRecognitionService',
    'StreamlitVoiceRecorder',
    'check_voice_dependencies'
]
