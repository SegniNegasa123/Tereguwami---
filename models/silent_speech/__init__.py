"""
Tereguwami Silent-Speech Module (Python Package Interface)
Decodes surface electromyography (sEMG) neuromuscular signals from facial subvocalizations.
"""

from models.silent_speech.emg_decoder import silent_speech_decoder, SilentSpeechEMGDecoder

__all__ = ["silent_speech_decoder", "SilentSpeechEMGDecoder"]
