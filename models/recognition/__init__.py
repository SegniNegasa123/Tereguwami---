"""
Tereguwami Recognition Layer (§8.2)
Temporal sequence baselines (CNN-LSTM, BiLSTM, GRU).
"""

from models.recognition.temporal_models import (
    temporal_recognizer,
    TemporalRecognitionEngine
)

__all__ = ["temporal_recognizer", "TemporalRecognitionEngine"]
