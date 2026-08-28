"""
CNN-LSTM Temporal Recognizer Module (§8.2)
1D/2D CNN feature encoder + LSTM temporal sequence classifier.
"""

from models.recognition.temporal_models import CNN_LSTM_Recognizer, HAS_TORCH

__all__ = ["CNN_LSTM_Recognizer", "HAS_TORCH"]
