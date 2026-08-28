"""
Bidirectional LSTM Recognizer Module (§8.2)
BiLSTM with temporal context over keypoint tracks.
"""

from models.recognition.temporal_models import BiLSTM_Recognizer, HAS_TORCH

__all__ = ["BiLSTM_Recognizer", "HAS_TORCH"]
