"""
GRU Sequence Recognizer Module (§8.2)
Gated Recurrent Unit network for low-compute mobile inference.
"""

from models.recognition.temporal_models import GRU_Recognizer, HAS_TORCH

__all__ = ["GRU_Recognizer", "HAS_TORCH"]
