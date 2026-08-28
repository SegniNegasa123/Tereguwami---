"""
Tereguwami Translation Layer (§8.4)
Gloss-free multimodal transformer and constrained decoding guardrails.
"""

from models.translation.gloss_free_transformer import (
    continuous_translator,
    ContinuousTranslationEngine
)
from models.translation.constrained_decoder import (
    constrained_decoder,
    ConstrainedDecoder
)

__all__ = [
    "continuous_translator",
    "ContinuousTranslationEngine",
    "constrained_decoder",
    "ConstrainedDecoder"
]
