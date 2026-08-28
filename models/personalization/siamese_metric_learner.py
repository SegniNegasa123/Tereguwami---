"""
Siamese / Prototypical Metric Learner Module (§8.6)
Metric-learning enrollment for few-shot sign adaptation (1-5 samples).
"""

from models.personalization.siamese_few_shot import (
    PrototypicalSignPersonalizer,
    sign_personalizer
)

__all__ = ["PrototypicalSignPersonalizer", "sign_personalizer"]
