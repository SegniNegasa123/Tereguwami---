"""
Tereguwami Production Layer (§8.5)
Generative pose sequence progressive transformer driving ESL avatar animations.
"""

from models.production.progressive_transformer import (
    avatar_production_engine,
    AvatarProductionEngine
)

__all__ = ["avatar_production_engine", "AvatarProductionEngine"]
