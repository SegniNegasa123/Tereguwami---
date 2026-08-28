"""
Tereguwami / SignAvatars 3D Production Tier (§8.5)
Holistic 3D SMPL-X & MANO articulated motion generator driving ESL avatar animations.
Adapted from SignAvatars (Zhengdi Yu et al., ECCV 2024 / SMPL-X Benchmark).
"""

from models.production.progressive_transformer import (
    avatar_production_engine,
    AvatarProductionEngine,
    SignAvatarsProductionEngine
)

__all__ = [
    "avatar_production_engine",
    "AvatarProductionEngine",
    "SignAvatarsProductionEngine"
]
