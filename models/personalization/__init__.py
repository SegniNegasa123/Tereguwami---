"""
Tereguwami Personalization Layer (§8.6)
Few-shot metric learning enrollment and federated client adaptation.
"""

from models.personalization.siamese_few_shot import (
    sign_personalizer,
    PrototypicalSignPersonalizer
)
from models.personalization.federated_client import (
    FederatedSignClient,
    create_federated_client
)

__all__ = [
    "sign_personalizer",
    "PrototypicalSignPersonalizer",
    "FederatedSignClient",
    "create_federated_client"
]
