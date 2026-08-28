# Layer 8.6 — Personalization Layer

Personalization and continuous adaptation enabling users to enroll custom, regional, or family-specific signs with few-shot examples while preserving user privacy.

## Modules
- `siamese_metric_learner.py`: Prototypical & Siamese networks for few-shot gesture enrollment (1–5 samples).
- `federated_client.py`: On-device local gradient computation and model parameter updating.
- `federated_server.py`: Flower-based federated averaging server with differential privacy guarantees (raw video never leaves user devices).
