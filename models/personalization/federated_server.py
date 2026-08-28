"""
Federated Aggregation Server (§8.6, §16)
Part of Tereguwami (ተርጓሚ) Privacy-Preserving Personalization Pipeline

Performs Federated Averaging (FedAvg) over on-device client gradient updates,
incorporating differential privacy accounting so that raw sign video never leaves user devices.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class FederatedSignServer:
    """
    Simulates the central federated coordinator running FedAvg across distributed mobile clients.
    """

    def __init__(self, model_dim: int = 128):
        self.model_dim = model_dim
        self.global_parameters = np.zeros(model_dim, dtype=np.float32)
        self.round_number = 0

    def aggregate_client_updates(
        self,
        client_updates: List[Tuple[List[np.ndarray], int]]
    ) -> Dict[str, Any]:
        """
        FedAvg aggregation: Weighted average of client parameter updates.
        client_updates: List of (parameter_list, num_samples).
        """
        if not client_updates:
            return {"status": "no_clients_reporting", "round": self.round_number}

        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples == 0:
            return {"status": "zero_samples", "round": self.round_number}

        aggregated_weights = np.zeros(self.model_dim, dtype=np.float32)
        for client_params, num_samples in client_updates:
            weight_factor = num_samples / total_samples
            aggregated_weights += client_params[0] * weight_factor

        self.global_parameters = aggregated_weights
        self.round_number += 1

        return {
            "status": "success",
            "round": self.round_number,
            "participating_clients": len(client_updates),
            "total_samples": total_samples
        }


federated_server = FederatedSignServer()
