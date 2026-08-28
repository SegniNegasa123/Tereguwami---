"""
Few-Shot Metric Learning Personalization Layer (§8.6)
Part of Tereguwami (ተርጓሚ) Personalization & Adaptive Enrollment Pipeline

Enables users to enroll new or family-specific Ethiopian signs using 1-to-5 shot examples
via Siamese / Prototypical embedding projections without retraining the global model.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class PrototypicalSignPersonalizer:
    """
    Metric-learning few-shot enrollment engine.
    Maps temporal keypoint trajectories to an L2-normalized metric embedding space (dimension 128)
    and classifies incoming gestures against user-enrolled prototype centers.
    """

    EMBEDDING_DIM = 128

    def __init__(self):
        # Local registry of enrolled sign prototypes: sign_name -> mean embedding vector
        self._enrolled_prototypes: Dict[str, np.ndarray] = {}
        self._exemplars_count: Dict[str, int] = {}
        # Random projection matrix for metric mapping: 1629 -> 128
        np.random.seed(42)
        self._projection_matrix = np.random.randn(1629, self.EMBEDDING_DIM).astype(np.float32)
        # Orthonormalize projection columns
        q, _ = np.linalg.qr(self._projection_matrix)
        self._projection_matrix = q

    def extract_embedding(self, keypoint_sequence: np.ndarray) -> np.ndarray:
        """
        Input: np.ndarray of shape (T, 1629).
        Computes temporal weighted average and projects into normalized metric space.
        """
        T = keypoint_sequence.shape[0]
        # Temporal center weighting
        weights = np.hanning(T) if T > 2 else np.ones(T)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            pooled = np.sum(keypoint_sequence * weights[:, np.newaxis], axis=0) / weights_sum
        else:
            pooled = np.mean(keypoint_sequence, axis=0)

        # Linear metric projection
        embedding = np.dot(pooled, self._projection_matrix)
        norm = np.linalg.norm(embedding)
        return (embedding / norm) if norm > 1e-6 else embedding

    def enroll_sign(self, sign_name: str, exemplar_sequences: List[np.ndarray]) -> Dict[str, Any]:
        """
        Enroll a new personal or family-specific sign with 1 to 5 exemplar demonstrations.
        """
        if not exemplar_sequences:
            raise ValueError("At least one exemplar sequence is required for enrollment.")

        embeddings = [self.extract_embedding(seq) for seq in exemplar_sequences]
        prototype = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(prototype)
        prototype = (prototype / norm) if norm > 1e-6 else prototype

        self._enrolled_prototypes[sign_name] = prototype
        self._exemplars_count[sign_name] = len(exemplar_sequences)

        return {
            "sign_name": sign_name,
            "shots_enrolled": len(exemplar_sequences),
            "status": "enrolled",
            "embedding_dimension": self.EMBEDDING_DIM
        }

    def recognize_custom_sign(
        self,
        query_sequence: np.ndarray,
        distance_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Query an unknown gesture against user enrolled personal prototypes.
        Returns match if cosine distance < distance_threshold.
        """
        if not self._enrolled_prototypes:
            return {
                "matched": False,
                "reason": "No custom signs enrolled in personal profile"
            }

        query_emb = self.extract_embedding(query_sequence)
        best_match = None
        min_distance = float("inf")

        for sign_name, prototype in self._enrolled_prototypes.items():
            # Cosine distance: 1 - cosine_similarity
            cos_sim = float(np.dot(query_emb, prototype))
            cos_dist = 1.0 - cos_sim
            if cos_dist < min_distance:
                min_distance = cos_dist
                best_match = sign_name

        if min_distance <= distance_threshold:
            confidence = max(0.0, min(1.0, 1.0 - min_distance))
            return {
                "matched": True,
                "sign_name": best_match,
                "confidence": round(confidence, 4),
                "metric_distance": round(min_distance, 4)
            }

        return {
            "matched": False,
            "nearest_candidate": best_match,
            "metric_distance": round(min_distance, 4),
            "threshold": distance_threshold
        }

    def list_enrolled_signs(self) -> List[Dict[str, Any]]:
        return [
            {"sign_name": k, "exemplars": self._exemplars_count[k]}
            for k in self._enrolled_prototypes
        ]


# Global personalizer singleton
sign_personalizer = PrototypicalSignPersonalizer()
