"""
Ethiopian Sign Language (ESL / ETHSL) Dataset Loader & Batch Generator
Part of Tereguwami (ተርጓሚ) Benchmark & Training Infrastructure (§10.1, §10.2)

Provides unified data loading, signer-independent split management, landmark sequence batching,
and annotation collation across continuous ESL domains (healthcare, legal, education, civic).
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class ESLDatasetLoader:
    """
    Manages loading of continuous Ethiopian Sign Language (ESL) multi-modal data.
    
    Adheres to PHOENIX-2014T and How2Sign annotation standards with:
    - 543 3D MediaPipe Holistic landmarks per frame (1629 coordinates)
    - Early-fusion non-manual markers (eyebrows, mouth, head pose)
    - Trilingual translations: Amharic (Ethiopic), Afaan Oromo, English
    - Signer-independent train/val/test splits (60/15/25 signers)
    """

    NUM_TOTAL_LANDMARKS = 543  # 33 pose + 21 LH + 21 RH + 468 face mesh
    COORDINATES_PER_LANDMARK = 3  # (x, y, z)
    FEATURE_DIM = NUM_TOTAL_LANDMARKS * COORDINATES_PER_LANDMARK  # 1629 floats

    def __init__(self, data_root: Optional[str] = None):
        if data_root is None:
            # Assume running from repository root or subfolder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_root = os.path.abspath(current_dir)
        else:
            self.data_root = os.path.abspath(data_root)

        self.annotations_path = os.path.join(self.data_root, "annotations", "sample_continuous_esl_annotations.json")
        self.splits_dir = os.path.join(self.data_root, "splits")
        self._samples: List[Dict[str, Any]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load annotations JSON from disk."""
        if os.path.exists(self.annotations_path):
            with open(self.annotations_path, "r", encoding="utf-8") as f:
                self._samples = json.load(f)
        else:
            self._samples = []

    def get_split_manifest(self, split_name: str) -> Dict[str, Any]:
        """Load a specific split manifest (e.g. signer_independent_test)."""
        filename = f"{split_name}.json" if not split_name.endswith(".json") else split_name
        path = os.path.join(self.splits_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Split manifest {split_name} not found at {path}")

    def filter_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Filter dataset samples by conversational domain."""
        return [s for s in self._samples if s.get("domain") == domain]

    def filter_by_split(self, split_name: str) -> List[Dict[str, Any]]:
        """Filter dataset samples to signers present in the designated split."""
        manifest = self.get_split_manifest(split_name)
        allowed_signers = set(manifest.get("signers", []))
        return [s for s in self._samples if s.get("signer_id") in allowed_signers]

    def generate_synthetic_keypoints(self, frame_count: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate realistic synthetic 3D keypoint temporal trajectories for pipeline validation
        when raw camera video is not accessible.
        
        Output shape: (T, 543, 3) where T is frame_count.
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.linspace(0, 2 * np.pi, frame_count)
        # Base body/hands/face coordinates centered around (0, 0, 0)
        landmarks = np.zeros((frame_count, self.NUM_TOTAL_LANDMARKS, self.COORDINATES_PER_LANDMARK), dtype=np.float32)

        # Generate smooth sinusoidal hand & head movements
        for i in range(self.NUM_TOTAL_LANDMARKS):
            freq = 0.5 + (i % 5) * 0.2
            phase = (i % 7) * (np.pi / 4)
            landmarks[:, i, 0] = 0.5 + 0.1 * np.sin(freq * t + phase)  # X
            landmarks[:, i, 1] = 0.5 + 0.1 * np.cos(freq * t + phase)  # Y
            landmarks[:, i, 2] = 0.05 * np.sin(2 * freq * t)           # Z depth

        return landmarks

    def get_sample_with_features(self, sample_index: int = 0) -> Dict[str, Any]:
        """Retrieve annotation paired with extracted or synthetic 3D landmarks."""
        if not self._samples:
            raise ValueError("No annotations loaded in dataset.")

        sample = self._samples[sample_index % len(self._samples)].copy()
        frame_count = sample["video_metadata"]["frame_count"]
        # In deployment, load extracted .npy file from data/processed/
        # In test/dev without raw video, synthesize consistent keypoints
        sample["keypoints"] = self.generate_synthetic_keypoints(frame_count, seed=sample_index)
        return sample

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_sample_with_features(idx)


# Global singleton instance
esl_dataset = ESLDatasetLoader()
