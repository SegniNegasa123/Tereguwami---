"""
High-Resolution Facial Mesh Crop & Semantics Extractor (§8.3)
Part of Tereguwami (ተርጓሚ) Non-Manual Grammar Extraction Pipeline
"""

from typing import Dict, Any, List, Optional
import numpy as np


class HighResFaceMeshExtractor:
    """
    Extracts high-resolution bounding crops around the face to compute
    fine-grained eyebrow aperture, mouthing, and head movement for ESL grammar.
    """

    def __init__(self, crop_size: int = 256):
        self.crop_size = crop_size

    def extract_face_crop(self, frame_rgb: np.ndarray, face_center: tuple) -> np.ndarray:
        """Crop and resize region of interest around the face."""
        h, w = frame_rgb.shape[:2]
        cx, cy = int(face_center[0] * w), int(face_center[1] * h)
        radius = int(min(h, w) * 0.2)

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(h, cy + radius)

        crop = frame_rgb[y1:y2, x1:x2]
        return crop

    def compute_aperture_dynamics(self, face_landmarks: np.ndarray) -> Dict[str, float]:
        """Compute relative aperture dynamics for mouthing and eye squints."""
        if len(face_landmarks) < 468:
            return {"eye_squint": 0.0, "mouth_open": 0.0}

        # Upper and lower eyelid
        left_eye_top = face_landmarks[159]
        left_eye_bot = face_landmarks[145]
        eye_gap = float(np.linalg.norm(left_eye_top - left_eye_bot))

        # Upper and lower lip
        lip_top = face_landmarks[13]
        lip_bot = face_landmarks[14]
        mouth_gap = float(np.linalg.norm(lip_top - lip_bot))

        return {
            "eye_squint": round(float(np.clip(1.0 - eye_gap * 20.0, 0.0, 1.0)), 3),
            "mouth_open": round(float(np.clip(mouth_gap * 15.0, 0.0, 1.0)), 3)
        }


highres_face_extractor = HighResFaceMeshExtractor()
