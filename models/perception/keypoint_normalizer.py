"""
Keypoint Normalization & Feature Alignment Module (§8.1, §8.3)
Part of Tereguwami (ተርጓሚ) Perception & Feature Engineering Pipeline

Performs spatial centering, distance-invariant scaling, dropped-frame spline interpolation,
and multi-stream separation (hands, pose, facial grammar).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class KeypointNormalizer:
    """
    Normalizes raw MediaPipe Holistic (T, 543, 3) coordinate arrays:
    1. Spatial Centering: Centers coordinate frame at midpoint of shoulders.
    2. Scale Invariance: Normalizes by Euclidean shoulder distance.
    3. Temporal Interpolation: Cubic / linear interpolation for dropped frames.
    4. Motion Derivatives: Calculates velocity (1st delta) and acceleration (2nd delta).
    5. Non-Manual Slicing: Extracts dedicated facial grammar Action Units.
    """

    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12

    # Key landmark indices in the 468-point face mesh for Ethiopian non-manual grammar
    EYEBROW_LEFT = [70, 63, 105, 66, 107]
    EYEBROW_RIGHT = [336, 296, 334, 293, 300]
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    NOSE_TIP = 1  # For head orientation / tilt

    def __init__(self, target_fps: int = 25):
        self.target_fps = target_fps

    def normalize_spatial(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates of shape (T, 543, 3) or (543, 3) relative to shoulder midpoint
        and scale by inter-shoulder distance.
        """
        is_batched = (len(landmarks.shape) == 3)
        if not is_batched:
            landmarks = np.expand_dims(landmarks, axis=0)

        T, num_pts, dims = landmarks.shape
        normalized = np.copy(landmarks)

        for t in range(T):
            left_shoulder = landmarks[t, self.LEFT_SHOULDER_IDX]
            right_shoulder = landmarks[t, self.RIGHT_SHOULDER_IDX]

            # If shoulders are detected (non-zero), use them as center and scale anchor
            shoulder_dist = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
            if shoulder_dist > 1e-4:
                center = (left_shoulder + right_shoulder) / 2.0
                scale = shoulder_dist
            else:
                # Fallback to centroid of all valid points
                valid_mask = np.any(landmarks[t] != 0, axis=-1)
                if np.any(valid_mask):
                    center = np.mean(landmarks[t, valid_mask], axis=0)
                    scale = np.std(landmarks[t, valid_mask]) + 1e-6
                else:
                    center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                    scale = 1.0

            normalized[t] = (landmarks[t] - center) / scale

        return normalized if is_batched else normalized[0]

    normalize_frame = normalize_spatial

    def interpolate_dropped_frames(self, sequence: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate missing or zero-dropped frames across the temporal dimension.
        Input sequence shape: (T, 543, 3).
        """
        T, num_pts, dims = sequence.shape
        if T <= 1:
            return sequence

        interpolated = np.copy(sequence)
        # Check if entire frame is zero
        frame_active = np.any(sequence != 0, axis=(1, 2))
        valid_indices = np.where(frame_active)[0]

        if len(valid_indices) == 0 or len(valid_indices) == T:
            return sequence

        for p in range(num_pts):
            for d in range(dims):
                valid_values = sequence[valid_indices, p, d]
                interpolated[:, p, d] = np.interp(
                    np.arange(T),
                    valid_indices,
                    valid_values
                )

        return interpolated

    def compute_derivatives(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate 1st-order velocities and 2nd-order accelerations along the temporal axis.
        Output shapes match input: (T, 543, 3).
        """
        T = sequence.shape[0]
        if T <= 1:
            velocity = np.zeros_like(sequence)
            acceleration = np.zeros_like(sequence)
            return velocity, acceleration

        velocity = np.zeros_like(sequence)
        velocity[1:] = sequence[1:] - sequence[:-1]
        velocity[0] = velocity[1]

        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = acceleration[1]

        return velocity, acceleration

    def extract_non_manual_features(self, sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract high-resolution facial semantic markers directly relevant to ESL grammar:
        - Eyebrow elevation (polar question vs wh-question marker)
        - Eye aperture
        - Mouth aperture & shape (mouthing vs mouth gesture)
        - Head tilt & rotation
        """
        T = sequence.shape[0]
        eyebrow_elevation = np.zeros(T, dtype=np.float32)
        mouth_aperture = np.zeros(T, dtype=np.float32)
        head_tilt = np.zeros(T, dtype=np.float32)

        face_offset = 33 + 21 + 21  # 75

        for t in range(T):
            # 1. Eyebrow elevation relative to nose tip
            nose_y = sequence[t, face_offset + self.NOSE_TIP, 1]
            left_brow_y = np.mean(sequence[t, face_offset + np.array(self.EYEBROW_LEFT), 1])
            right_brow_y = np.mean(sequence[t, face_offset + np.array(self.EYEBROW_RIGHT), 1])
            avg_brow_y = (left_brow_y + right_brow_y) / 2.0
            eyebrow_elevation[t] = float(nose_y - avg_brow_y)

            # 2. Mouth aperture (distance between upper and lower lip inner landmarks)
            upper_lip_y = sequence[t, face_offset + 14, 1]
            lower_lip_y = sequence[t, face_offset + 17, 1]
            mouth_aperture[t] = float(abs(lower_lip_y - upper_lip_y))

            # 3. Head lateral tilt (angle between outer eye corners)
            left_eye_x, left_eye_y = sequence[t, face_offset + 33, 0], sequence[t, face_offset + 33, 1]
            right_eye_x, right_eye_y = sequence[t, face_offset + 263, 0], sequence[t, face_offset + 263, 1]
            head_tilt[t] = float(np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))

        return {
            "eyebrow_elevation": eyebrow_elevation,
            "mouth_aperture": mouth_aperture,
            "head_tilt": head_tilt
        }

    def prepare_multimodal_tensor(self, raw_sequence: np.ndarray) -> np.ndarray:
        """
        Complete normalization pipeline returning a flat feature matrix ready for sequence modeling.
        Returns: np.ndarray of shape (T, 543 * 3) = (T, 1629).
        """
        interpolated = self.interpolate_dropped_frames(raw_sequence)
        normalized = self.normalize_spatial(interpolated)
        T, num_pts, dims = normalized.shape
        return normalized.reshape(T, num_pts * dims)


# Global normalizer instance
keypoint_normalizer = KeypointNormalizer()
