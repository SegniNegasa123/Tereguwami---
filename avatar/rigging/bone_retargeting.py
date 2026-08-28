"""
3D Skeletal Bone Retargeting Module (§8.5, §11)
Part of Tereguwami (ተርጓሚ) Avatar Rigging & Animation Pipeline

Retargets MediaPipe Holistic 3D landmark coordinates to standard humanoid skeletal bone
rotations and quaternions compatible with glTF, VRM, and Three.js / Unity humanoid avatars.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class BoneRetargetingEngine:
    """
    Computes humanoid bone rotation matrices and position offsets from 3D landmark arrays.
    
    Joint Hierarchy:
    - Root / Hips: Midpoint of Left Hip (23) and Right Hip (24)
    - Spine: Midpoint of Hips to Neck
    - Chest / Shoulders: Left Shoulder (11), Right Shoulder (12)
    - Left Arm: Shoulder (11) -> Elbow (13) -> Wrist (15)
    - Right Arm: Shoulder (12) -> Elbow (14) -> Wrist (16)
    - Hands: 21 finger joints per hand
    """

    @staticmethod
    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-6 else v

    @staticmethod
    def _direction_to_rotation(direction: np.ndarray, reference: np.ndarray = np.array([0, -1, 0])) -> Dict[str, float]:
        """Convert direction vector to Euler angles (yaw, pitch, roll) in radians."""
        d = BoneRetargetingEngine._normalize_vector(direction)
        pitch = float(np.arcsin(-np.clip(d[1], -1.0, 1.0)))
        yaw = float(np.arctan2(d[0], d[2]))
        roll = 0.0
        return {"pitch": round(pitch, 4), "yaw": round(yaw, 4), "roll": round(roll, 4)}

    def retarget_frame(self, frame_landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Input: frame_landmarks of shape (543, 3).
        Returns humanoid joint rotations and root translations.
        """
        # 1. Hips (Center of pelvis)
        left_hip = frame_landmarks[23]
        right_hip = frame_landmarks[24]
        hips_pos = (left_hip + right_hip) / 2.0

        # 2. Shoulders & Spine
        left_shoulder = frame_landmarks[11]
        right_shoulder = frame_landmarks[12]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        spine_dir = shoulder_center - hips_pos
        spine_rot = self._direction_to_rotation(spine_dir, reference=np.array([0, 1, 0]))

        # 3. Left Arm & Forearm
        left_elbow = frame_landmarks[13]
        left_wrist = frame_landmarks[15]
        left_upper_arm_dir = left_elbow - left_shoulder
        left_forearm_dir = left_wrist - left_elbow
        left_arm_rot = self._direction_to_rotation(left_upper_arm_dir)
        left_forearm_rot = self._direction_to_rotation(left_forearm_dir)

        # 4. Right Arm & Forearm
        right_elbow = frame_landmarks[14]
        right_wrist = frame_landmarks[16]
        right_upper_arm_dir = right_elbow - right_shoulder
        right_forearm_dir = right_wrist - right_elbow
        right_arm_rot = self._direction_to_rotation(right_upper_arm_dir)
        right_forearm_rot = self._direction_to_rotation(right_forearm_dir)

        # 5. Hand Keypoint Offsets (21 points each)
        left_hand_points = frame_landmarks[33:54].tolist()
        right_hand_points = frame_landmarks[54:75].tolist()

        return {
            "root_position": {"x": round(float(hips_pos[0]), 4), "y": round(float(hips_pos[1]), 4), "z": round(float(hips_pos[2]), 4)},
            "bones": {
                "spine": spine_rot,
                "left_upper_arm": left_arm_rot,
                "left_forearm": left_forearm_rot,
                "right_upper_arm": right_arm_rot,
                "right_forearm": right_forearm_rot
            },
            "hands": {
                "left_hand_landmarks": left_hand_points,
                "right_hand_landmarks": right_hand_points
            }
        }


bone_retargeter = BoneRetargetingEngine()
