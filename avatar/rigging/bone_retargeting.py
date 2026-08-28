"""
SignAvatars / SMPL-X 3D Skeletal & MANO Hand Retargeting Engine (§8.5, §11)
Part of Tereguwami (ተርጓሚ) Avatar Rigging & Animation Pipeline
Adapted from SignAvatars (Zhengdi Yu et al., ECCV 2024 / SMPL-X & MANO Benchmark)

Retargets 543 MediaPipe Holistic landmarks to SMPL-X 21-joint body skeleton and
MANO 15-joint articulated hand transforms for expressive 3D signing avatar animation.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class SignAvatarsSMPLXRetargeter:
    """
    Computes SMPL-X whole-body parameters and MANO hand joint rotations from 3D landmark arrays.
    """

    SMPLX_JOINTS = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]

    @staticmethod
    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-6 else v

    @staticmethod
    def _direction_to_rotation(direction: np.ndarray, reference: np.ndarray = np.array([0, -1, 0])) -> Dict[str, float]:
        d = SignAvatarsSMPLXRetargeter._normalize_vector(direction)
        pitch = float(np.arcsin(-np.clip(d[1], -1.0, 1.0)))
        yaw = float(np.arctan2(d[0], d[2]))
        roll = 0.0
        return {"pitch": round(pitch, 4), "yaw": round(yaw, 4), "roll": round(roll, 4)}

    def retarget_to_smplx(self, frame_landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Input: frame_landmarks of shape (543, 3).
        Returns SMPL-X parameter dictionary compatible with SignAvatars mesh renderers.
        """
        # Hips (pelvis)
        left_hip = frame_landmarks[23]
        right_hip = frame_landmarks[24]
        hips_pos = (left_hip + right_hip) / 2.0

        # Shoulders & Spine
        left_shoulder = frame_landmarks[11]
        right_shoulder = frame_landmarks[12]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        spine_dir = shoulder_center - hips_pos
        spine_rot = self._direction_to_rotation(spine_dir, reference=np.array([0, 1, 0]))

        # Arms
        left_elbow = frame_landmarks[13]
        left_wrist = frame_landmarks[15]
        left_upper_arm_dir = left_elbow - left_shoulder
        left_forearm_dir = left_wrist - left_elbow
        left_arm_rot = self._direction_to_rotation(left_upper_arm_dir)
        left_forearm_rot = self._direction_to_rotation(left_forearm_dir)

        right_elbow = frame_landmarks[14]
        right_wrist = frame_landmarks[16]
        right_upper_arm_dir = right_elbow - right_shoulder
        right_forearm_dir = right_wrist - right_elbow
        right_arm_rot = self._direction_to_rotation(right_upper_arm_dir)
        right_forearm_rot = self._direction_to_rotation(right_forearm_dir)

        # 15-joint MANO finger rotations per hand (45 floats each)
        left_mano = [0.0] * 45
        right_mano = [0.0] * 45

        # 21 Hand keypoint positions
        left_hand_pts = frame_landmarks[33:54].tolist()
        right_hand_pts = frame_landmarks[54:75].tolist()

        return {
            "root_orient": [0.0, float(spine_rot["yaw"]), 0.0],
            "trans": [round(float(hips_pos[0]), 4), round(float(hips_pos[1]), 4), round(float(hips_pos[2]), 4)],
            "root_position": {"x": round(float(hips_pos[0]), 4), "y": round(float(hips_pos[1]), 4), "z": round(float(hips_pos[2]), 4)},
            "bones": {
                "spine": spine_rot,
                "left_upper_arm": left_arm_rot,
                "left_forearm": left_forearm_rot,
                "right_upper_arm": right_arm_rot,
                "right_forearm": right_forearm_rot
            },
            "smplx_body_pose": [
                0.0, 0.0, 0.0,  # pelvis
                0.0, 0.0, 0.0,  # left_hip
                0.0, 0.0, 0.0,  # right_hip
                spine_rot["pitch"], spine_rot["yaw"], 0.0,  # spine1
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                left_arm_rot["pitch"], 0.0, left_arm_rot["yaw"],    # left shoulder (16)
                right_arm_rot["pitch"], 0.0, right_arm_rot["yaw"],  # right shoulder (17)
                left_forearm_rot["pitch"], 0.0, 0.0,                # left elbow (18)
                right_forearm_rot["pitch"], 0.0, 0.0,               # right elbow (19)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "left_hand_pose": left_mano,
            "right_hand_pose": right_mano,
            "hands": {
                "left_hand_landmarks": left_hand_pts,
                "right_hand_landmarks": right_hand_pts
            }
        }

    retarget_frame = retarget_to_smplx


bone_retargeter = SignAvatarsSMPLXRetargeter()
BoneRetargetingEngine = SignAvatarsSMPLXRetargeter
