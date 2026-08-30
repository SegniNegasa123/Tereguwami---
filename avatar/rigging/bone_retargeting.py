"""
Tereguwami Layer 4: Bone Retargeter
Maps SignAvatars SMPL-X/MANO joint tensors into standard Ready Player Me / Mixamo bone names.
"""

from typing import Dict, Any

SMPLX_TO_RPM_BODY_MAP = {
    "pelvis": "Hips",
    "spine1": "Spine",
    "spine2": "Spine1",
    "spine3": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}

MANO_TO_RPM_HAND_MAP = {
    # Right Hand Digits
    "right_thumb_1": "RightHandThumb1",
    "right_thumb_2": "RightHandThumb2",
    "right_thumb_3": "RightHandThumb3",
    "right_index_1": "RightHandIndex1",
    "right_index_2": "RightHandIndex2",
    "right_index_3": "RightHandIndex3",
    "right_middle_1": "RightHandMiddle1",
    "right_middle_2": "RightHandMiddle2",
    "right_middle_3": "RightHandMiddle3",
    "right_ring_1": "RightHandRing1",
    "right_ring_2": "RightHandRing2",
    "right_ring_3": "RightHandRing3",
    "right_pinky_1": "RightHandPinky1",
    "right_pinky_2": "RightHandPinky2",
    "right_pinky_3": "RightHandPinky3",
    # Left Hand Digits
    "left_thumb_1": "LeftHandThumb1",
    "left_thumb_2": "LeftHandThumb2",
    "left_thumb_3": "LeftHandThumb3",
    "left_index_1": "LeftHandIndex1",
    "left_index_2": "LeftHandIndex2",
    "left_index_3": "LeftHandIndex3",
    "left_middle_1": "LeftHandMiddle1",
    "left_middle_2": "LeftHandMiddle2",
    "left_middle_3": "LeftHandMiddle3",
    "left_ring_1": "LeftHandRing1",
    "left_ring_2": "LeftHandRing2",
    "left_ring_3": "LeftHandRing3",
    "left_pinky_1": "LeftHandPinky1",
    "left_pinky_2": "LeftHandPinky2",
    "left_pinky_3": "LeftHandPinky3",
}


def retarget_smplx_frame(smplx_dict: dict) -> dict:
    """Converts raw SignAvatars SMPL-X Euler rotations into RPM-compatible key-value pairs."""
    rpm_rotations = {}
    for smplx_key, rpm_bone in {**SMPLX_TO_RPM_BODY_MAP, **MANO_TO_RPM_HAND_MAP}.items():
        if smplx_key in smplx_dict:
            rot = smplx_dict[smplx_key]
            rpm_rotations[rpm_bone] = {"x": float(rot[0]), "y": float(rot[1]), "z": float(rot[2])}
    return rpm_rotations
