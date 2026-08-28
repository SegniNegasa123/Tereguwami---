"""
SignAvatars / FLAME 50-Expression & FACS Blendshape Mappings (§8.3, §8.5)
Part of Tereguwami (ተርጓሚ) SignAvatars Avatar Rigging & Facial Semantics Tier
Adapted from SignAvatars (Zhengdi Yu et al., ECCV 2024 / FLAME Expression Benchmark)

Maps 3D facial mesh Action Units directly into FLAME 50-dimensional expression coefficients
and Apple ARKit / glTF 52 standardized facial blendshapes for realistic mouthing and non-manual grammar.
"""

from typing import Dict, Any, List
import numpy as np


class SignAvatarsBlendshapeMapper:
    """
    Computes FLAME 50-expression vectors and ARKit blendshapes from 3D face mesh coordinates.
    """

    # MediaPipe Face Landmark Indices
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    NOSE_TIP = 1
    CHIN = 152

    def map_landmarks_to_blendshapes(self, face_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Input: face_landmarks of shape (468, 3).
        Returns standardized ARKit blendshapes.
        """
        face_height = float(np.linalg.norm(face_landmarks[self.NOSE_TIP] - face_landmarks[self.CHIN]))
        if face_height < 1e-4:
            face_height = 0.2

        # 1. Eyebrow Elevation (AU1/2 - Polar Question)
        left_brow_y = float(np.mean(face_landmarks[self.LEFT_EYEBROW, 1]))
        right_brow_y = float(np.mean(face_landmarks[self.RIGHT_EYEBROW, 1]))
        brow_y = (left_brow_y + right_brow_y) / 2.0
        nose_y = float(face_landmarks[self.NOSE_TIP, 1])
        raw_brow_elev = float((nose_y - brow_y) / face_height)
        brow_inner_up = float(np.clip((raw_brow_elev - 0.35) * 4.0, 0.0, 1.0))

        # 2. Eyebrow Furrow (AU4 - Wh-Question)
        brow_dist = float(np.linalg.norm(face_landmarks[self.LEFT_EYEBROW[-1]] - face_landmarks[self.RIGHT_EYEBROW[-1]]))
        brow_down = float(np.clip((0.15 - (brow_dist / face_height)) * 5.0, 0.0, 1.0))

        # 3. Jaw Open (AU26/27 - Mouthing)
        lip_gap = float(np.linalg.norm(face_landmarks[self.UPPER_LIP] - face_landmarks[self.LOWER_LIP]))
        jaw_open = float(np.clip((lip_gap / face_height) * 4.0, 0.0, 1.0))

        # 4. Mouth Smile / Stretch
        mouth_width = float(np.linalg.norm(face_landmarks[self.LEFT_MOUTH_CORNER] - face_landmarks[self.RIGHT_MOUTH_CORNER]))
        mouth_smile = float(np.clip((mouth_width / face_height - 0.45) * 3.0, 0.0, 1.0))

        # 5. Eye Aperture / Blink (AU45)
        left_eye_aperture = float(abs(face_landmarks[159, 1] - face_landmarks[145, 1]))
        eye_blink_left = float(np.clip(1.0 - (left_eye_aperture / (face_height * 0.1)), 0.0, 1.0))

        return {
            "browInnerUp": round(brow_inner_up, 3),
            "browDownLeft": round(brow_down, 3),
            "browDownRight": round(brow_down, 3),
            "jawOpen": round(jaw_open, 3),
            "mouthSmileLeft": round(mouth_smile, 3),
            "mouthSmileRight": round(mouth_smile, 3),
            "eyeBlinkLeft": round(eye_blink_left, 3),
            "eyeBlinkRight": round(eye_blink_left, 3),
            "mouthPucker": 0.0,
            "mouthFunnel": round(jaw_open * 0.5, 3)
        }

    def map_to_flame_expression_vector(self, face_landmarks: np.ndarray) -> List[float]:
        """
        Maps 3D face mesh to FLAME 50-dimensional continuous expression coefficients.
        """
        bs = self.map_landmarks_to_blendshapes(face_landmarks)
        flame_vector = [0.0] * 50
        flame_vector[0] = bs["browInnerUp"]
        flame_vector[1] = bs["browInnerUp"] * 0.9
        flame_vector[2] = bs["browDownLeft"]
        flame_vector[3] = bs["mouthSmileLeft"]
        flame_vector[4] = bs["jawOpen"]
        flame_vector[5] = bs["eyeBlinkLeft"]
        return flame_vector


blendshape_mapper = SignAvatarsBlendshapeMapper()
FacialBlendshapeMapper = SignAvatarsBlendshapeMapper
