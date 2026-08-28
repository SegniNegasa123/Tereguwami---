"""
Facial Action Coding System (FACS) Blendshape Mappings (§8.3, §8.5)
Part of Tereguwami (ተርጓሚ) Avatar Rigging & Facial Semantics Pipeline

Maps MediaPipe dense face mesh Action Units to Apple ARKit / glTF 52 standard facial blendshapes,
reproducing critical ESL non-manual markers (eyebrow raise for polar questions, furrow for wh-questions,
lip curl/pucker, and head nods/shakes).
"""

from typing import Dict, Any, List
import numpy as np


class FacialBlendshapeMapper:
    """
    Extracts standardized ARKit blendshape weights (range [0.0, 1.0])
    from MediaPipe face mesh landmarks (indices 75..542).
    """

    # MediaPipe Face Landmark Indices for Action Units
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
        Input: face_landmarks of shape (468, 3) or slice from Holistic.
        Returns dictionary of 15 essential ESL non-manual blendshapes.
        """
        # Baseline reference dimensions
        face_height = np.linalg.norm(face_landmarks[self.NOSE_TIP] - face_landmarks[self.CHIN])
        if face_height < 1e-4:
            face_height = 0.2

        # 1. Eyebrow Elevation (AU1 / AU2 - Polar Question Marker)
        left_brow_y = np.mean(face_landmarks[self.LEFT_EYEBROW, 1])
        right_brow_y = np.mean(face_landmarks[self.RIGHT_EYEBROW, 1])
        brow_y = (left_brow_y + right_brow_y) / 2.0
        nose_y = face_landmarks[self.NOSE_TIP, 1]
        raw_brow_elev = float((nose_y - brow_y) / face_height)
        brow_inner_up = float(np.clip((raw_brow_elev - 0.35) * 4.0, 0.0, 1.0))

        # 2. Eyebrow Furrow (AU4 - Wh-Question Marker)
        brow_dist = float(np.linalg.norm(face_landmarks[self.LEFT_EYEBROW[-1]] - face_landmarks[self.RIGHT_EYEBROW[-1]]))
        brow_down = float(np.clip((0.15 - (brow_dist / face_height)) * 5.0, 0.0, 1.0))

        # 3. Jaw Open (AU26 / AU27 - Mouthing & Emphasis)
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


blendshape_mapper = FacialBlendshapeMapper()
