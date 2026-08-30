"""
Tereguwami Layer 4: Facial Blendshape Mapper
Converts Non-Manual Grammatical Action Units into ARKit blendshapes present in the RPM GLB model.
"""


def map_facs_to_arkit(au_vector: dict) -> dict:
    """
    Transforms AU1/2/4 and mouth kinematics into ARKit blendshape weights (0.0 to 1.0).
    """
    arkit_weights = {}

    # Non-Manual Question Markers (AU1 + AU2: Brow Raise)
    if "AU1" in au_vector:
        arkit_weights["browInnerUp"] = max(0.0, min(1.0, float(au_vector["AU1"])))
    if "AU2" in au_vector:
        val = max(0.0, min(1.0, float(au_vector["AU2"])))
        arkit_weights["browOuterUpLeft"] = val
        arkit_weights["browOuterUpRight"] = val

    # Non-Manual Negation & Intensity Markers (AU4: Brow Furrow / Corrugator)
    if "AU4" in au_vector:
        val = max(0.0, min(1.0, float(au_vector["AU4"])))
        arkit_weights["browDownLeft"] = val
        arkit_weights["browDownRight"] = val

    # Mouthings & Mouthing Morphemes
    if "jaw_open" in au_vector:
        arkit_weights["jawOpen"] = max(0.0, min(1.0, float(au_vector["jaw_open"])))
    if "lip_pucker" in au_vector:
        arkit_weights["mouthPucker"] = max(0.0, min(1.0, float(au_vector["lip_pucker"])))

    return arkit_weights
