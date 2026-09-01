"""
Translation Endpoints (/api/v1/translate) (§8.4)
Part of Tereguwami (ተርጓሚ) API Gateway
"""

from fastapi import APIRouter, HTTPException, Depends
import numpy as np
from backend.api.schemas import (
    TranslationRequest, TranslationResponse,
    DirectFrameTranslationRequest
)
from backend.auth.rbac import get_current_user
from models.translation.gloss_free_transformer import continuous_translator
from models.translation.constrained_decoder import constrained_decoder

router = APIRouter(prefix="/api/v1/translate", tags=["Translation"])


@router.post("", response_model=TranslationResponse)
async def translate_keypoints(
    payload: TranslationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Translates a continuous sequence of 543 3D normalized skeletal keypoints
    into fluent text in Amharic, Afaan Oromo, or English.
    """
    raw_keypoints = payload.keypoints
    if not raw_keypoints or len(raw_keypoints) < 5:
        raise HTTPException(
            status_code=400,
            detail="At least 5 frames of keypoints are required for continuous translation."
        )

    # Convert to NumPy array
    try:
        arr = np.array(raw_keypoints, dtype=np.float32)
        # Flatten spatial dimensions: (T, 543, 3) -> (T, 1629)
        T = arr.shape[0]
        flattened = arr.reshape(T, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid keypoint tensor shape: {str(e)}")

    # Execute translation through continuous translation engine
    raw_translation = continuous_translator.translate(
        keypoint_features=flattened,
        target_lang=payload.target_language,
        domain_hint=payload.domain_hint
    )

    # Apply safety decoding constraints (§8.4)
    constrained_result = constrained_decoder.decode_with_constraints(
        candidate_text=raw_translation["translated_text"],
        confidence_score=raw_translation["confidence_score"],
        recognized_glosses=[raw_translation.get("matched_template", "SIGN")],
        domain=payload.domain_hint or "everyday_civic"
    )

    return TranslationResponse(
        translated_text=constrained_result["final_text"],
        subtitle_text=raw_translation.get("subtitle_text"),
        target_language=payload.target_language,
        confidence_score=constrained_result["confidence_score"],
        status=raw_translation["status"],
        is_faithful=constrained_result["is_faithful"],
        requires_clarification=constrained_result["requires_clarification"],
        requires_human_verification=constrained_result["requires_human_verification"] or payload.high_stakes_verification,
        frame_count=T,
        matched_template=raw_translation.get("matched_template")
    )


@router.post("/frame", response_model=TranslationResponse)
async def translate_single_frame(
    payload: DirectFrameTranslationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Extracts 543 3D spatial landmarks from an uploaded image frame
    and returns immediate incremental translation analysis.
    """
    import base64
    import io
    from models.perception.mediapipe_extractor import mediapipe_extractor
    from models.perception.keypoint_normalizer import keypoint_normalizer

    # Create frame sequence from base64 image or synthetic landmark trajectory
    img_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    if payload.image_base64:
        try:
            # Strip data URL prefix if present
            b64_data = payload.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_data)
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                img_rgb = np.array(img)
            except Exception:
                pass
        except Exception:
            pass

    # Extract 543 landmarks from frame
    raw_landmarks = mediapipe_extractor.process_frame(img_rgb)
    norm_landmarks = keypoint_normalizer.normalize_spatial(raw_landmarks)
    if len(norm_landmarks.shape) == 3:
        norm_landmarks = norm_landmarks[0]

    # Build sequence window (T=20 frames) for continuous translator
    seq_frames = []
    for i in range(20):
        jittered = norm_landmarks.copy()
        jittered[:, :2] += np.sin(i * 0.2) * 0.01
        seq_frames.append(jittered.flatten())
    seq_array = np.array(seq_frames, dtype=np.float32)

    raw_translation = continuous_translator.translate(
        keypoint_features=seq_array,
        target_lang=payload.target_language,
        domain_hint=payload.domain_hint
    )

    constrained_result = constrained_decoder.decode_with_constraints(
        candidate_text=raw_translation["translated_text"],
        confidence_score=raw_translation["confidence_score"],
        recognized_glosses=[raw_translation.get("matched_template", "SIGN")],
        domain=payload.domain_hint or "everyday_civic"
    )

    return TranslationResponse(
        translated_text=constrained_result["final_text"],
        subtitle_text=raw_translation.get("subtitle_text"),
        target_language=payload.target_language,
        confidence_score=constrained_result["confidence_score"],
        status=raw_translation["status"],
        is_faithful=constrained_result["is_faithful"],
        requires_clarification=constrained_result["requires_clarification"],
        requires_human_verification=constrained_result["requires_human_verification"] or payload.high_stakes_verification,
        frame_count=20,
        matched_template=raw_translation.get("matched_template")
    )


@router.get("/vocalize")
async def vocalize_text(text: str, lang: str = "am"):
    """
    Synthesizes and streams crystal-clear audio/mpeg for Amharic, Afaan Oromoo, or English.
    Bypasses browser referer anti-hotlink blocking with fast edge streaming.
    """
    import urllib.parse
    import urllib.request
    from fastapi.responses import Response

    clean_text = (text or "").strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")

    lang_code = "om" if lang == "om" else ("am" if lang == "am" else "en")
    encoded = urllib.parse.quote(clean_text)
    tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={lang_code}&client=tw-ob&q={encoded}"

    req = urllib.request.Request(
        tts_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            audio_bytes = resp.read()
            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": "inline"
                }
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS synthesis failed: {str(e)}")


