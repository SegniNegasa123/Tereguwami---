"""
Avatar Production Reverse-Channel Endpoints (/api/v1/produce) (§8.5)
Part of Tereguwami (ተርጓሚ) API Gateway
"""

from fastapi import APIRouter, HTTPException, Depends
from backend.api.schemas import AvatarProductionRequest, AvatarProductionResponse, AvatarFrameData
from backend.auth.rbac import get_current_user
from models.production.progressive_transformer import avatar_production_engine

router = APIRouter(prefix="/api/v1/produce", tags=["Avatar Production"])


@router.post("", response_model=AvatarProductionResponse)
async def produce_avatar_animation(
    payload: AvatarProductionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generates a continuous 3D skeletal and blendshape avatar animation stream
    from spoken or written text input (Amharic, Afaan Oromo, English).
    """
    prompt = payload.text_prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Text prompt cannot be empty.")

    result = avatar_production_engine.generate_avatar_stream(
        text_input=prompt,
        source_lang=payload.source_language,
        signing_speed=payload.signing_speed
    )

    frames = [
        AvatarFrameData(
            frame_index=f["frame_index"],
            timestamp_ms=f["timestamp_ms"],
            left_hand=f["left_hand"],
            right_hand=f["right_hand"],
            blendshapes=f["blendshapes"],
            smplx=f.get("smplx")
        )
        for f in result["frames"]
    ]

    return AvatarProductionResponse(
        text_prompt=result["text_prompt"],
        source_language=result["source_language"],
        total_frames=result["total_frames"],
        fps=result["fps"],
        duration_seconds=result["duration_seconds"],
        is_question=result["is_question"],
        is_negation=result["is_negation"],
        model_architecture=result.get("model_architecture", "SignAvatars_SMPLX_Holistic"),
        frames=frames
    )

