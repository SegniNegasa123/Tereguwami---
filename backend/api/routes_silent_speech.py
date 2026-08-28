"""
Silent-Speech Neuromotor Endpoints (/api/v1/silent-speech) (§8.7)
Part of Tereguwami (ተርጓሚ) API Gateway
"""

from fastapi import APIRouter, HTTPException, Depends
import numpy as np
from backend.api.schemas import SilentSpeechRequest, SilentSpeechResponse
from backend.auth.rbac import get_current_user
from models.silent_speech.emg_decoder import silent_speech_decoder

router = APIRouter(prefix="/api/v1/silent-speech", tags=["Silent Speech"])


@router.post("/decode", response_model=SilentSpeechResponse)
async def decode_subvocal_emg(
    payload: SilentSpeechRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Decodes a window of 6-channel surface electromyography (sEMG) voltages
    from facial/jaw subvocalizations into intended closed-vocabulary speech tokens.
    """
    raw_signals = payload.emg_signals
    if not raw_signals or len(raw_signals) < 100:
        raise HTTPException(
            status_code=400,
            detail="At least 100 sEMG samples (0.1s at 1000 Hz) are required for decoding."
        )

    arr = np.array(raw_signals, dtype=np.float32)
    if arr.shape[1] != 6:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 6 differential electrode channels, received {arr.shape[1]}"
        )

    result = silent_speech_decoder.decode_subvocalization(arr)
    return SilentSpeechResponse(**result)
