"""
Personalization & Few-Shot Sign Enrollment Endpoints (/api/v1/personalize) (§8.6)
Part of Tereguwami (ተርጓሚ) API Gateway
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import numpy as np
from backend.api.schemas import (
    SignEnrollmentRequest, SignEnrollmentResponse,
    CustomSignQueryRequest, CustomSignQueryResponse
)
from backend.auth.rbac import get_current_user
from models.personalization.siamese_few_shot import sign_personalizer

router = APIRouter(prefix="/api/v1/personalize", tags=["Personalization"])


@router.post("/enroll", response_model=SignEnrollmentResponse)
async def enroll_custom_sign(
    payload: SignEnrollmentRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Enrolls a personal, family-specific, or regional sign variation using 1-to-5 exemplar demonstrations.
    """
    sign_name = payload.sign_name.strip()
    if not sign_name:
        raise HTTPException(status_code=400, detail="Sign name cannot be empty.")

    if not payload.exemplar_keypoints:
        raise HTTPException(status_code=400, detail="At least one exemplar sequence is required.")

    exemplar_arrays = []
    for seq in payload.exemplar_keypoints:
        arr = np.array(seq, dtype=np.float32)
        if len(arr.shape) == 3:
            arr = arr.reshape(arr.shape[0], -1)
        exemplar_arrays.append(arr)

    result = sign_personalizer.enroll_sign(sign_name, exemplar_arrays)
    return SignEnrollmentResponse(**result)


@router.post("/query", response_model=CustomSignQueryResponse)
async def query_custom_sign(
    payload: CustomSignQueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Query a keypoint gesture sequence against enrolled custom sign prototypes.
    """
    arr = np.array(payload.keypoints, dtype=np.float32)
    if len(arr.shape) == 3:
        arr = arr.reshape(arr.shape[0], -1)

    result = sign_personalizer.recognize_custom_sign(arr, distance_threshold=payload.distance_threshold)
    return CustomSignQueryResponse(**result)


@router.get("/list")
async def list_enrolled_signs(current_user: dict = Depends(get_current_user)):
    """List all custom signs enrolled in the current active profile."""
    return sign_personalizer.list_enrolled_signs()
