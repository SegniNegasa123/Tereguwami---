"""
Pydantic API Schemas & Data Transfer Objects
Part of Tereguwami (ተርጓሚ) FastAPI Services (§8, §11)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==========================================
# 1. Translation Schemas
# ==========================================
class TranslationRequest(BaseModel):
    keypoints: List[List[List[float]]] = Field(
        ...,
        description="Temporal sequence of 543 3D normalized landmarks of shape [T, 543, 3]"
    )
    target_language: str = Field(
        default="am",
        description="Target output language code ('am' for Amharic, 'om' for Afaan Oromo, 'en' for English)"
    )
    domain_hint: Optional[str] = Field(
        default=None,
        description="Optional domain context ('healthcare', 'legal_court', 'education', 'everyday_civic')"
    )
    high_stakes_verification: bool = Field(
        default=False,
        description="Enable medical/legal high-stakes verification mode with strict thresholds"
    )


class TranslationResponse(BaseModel):
    translated_text: str
    target_language: str
    confidence_score: float
    status: str
    is_faithful: bool
    requires_clarification: bool
    requires_human_verification: bool
    frame_count: int
    matched_template: Optional[str] = None


# ==========================================
# 2. Production (Reverse Channel) Schemas
# ==========================================
class AvatarProductionRequest(BaseModel):
    text_prompt: str = Field(..., description="Sentence in Amharic, Afaan Oromo, or English to synthesize into sign")
    source_language: str = Field(default="am", description="Source language code ('am', 'om', 'en')")
    signing_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Signing playback speed multiplier")
    avatar_id: str = Field(default="esl_standard_avatar", description="Identifier of the 3D avatar model")


class AvatarFrameData(BaseModel):
    frame_index: int
    timestamp_ms: int
    left_hand: Dict[str, Any]
    right_hand: Dict[str, Any]
    blendshapes: Dict[str, float]
    smplx: Optional[Dict[str, Any]] = None


class AvatarProductionResponse(BaseModel):
    text_prompt: str
    source_language: str
    total_frames: int
    fps: int
    duration_seconds: float
    is_question: bool
    is_negation: bool
    model_architecture: Optional[str] = "SignAvatars_SMPLX_Holistic"
    frames: List[AvatarFrameData]



# ==========================================
# 3. Personalization Schemas
# ==========================================
class SignEnrollmentRequest(BaseModel):
    sign_name: str = Field(..., description="User label for the personal or community sign")
    exemplar_keypoints: List[List[List[List[float]]]] = Field(
        ...,
        description="List of 1 to 5 exemplar keypoint sequences, each of shape [T_i, 543, 3]"
    )


class SignEnrollmentResponse(BaseModel):
    sign_name: str
    shots_enrolled: int
    status: str
    embedding_dimension: int


class CustomSignQueryRequest(BaseModel):
    keypoints: List[List[List[float]]] = Field(..., description="Query keypoint sequence [T, 543, 3]")
    distance_threshold: float = Field(default=0.45, description="Maximum metric distance for positive match")


class CustomSignQueryResponse(BaseModel):
    matched: bool
    sign_name: Optional[str] = None
    confidence: Optional[float] = None
    metric_distance: float


# ==========================================
# 4. Silent-Speech (sEMG) Schemas
# ==========================================
class SilentSpeechRequest(BaseModel):
    emg_signals: List[List[float]] = Field(
        ...,
        description="2D array of sEMG signal readings of shape [Samples, 6 Channels] sampled at 1000 Hz"
    )


class SilentSpeechResponse(BaseModel):
    decoded_word: str
    confidence: float
    signal_channels: int
    sample_count: int
    is_calibrated: bool
    channel_type: str


# ==========================================
# 5. Governance & Consent Schemas
# ==========================================
class ConsentVerificationRequest(BaseModel):
    signer_id: str


class ConsentVerificationResponse(BaseModel):
    signer_id: str
    consent_active: bool
    withdrawal_requested: bool
    governance_status: str


class ConsentWithdrawalRequest(BaseModel):
    signer_id: str
    reason: Optional[str] = None


class ConsentWithdrawalResponse(BaseModel):
    signer_id: str
    status: str
    message: str
    audit_id: str


# ==========================================
# 6. System Health & Telemetry Schemas
# ==========================================
class SystemHealthResponse(BaseModel):
    status: str
    project_name: str
    version: str
    active_layers: List[str]
    gpu_available: bool
    inference_latency_cpu_ms: float


# ==========================================
# 7. Authentication & User Profile Schemas
# ==========================================
class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="Unique username")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Raw password to be PBKDF2 hashed")
    role: Optional[str] = Field(default="registered_signer", description="Assigned role in RBAC hierarchy")
    preferred_language: Optional[str] = Field(default="am", description="Preferred interface language ('am', 'om', 'en')")


class UserLoginRequest(BaseModel):
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., description="Plaintext password to authenticate")


class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in_seconds: int
    user_id: int
    username: str
    role: str
    preferred_language: str


class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    preferred_language: str
    created_at: float


# ==========================================
# 8. Frame Translation & Landmark Extraction Schemas
# ==========================================
class DirectFrameTranslationRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded RGB camera frame image (JPEG/PNG)")
    target_language: str = Field(default="am", description="Target output language code")
    domain_hint: Optional[str] = Field(default=None, description="Domain context hint")
    high_stakes_verification: bool = Field(default=False, description="Enable clinical/legal verification")


# ==========================================
# 9. Leaderboard Submission Schemas
# ==========================================
class LeaderboardSubmissionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the evaluated model")
    organization: str = Field(..., description="Submitting research team or institution")
    contact_email: str = Field(..., description="Contact email of author")
    signer_independent_acc: float = Field(..., ge=0.0, le=100.0, description="Accuracy on unseen signers")
    signer_dependent_acc: float = Field(..., ge=0.0, le=100.0, description="Accuracy on seen signers")
    bleu_4: float = Field(..., ge=0.0, le=100.0, description="Continuous translation BLEU-4 score")
    non_manual_f1: float = Field(..., ge=0.0, le=100.0, description="Grammatical marker F1 score")


class LeaderboardRecordResponse(BaseModel):
    rank: int
    model_name: str
    organization: str
    signer_independent_acc: float
    signer_dependent_acc: float
    generalization_gap: float
    bleu_4: float
    non_manual_f1: float
    date: str

