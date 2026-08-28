"""
Authentication & User Account Endpoints (/api/v1/auth) (§11, §12)
Part of Tereguwami (ተርጓሚ) API Gateway
"""

import time
from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from backend.api.schemas import (
    UserRegisterRequest, UserLoginRequest,
    UserLoginResponse, UserProfileResponse
)
from backend.auth.security import security_manager
from backend.auth.rbac import get_current_user, require_role, ROLE_HIERARCHY
from backend.db.session import get_db
from backend.db.models import User, AuditLog

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & User Management"])


@router.post("/register", response_model=UserLoginResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserRegisterRequest,
    db: Session = Depends(get_db)
):
    """
    Registers a new user (signer, researcher, or institutional client)
    with PBKDF2 hashed password storage and auto-issued access token.
    """
    # Check if username or email already exists
    existing_user = db.query(User).filter(
        (User.username == payload.username) | (User.email == payload.email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username or email address is already registered."
        )

    # Validate role
    role = payload.role or "registered_signer"
    if role not in ROLE_HIERARCHY:
        role = "registered_signer"

    hashed_pw = security_manager.hash_password(payload.password)
    new_user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hashed_pw,
        role=role,
        preferred_language=payload.preferred_language or "am",
        created_at=time.time()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Log audit event
    audit_log = AuditLog(
        action="USER_REGISTRATION",
        performed_by=new_user.username,
        details=f"User {new_user.username} registered with role '{role}'."
    )
    db.add(audit_log)
    db.commit()

    token = security_manager.create_access_token(user_id=new_user.username, role=role)

    return UserLoginResponse(
        access_token=token,
        token_type="Bearer",
        expires_in_seconds=security_manager.TOKEN_EXPIRATION_SECONDS,
        user_id=new_user.id,
        username=new_user.username,
        role=new_user.role,
        preferred_language=new_user.preferred_language
    )


@router.post("/login", response_model=UserLoginResponse)
async def login_user(
    payload: UserLoginRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticates user credentials and issues a signed JWT bearer access token.
    """
    user = db.query(User).filter(
        (User.username == payload.username_or_email) | (User.email == payload.username_or_email)
    ).first()

    if not user or not security_manager.verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password credentials."
        )

    token = security_manager.create_access_token(user_id=user.username, role=user.role)

    return UserLoginResponse(
        access_token=token,
        token_type="Bearer",
        expires_in_seconds=security_manager.TOKEN_EXPIRATION_SECONDS,
        user_id=user.id,
        username=user.username,
        role=user.role,
        preferred_language=user.preferred_language
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_my_profile(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns full profile details for the currently authenticated user session.
    """
    username = current_user.get("sub")
    if not username or username == "guest_anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token."
        )

    user = db.query(User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User record not found.")

    return UserProfileResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        preferred_language=user.preferred_language,
        created_at=user.created_at
    )


@router.get("/users", response_model=List[UserProfileResponse])
async def list_registered_users(
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role("deaf_advisory_board"))
):
    """
    Lists registered users for institutional oversight and ethics boards.
    Requires minimum role 'deaf_advisory_board' or 'admin'.
    """
    users = db.query(User).order_by(User.id.asc()).all()
    return [
        UserProfileResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            role=u.role,
            preferred_language=u.preferred_language,
            created_at=u.created_at
        )
        for u in users
    ]
