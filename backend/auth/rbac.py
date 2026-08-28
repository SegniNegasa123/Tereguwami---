"""
Role-Based Access Control (RBAC) Guardrails
Part of Tereguwami (ተርጓሚ) Auth Infrastructure (§11, §16)
"""

from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.auth.security import security_manager

security_scheme = HTTPBearer(auto_error=False)

ROLE_HIERARCHY = {
    "anonymous": 0,
    "registered_signer": 1,
    "researcher": 2,
    "deaf_advisory_board": 3,
    "institutional_client": 4,
    "admin": 5
}


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)) -> Dict[str, Any]:
    """Extract authenticated user context from bearer token; defaults to anonymous."""
    if not credentials:
        return {"sub": "guest_anonymous", "role": "anonymous"}

    payload = security_manager.decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")

    return payload


def require_role(min_role: str):
    """Dependency enforcing minimum authorization level."""
    def role_checker(user: Dict[str, Any] = Depends(get_current_user)):
        user_role = user.get("role", "anonymous")
        user_level = ROLE_HIERARCHY.get(user_role, 0)
        required_level = ROLE_HIERARCHY.get(min_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Requires minimum role '{min_role}', current role is '{user_role}'"
            )
        return user
    return role_checker
