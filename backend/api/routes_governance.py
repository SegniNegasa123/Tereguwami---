"""
Deaf-Led Governance & Consent Enforcement Endpoints (/api/v1/governance) (§10.3, §16)
Part of Tereguwami (ተርጓሚ) API Gateway
"""

import time
import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.api.schemas import (
    ConsentVerificationRequest, ConsentVerificationResponse,
    ConsentWithdrawalRequest, ConsentWithdrawalResponse
)
from backend.auth.rbac import get_current_user, require_role
from backend.db.session import get_db
from backend.db.models import SignerConsent, AuditLog

router = APIRouter(prefix="/api/v1/governance", tags=["Governance & Ethics"])


@router.post("/consent/verify", response_model=ConsentVerificationResponse)
async def verify_signer_consent(
    payload: ConsentVerificationRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Verify whether a signer's informed consent is active and valid for dataset release.
    """
    record = db.query(SignerConsent).filter_by(signer_id=payload.signer_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Signer ID '{payload.signer_id}' not found in registry.")

    return ConsentVerificationResponse(
        signer_id=record.signer_id,
        consent_active=record.consent_active and not record.video_withdrawal_requested,
        withdrawal_requested=record.video_withdrawal_requested,
        governance_status="Deaf Advisory Board Verified" if record.consent_active else "Withdrawn / Inactive"
    )


@router.post("/consent/withdraw", response_model=ConsentWithdrawalResponse)
async def withdraw_signer_consent(
    payload: ConsentWithdrawalRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Enforce permanent, unconditional right to withdraw video footage and data from future dataset releases.
    """
    record = db.query(SignerConsent).filter_by(signer_id=payload.signer_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Signer ID '{payload.signer_id}' not found in registry.")

    record.video_withdrawal_requested = True
    record.consent_active = False
    record.withdrawal_date = time.time()

    audit_id = f"AUDIT_{uuid.uuid4().hex[:8].upper()}"
    log_entry = AuditLog(
        action="CONSENT_WITHDRAWAL",
        performed_by=current_user.get("sub", "anonymous_signer"),
        details=f"Signer {payload.signer_id} invoked withdrawal rights. Reason: {payload.reason or 'User requested'}. Token: {audit_id}"
    )
    db.add(log_entry)
    db.commit()

    return ConsentWithdrawalResponse(
        signer_id=payload.signer_id,
        status="withdrawn",
        message="Signer consent successfully withdrawn. Footage flagged for immediate quarantine and exclusion from future releases.",
        audit_id=audit_id
    )


@router.get("/audit-logs")
async def list_governance_audit_logs(
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role("deaf_advisory_board"))
):
    """
    Audit log inspection endpoint restricted to the Deaf Advisory Board and Institutional Overseers.
    """
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
    return [
        {
            "id": l.id,
            "action": l.action,
            "performed_by": l.performed_by,
            "details": l.details,
            "timestamp": l.timestamp
        }
        for l in logs
    ]
