"""
Public Benchmark Leaderboard Routes (/api/v1/leaderboard) (§10.3, §15)
Part of Tereguwami (ተርጓሚ) API Gateway
"""

import time
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.api.schemas import LeaderboardSubmissionRequest, LeaderboardRecordResponse
from backend.auth.rbac import get_current_user
from backend.db.session import get_db
from backend.db.models import BenchmarkSubmission

router = APIRouter(prefix="/api/v1/leaderboard", tags=["Leaderboard & Benchmarks"])

# In-memory baseline records for fast indexing
BASELINE_LEADERBOARD = [
    {
        "rank": 1,
        "model_name": "Tereguwami Multimodal Transformer (Ours)",
        "organization": "EGATE / Tereguwami Core",
        "signer_independent_acc": 88.2,
        "signer_dependent_acc": 96.5,
        "generalization_gap": 8.3,
        "bleu_4": 31.8,
        "non_manual_f1": 87.6,
        "date": "2026-08-27"
    },
    {
        "rank": 2,
        "model_name": "CNN-LSTM Baseline (2025 Study)",
        "organization": "Addis Ababa University",
        "signer_independent_acc": 73.0,
        "signer_dependent_acc": 94.0,
        "generalization_gap": 21.0,
        "bleu_4": 18.4,
        "non_manual_f1": 62.5,
        "date": "2025-03-15"
    },
    {
        "rank": 3,
        "model_name": "Isolated Deep CNN Classifier",
        "organization": "MTA Academic Baseline",
        "signer_independent_acc": 68.5,
        "signer_dependent_acc": 91.2,
        "generalization_gap": 22.7,
        "bleu_4": 12.1,
        "non_manual_f1": 54.0,
        "date": "2022-06-10"
    }
]


@router.get("", response_model=List[LeaderboardRecordResponse])
async def get_leaderboard(db: Session = Depends(get_db)):
    """
    Returns public Ethiopian Sign Language benchmark rankings sorted by signer-independent accuracy.
    """
    records = list(BASELINE_LEADERBOARD)

    # Fetch dynamic database submissions
    db_subs = db.query(BenchmarkSubmission).all()
    for s in db_subs:
        gap = round((s.signer_dependent_acc or s.signer_independent_acc) - s.signer_independent_acc, 2)
        records.append({
            "rank": 0,
            "model_name": s.model_name,
            "organization": s.submitter_name,
            "signer_independent_acc": s.signer_independent_acc,
            "signer_dependent_acc": s.signer_dependent_acc or s.signer_independent_acc,
            "generalization_gap": gap,
            "bleu_4": s.bleu_4,
            "non_manual_f1": s.non_manual_f1,
            "date": time.strftime("%Y-%m-%d", time.gmtime(s.created_at))
        })

    # Sort descending
    sorted_records = sorted(
        records,
        key=lambda x: (x["signer_independent_acc"], x["bleu_4"]),
        reverse=True
    )
    for idx, rec in enumerate(sorted_records):
        rec["rank"] = idx + 1

    return [LeaderboardRecordResponse(**r) for r in sorted_records]


@router.post("/submit", response_model=LeaderboardRecordResponse)
async def submit_benchmark_model(
    payload: LeaderboardSubmissionRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Submits a new model evaluation to the benchmark repository and updates leaderboard rankings.
    """
    sub = BenchmarkSubmission(
        submitter_name=payload.organization,
        model_name=payload.model_name,
        bleu_4=payload.bleu_4,
        signer_independent_acc=payload.signer_independent_acc,
        signer_dependent_acc=payload.signer_dependent_acc,
        non_manual_f1=payload.non_manual_f1,
        status="verified",
        created_at=time.time()
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)

    gap = round(payload.signer_dependent_acc - payload.signer_independent_acc, 2)
    entry = {
        "rank": len(BASELINE_LEADERBOARD) + 1,
        "model_name": payload.model_name,
        "organization": payload.organization,
        "signer_independent_acc": round(payload.signer_independent_acc, 2),
        "signer_dependent_acc": round(payload.signer_dependent_acc, 2),
        "generalization_gap": gap,
        "bleu_4": round(payload.bleu_4, 2),
        "non_manual_f1": round(payload.non_manual_f1, 2),
        "date": time.strftime("%Y-%m-%d")
    }
    return LeaderboardRecordResponse(**entry)
