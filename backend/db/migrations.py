"""
Database Initialization & Data Seeding
Part of Tereguwami (ተርጓሚ) Persistence Tier
"""

import time
from backend.db.session import init_db, SessionLocal
from backend.db.models import User, BenchmarkSubmission, SignerConsent, AuditLog
from backend.auth.security import security_manager


def run_migrations_and_seed():
    """Create all schemas and populate benchmark baselines and sample accounts."""
    init_db()
    db = SessionLocal()
    try:
        # 1. Seed Deaf Advisory Board and Admin accounts if not present
        if not db.query(User).filter_by(username="deaf_advisory_lead").first():
            advisory_user = User(
                username="deaf_advisory_lead",
                email="board@tereguwami.org",
                hashed_password=security_manager.hash_password("Tereguwami2026!"),
                role="deaf_advisory_board",
                preferred_language="am"
            )
            db.add(advisory_user)

        if not db.query(User).filter_by(username="institutional_broadcaster").first():
            broadcaster = User(
                username="institutional_broadcaster",
                email="api@broadcaster.et",
                hashed_password=security_manager.hash_password("BroadcastRelay2026!"),
                role="institutional_client",
                preferred_language="am"
            )
            db.add(broadcaster)

        # 2. Seed Baseline Benchmark Submissions (§6.1 Table 2 & Slide 4)
        if not db.query(BenchmarkSubmission).filter_by(model_name="2025 Ethiopian Skeleton Study Baseline").first():
            sub_2025 = BenchmarkSubmission(
                submitter_name="Abeje et al. (AAU / Nature Portfolio)",
                model_name="2025 Ethiopian Skeleton Study Baseline",
                bleu_4=18.4,
                signer_independent_acc=73.0,
                signer_dependent_acc=94.0,
                non_manual_f1=62.5,
                status="published_baseline"
            )
            db.add(sub_2025)

        if not db.query(BenchmarkSubmission).filter_by(model_name="Tereguwami Multimodal Transformer (Ours)").first():
            sub_tereguwami = BenchmarkSubmission(
                submitter_name="Tereguwami Research Team",
                model_name="Tereguwami Multimodal Transformer (Ours)",
                bleu_4=31.8,
                signer_independent_acc=88.2,
                signer_dependent_acc=96.5,
                non_manual_f1=87.6,
                status="active_sota"
            )
            db.add(sub_tereguwami)

        # 3. Seed Signer Consents
        for i in range(1, 101):
            s_id = f"SIGNER_{i:02d}"
            if not db.query(SignerConsent).filter_by(signer_id=s_id).first():
                consent = SignerConsent(
                    signer_id=s_id,
                    consent_active=True,
                    video_withdrawal_requested=False,
                    signed_date=time.time()
                )
                db.add(consent)

        # 4. Audit Log
        db.add(AuditLog(
            action="SYSTEM_INIT",
            performed_by="system",
            details="Initialized Tereguwami database with standard schemas and benchmark baselines."
        ))

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    run_migrations_and_seed()
    print("Database migrations and seeding completed successfully.")
