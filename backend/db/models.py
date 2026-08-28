"""
SQLAlchemy Relational Database Models (§8, §11)
Part of Tereguwami (ተርጓሚ) Persistence Tier
"""

import time
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="registered_signer")  # anonymous, registered_signer, researcher, deaf_advisory_board, institutional_client
    preferred_language = Column(String(10), default="am")
    created_at = Column(Float, default=time.time)


class EnrolledSign(Base):
    __tablename__ = "enrolled_signs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    sign_name = Column(String(64), index=True, nullable=False)
    embedding_json = Column(Text, nullable=False)  # Stored as serialized vector; maps to pgvector in PostgreSQL
    shots_count = Column(Integer, default=1)
    created_at = Column(Float, default=time.time)


class BenchmarkSubmission(Base):
    __tablename__ = "benchmark_submissions"

    id = Column(Integer, primary_key=True, index=True)
    submitter_name = Column(String(128), nullable=False)
    model_name = Column(String(128), nullable=False)
    bleu_4 = Column(Float, nullable=False)
    signer_independent_acc = Column(Float, nullable=False)
    signer_dependent_acc = Column(Float, nullable=True)
    non_manual_f1 = Column(Float, nullable=False)
    status = Column(String(32), default="verified")
    created_at = Column(Float, default=time.time)


class SignerConsent(Base):
    __tablename__ = "signer_consents"

    id = Column(Integer, primary_key=True, index=True)
    signer_id = Column(String(64), unique=True, index=True, nullable=False)
    consent_active = Column(Boolean, default=True)
    video_withdrawal_requested = Column(Boolean, default=False)
    signed_date = Column(Float, default=time.time)
    withdrawal_date = Column(Float, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(128), nullable=False)
    performed_by = Column(String(64), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(Float, default=time.time)
