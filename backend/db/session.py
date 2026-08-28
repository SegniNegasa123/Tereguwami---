"""
Database Session Management
Part of Tereguwami (ተርጓሚ) Persistence Tier (§11)
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.models import Base

# Defaults to local SQLite file database for lightweight portability; seamlessly switches to PostgreSQL via env var
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///tereguwami.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI database dependency yields session and safely closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
