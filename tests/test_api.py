"""
Unit Tests: FastAPI Backend REST Endpoints
Part of Tereguwami (ተርጓሚ) Automated Test Suite
"""

import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "Tereguwami" in data["project"]


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert len(data["active_layers"]) == 9


def test_translation_endpoint():
    # 20 frames of 543 3D landmarks
    dummy_keypoints = [[[0.5, 0.5, 0.0] for _ in range(543)] for _ in range(20)]
    payload = {
        "keypoints": dummy_keypoints,
        "target_language": "am",
        "domain_hint": "healthcare",
        "high_stakes_verification": False
    }
    resp = client.post("/api/v1/translate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "translated_text" in data
    assert data["target_language"] == "am"
    assert data["confidence_score"] > 0.0


def test_production_endpoint():
    payload = {
        "text_prompt": "መድኃኒቱን ከምግብ በፊት ይውሰዱ።",
        "source_language": "am",
        "signing_speed": 1.0
    }
    resp = client.post("/api/v1/produce", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_frames"] >= 30
    assert len(data["frames"]) == data["total_frames"]


def test_personalization_endpoints():
    seq = [[[0.1, 0.2, 0.3] for _ in range(543)] for _ in range(10)]
    enroll_payload = {
        "sign_name": "TEST_CUSTOM_SIGN",
        "exemplar_keypoints": [seq]
    }
    resp = client.post("/api/v1/personalize/enroll", json=enroll_payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "enrolled"

    # Query
    query_payload = {
        "keypoints": seq,
        "distance_threshold": 0.5
    }
    query_resp = client.post("/api/v1/personalize/query", json=query_payload)
    assert query_resp.status_code == 200
    assert query_resp.json()["matched"] is True

    # List
    list_resp = client.get("/api/v1/personalize/list")
    assert list_resp.status_code == 200
    assert any(item["sign_name"] == "TEST_CUSTOM_SIGN" for item in list_resp.json())


def test_silent_speech_endpoint():
    # 200 samples of 6 channels
    emg_data = [[0.05 * (ch + 1) for ch in range(6)] for _ in range(200)]
    resp = client.post("/api/v1/silent-speech/decode", json={"emg_signals": emg_data})
    assert resp.status_code == 200
    data = resp.json()
    assert "decoded_word" in data
    assert data["signal_channels"] == 6


def test_governance_endpoints():
    from backend.db.session import SessionLocal
    from backend.db.models import SignerConsent

    # Ensure clean test state for test signer
    test_id = "SIGNER_TEST_GOV"
    db = SessionLocal()
    try:
        record = db.query(SignerConsent).filter_by(signer_id=test_id).first()
        if not record:
            record = SignerConsent(signer_id=test_id, consent_active=True, video_withdrawal_requested=False)
            db.add(record)
        else:
            record.consent_active = True
            record.video_withdrawal_requested = False
        db.commit()
    finally:
        db.close()

    # 1. Verify active consent
    resp = client.post("/api/v1/governance/consent/verify", json={"signer_id": test_id})
    assert resp.status_code == 200
    assert resp.json()["consent_active"] is True

    # 2. Withdraw consent
    w_resp = client.post("/api/v1/governance/consent/withdraw", json={
        "signer_id": test_id,
        "reason": "Automated test withdrawal request"
    })
    assert w_resp.status_code == 200
    assert w_resp.json()["status"] == "withdrawn"

    # 3. Verify updated consent status
    v_resp = client.post("/api/v1/governance/consent/verify", json={"signer_id": test_id})
    assert v_resp.status_code == 200
    assert v_resp.json()["consent_active"] is False


def test_auth_endpoints():
    import uuid
    uid = uuid.uuid4().hex[:6]
    test_user = f"user_{uid}"
    test_email = f"{test_user}@tereguwami.org"
    test_pw = "SecureSignerPassword123!"

    # 1. Register user
    reg_resp = client.post("/api/v1/auth/register", json={
        "username": test_user,
        "email": test_email,
        "password": test_pw,
        "role": "registered_signer",
        "preferred_language": "am"
    })
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()
    assert "access_token" in reg_data
    assert reg_data["username"] == test_user
    token = reg_data["access_token"]

    # 2. Login user
    login_resp = client.post("/api/v1/auth/login", json={
        "username_or_email": test_user,
        "password": test_pw
    })
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert login_data["access_token"] is not None

    # 3. Inspect profile with Bearer auth
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == test_user


def test_leaderboard_endpoints():
    # 1. Fetch public leaderboard
    list_resp = client.get("/api/v1/leaderboard")
    assert list_resp.status_code == 200
    records = list_resp.json()
    assert len(records) >= 3
    assert records[0]["rank"] == 1

    # 2. Submit new evaluation
    sub_payload = {
        "model_name": "Test Continuous Graph Transformer",
        "organization": "Test Research Lab",
        "contact_email": "test@lab.edu.et",
        "signer_independent_acc": 89.5,
        "signer_dependent_acc": 97.1,
        "bleu_4": 33.2,
        "non_manual_f1": 88.0
    }
    sub_resp = client.post("/api/v1/leaderboard/submit", json=sub_payload)
    assert sub_resp.status_code == 200
    assert sub_resp.json()["model_name"] == sub_payload["model_name"]


def test_frame_translation_endpoint():
    frame_payload = {
        "image_base64": None,
        "target_language": "am",
        "domain_hint": "healthcare",
        "high_stakes_verification": False
    }
    resp = client.post("/api/v1/translate/frame", json=frame_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "translated_text" in data
    assert data["target_language"] == "am"
    assert data["confidence_score"] > 0.0


