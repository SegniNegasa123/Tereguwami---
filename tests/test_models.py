"""
Unit Tests: AI/ML Models & Preprocessing Modules
Part of Tereguwami (ተርጓሚ) Automated Test Suite
"""

import pytest
import numpy as np
from data.geez_preprocessor import geez_preprocessor
from data.esl_dataset_loader import esl_dataset
from models.perception.keypoint_normalizer import keypoint_normalizer
from models.recognition.temporal_models import temporal_recognizer
from models.translation.gloss_free_transformer import continuous_translator
from models.translation.constrained_decoder import constrained_decoder
from models.production.progressive_transformer import avatar_production_engine
from models.personalization.siamese_few_shot import sign_personalizer
from models.personalization.federated_client import create_federated_client
from models.silent_speech.emg_decoder import silent_speech_decoder


def test_geez_preprocessor():
    raw = "ዶክተር፡ብርቱ፤የራስ-ምታት፡አለኝ።"
    cleaned = geez_preprocessor.clean(raw)
    assert "፡" not in cleaned
    assert "፤" not in cleaned
    assert "ዶክተር" in cleaned
    assert "አለኝ" in cleaned


def test_keypoint_normalizer_spatial():
    dummy_landmarks = np.ones((10, 543, 3), dtype=np.float32) * 0.5
    dummy_landmarks[:, 11] = [0.4, 0.5, 0.0]  # Left shoulder
    dummy_landmarks[:, 12] = [0.6, 0.5, 0.0]  # Right shoulder

    normalized = keypoint_normalizer.normalize_spatial(dummy_landmarks)
    assert normalized.shape == (10, 543, 3)
    # Shoulders should be centered around origin x=0
    midpoint_x = (normalized[:, 11, 0] + normalized[:, 12, 0]) / 2.0
    assert np.allclose(midpoint_x, 0.0, atol=1e-3)


def test_keypoint_normalizer_derivatives():
    seq = np.zeros((20, 543, 3), dtype=np.float32)
    for t in range(20):
        seq[t] = t * 0.1
    vel, acc = keypoint_normalizer.compute_derivatives(seq)
    assert vel.shape == seq.shape
    assert acc.shape == seq.shape
    assert np.allclose(vel[1:], 0.1, atol=1e-4)


def test_temporal_recognizer():
    dummy_seq = np.random.randn(30, 1629).astype(np.float32)
    res = temporal_recognizer.predict(dummy_seq)
    assert "predicted_class" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["candidates"]) == 3


def test_continuous_translator_and_constrained_decoder():
    dummy_feat = np.random.randn(60, 1629).astype(np.float32)
    trans = continuous_translator.translate(dummy_feat, target_lang="am")
    assert "translated_text" in trans
    assert trans["confidence_score"] > 0.5

    guarded = constrained_decoder.decode_with_constraints(
        candidate_text=trans["translated_text"],
        confidence_score=trans["confidence_score"],
        recognized_glosses=["DOCTOR", "HEADACHE"],
        domain="healthcare"
    )
    assert guarded["is_faithful"] is True
    assert guarded["final_text"] == trans["translated_text"]


def test_avatar_production():
    res = avatar_production_engine.generate_avatar_stream("ጤና ይስጥልኝ እንደምን ነዎት?", source_lang="am")
    assert res["total_frames"] >= 30
    assert len(res["frames"]) == res["total_frames"]
    assert "browInnerUp" in res["frames"][0]["blendshapes"]
    assert "mouthSmile" in res["frames"][0]["blendshapes"]


def test_few_shot_personalization():
    seq1 = np.random.randn(40, 1629).astype(np.float32)
    seq2 = seq1 + np.random.normal(0, 0.01, size=seq1.shape).astype(np.float32)
    enroll_res = sign_personalizer.enroll_sign("FAMILY_SIGN_HOME", [seq1, seq2])
    assert enroll_res["shots_enrolled"] == 2

    # Query with identical sequence
    query_res = sign_personalizer.recognize_custom_sign(seq1, distance_threshold=0.5)
    assert query_res["matched"] is True
    assert query_res["sign_name"] == "FAMILY_SIGN_HOME"


def test_federated_client_dp():
    client = create_federated_client("CLIENT_ETH_01")
    dummy_data = [np.random.randn(128).astype(np.float32) for _ in range(5)]
    weights, samples, meta = client.fit(dummy_data)
    assert samples == 5
    assert meta["noise_injected"] is True
    assert len(weights) == 1
    assert weights[0].shape == (128,)


def test_silent_speech_emg():
    emg_window = np.random.randn(500, 6).astype(np.float32)
    res = silent_speech_decoder.decode_subvocalization(emg_window)
    assert res["decoded_word"] in silent_speech_decoder.vocabulary
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["signal_channels"] == 6
