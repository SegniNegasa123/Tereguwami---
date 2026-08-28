"""
Unit Tests: Real-Time Streaming & WebSocket Pipeline
Part of Tereguwami (ተርጓሚ) Automated Test Suite
"""

import json
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_websocket_ping_pong():
    with client.websocket_connect("/ws/stream/TEST_ROOM_01") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        data = ws.receive_text()
        parsed = json.loads(data)
        assert parsed["type"] == "pong"


def test_websocket_frame_ingestion():
    with client.websocket_connect("/ws/stream/TEST_ROOM_02") as ws:
        # Send 30 frames
        for i in range(30):
            frame = {
                "type": "frame",
                "frame_index": i,
                "timestamp_ms": i * 40,
                "landmarks": [[0.5, 0.5, 0.0] for _ in range(543)]
            }
            ws.send_text(json.dumps(frame))

        # Check for partial hypothesis broadcast after 15/30 frames
        msg = ws.receive_text()
        parsed = json.loads(msg)
        assert parsed["type"] == "partial_hypothesis"
        assert "text" in parsed


def test_websocket_speech_to_avatar():
    with client.websocket_connect("/ws/stream/TEST_ROOM_03") as ws:
        req = {
            "type": "speech_to_avatar",
            "text": "ሰላም እንደምን ነህ?"
        }
        ws.send_text(json.dumps(req))
        msg = ws.receive_text()
        parsed = json.loads(msg)
        assert parsed["type"] == "avatar_production_ready"
        assert parsed["total_frames"] >= 30
