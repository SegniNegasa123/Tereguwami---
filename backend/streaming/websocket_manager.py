"""
Real-Time WebSocket Session Manager (§8, §11)
Part of Tereguwami (ተርጓሚ) Streaming Pipeline

Manages bidirectional, low-latency WebSocket connections for real-time continuous
signing sessions, landmark frame accumulation, streaming hypothesis broadcasts,
and avatar animation dispatch.
"""

from typing import Dict, List, Set, Any, Optional
import json
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from models.translation.gloss_free_transformer import continuous_translator
from models.translation.constrained_decoder import constrained_decoder
from models.production.progressive_transformer import avatar_production_engine


class StreamingSession:
    """Represents an active bidirectional signing session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.active_sockets: Set[WebSocket] = set()
        self.frame_buffer: List[np.ndarray] = []
        self.max_buffer_size = 75  # ~3 seconds at 25 FPS
        self.target_lang = "am"
        self.domain = "everyday_civic"


class WebSocketConnectionManager:
    """Manages active WebSocket sessions and broadcasts streaming translation updates."""

    def __init__(self):
        self.active_sessions: Dict[str, StreamingSession] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = StreamingSession(session_id)
        self.active_sessions[session_id].active_sockets.add(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.active_sockets.discard(websocket)
            if not session.active_sockets:
                del self.active_sessions[session_id]

    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            payload_str = json.dumps(message)
            dead_sockets = []
            for ws in session.active_sockets:
                try:
                    await ws.send_text(payload_str)
                except Exception:
                    dead_sockets.append(ws)
            for dead in dead_sockets:
                session.active_sockets.discard(dead)

    async def handle_incoming_frame(self, session_id: str, frame_data: Dict[str, Any]):
        """Process incoming keypoint frame from signer camera."""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]
        landmarks = frame_data.get("landmarks")
        if landmarks:
            arr = np.array(landmarks, dtype=np.float32).flatten()
            session.frame_buffer.append(arr)

            # Keep buffer bounded
            if len(session.frame_buffer) > session.max_buffer_size:
                session.frame_buffer.pop(0)

            # Trigger hypothesis computation every 15 frames (~0.5s)
            if len(session.frame_buffer) >= 20 and (len(session.frame_buffer) % 15 == 0):
                buffer_array = np.array(session.frame_buffer)
                translation = continuous_translator.translate(
                    keypoint_features=buffer_array,
                    target_lang=session.target_lang,
                    domain_hint=session.domain
                )
                constrained = constrained_decoder.decode_with_constraints(
                    candidate_text=translation["translated_text"],
                    confidence_score=translation["confidence_score"],
                    recognized_glosses=[translation.get("matched_template", "SIGN")],
                    domain=session.domain
                )

                # Broadcast partial hypothesis to all connected participants
                await self.broadcast_to_session(session_id, {
                    "type": "partial_hypothesis",
                    "text": constrained["final_text"],
                    "confidence": constrained["confidence_score"],
                    "requires_clarification": constrained["requires_clarification"],
                    "timestamp_ms": frame_data.get("timestamp_ms", 0)
                })

    async def handle_speech_to_avatar(self, session_id: str, text_message: str):
        """Process spoken or typed text from hearing participant and stream avatar animation."""
        avatar_result = avatar_production_engine.generate_avatar_stream(
            text_input=text_message,
            source_lang="am",
            signing_speed=1.0
        )
        await self.broadcast_to_session(session_id, {
            "type": "avatar_production_ready",
            "text": text_message,
            "total_frames": avatar_result["total_frames"],
            "fps": avatar_result["fps"],
            "duration_seconds": avatar_result["duration_seconds"],
            "frames": avatar_result["frames"]
        })


# Global WebSocket manager singleton
ws_manager = WebSocketConnectionManager()
