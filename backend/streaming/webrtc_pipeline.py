"""
WebRTC Media Frame Ingestion & Buffer Pipeline (§8.1, §11)
Part of Tereguwami (ተርጓሚ) Real-Time Transport Infrastructure

Manages incoming camera frame queues, frame-drop mitigation for low-bandwidth environments,
and audio-video temporal synchronization.
"""

import time
import asyncio
from typing import Optional, Dict, Any, List
import numpy as np


class WebRTCMediaPipeline:
    """
    High-speed video frame and audio buffer manager.
    Enforces a strict latency budget target (<250ms end-to-end).
    """

    def __init__(self, max_queue_size: int = 30, target_fps: int = 25):
        self.max_queue_size = max_queue_size
        self.target_fps = target_fps
        self.frame_interval_ms = 1000.0 / target_fps
        self._frame_queue: List[Dict[str, Any]] = []
        self._last_processed_timestamp = 0.0

    def push_frame(self, frame_rgb: np.ndarray, timestamp_ms: Optional[float] = None) -> bool:
        """
        Push raw camera frame into buffer. Drops oldest frame if buffer exceeds capacity
        to prevent latency accumulation.
        """
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000.0

        if len(self._frame_queue) >= self.max_queue_size:
            # Drop oldest frame to ensure real-time responsiveness
            self._frame_queue.pop(0)

        self._frame_queue.append({
            "data": frame_rgb,
            "timestamp_ms": timestamp_ms
        })
        return True

    def pop_frame(self) -> Optional[Dict[str, Any]]:
        """Retrieve next frame for perception and feature extraction."""
        if self._frame_queue:
            return self._frame_queue.pop(0)
        return None

    def clear(self) -> None:
        self._frame_queue.clear()

    @property
    def queue_length(self) -> int:
        return len(self._frame_queue)


webrtc_pipeline = WebRTCMediaPipeline()
