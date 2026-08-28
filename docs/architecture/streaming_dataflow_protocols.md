# Streaming Protocols & Dataflow Architecture (§8, §11)

Tereguwami guarantees sub-250ms end-to-end latency for live two-way conversations between Deaf signers and hearing non-signers.

---

## 1. Transport Stack
- **WebRTC**: Low-latency video frame transport for high-bandwidth camera capture.
- **WebSockets**: Lightweight bidirectional JSON/binary frame transmission on endpoint `/ws/stream/{session_id}`.

## 2. Frame Packet Specification
```json
{
  "type": "frame",
  "session_id": "ROOM_8921",
  "frame_index": 42,
  "timestamp_ms": 1724760000120,
  "landmarks": [
    [0.482, 0.312, -0.015],
    [0.514, 0.320, -0.012]
  ]
}
```

## 3. Streaming Hypothesis Specification
```json
{
  "type": "partial_hypothesis",
  "session_id": "ROOM_8921",
  "text": "ዶክተር ብርቱ የራስ ምታት አለኝ።",
  "confidence": 0.98,
  "is_final": false,
  "requires_clarification": false,
  "timestamp_ms": 1724760000850
}
```

## 4. Latency Breakdown Budget
| Pipeline Stage | Processing Target |
|---|---|
| Camera Frame Capture & MediaPipe Keypoints | $\le 30$ ms |
| Network WebSocket Ingestion | $\le 25$ ms |
| Spatio-Temporal Transformer Encoder | $\le 60$ ms |
| Multilingual Beam Search Decoding | $\le 55$ ms |
| Speech Synthesis / Avatar Dispatch | $\le 40$ ms |
| **Total End-to-End Latency** | **$\le 210$ ms (< 250 ms target)** |
