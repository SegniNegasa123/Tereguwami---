# The Nine Cooperating Layers: Comprehensive Technical Specification

Tereguwami is engineered as nine decoupled, benchmarkable layers (§8). Each layer defines formal interfaces, input/output tensors, latency budgets, and fallback mechanisms.

---

### Layer 1: Perception Layer (§8.1)
- **Primary Model**: MediaPipe Holistic (Pose + Left Hand + Right Hand + Face Mesh).
- **Input**: Raw RGB video stream (25–30 FPS, minimum 720p).
- **Output**: 543 normalized 3D keypoint coordinates (shape: `[T, 543, 3]`).
- **Latency Budget**: $\le 33$ ms per frame on commodity mobile CPUs.
- **Fail-Safe**: Dropped frames interpolated via cubic spline; occluded hands marked with confidence $0.0$.

### Layer 2: Recognition Layer (§8.2)
- **Primary Model**: Temporal Sequence Models (CNN-LSTM / BiLSTM / GRU).
- **Input**: Spatially normalized keypoint trajectory `[B, T, 1629]`.
- **Output**: Lexical sign classification probabilities over vocabulary `V`.
- **Benchmark Alignment**: Directly benchmarked against the 2025 Ethiopian academic baseline.

### Layer 3: Non-Manual Marker & Facial Semantics Layer (§8.3)
- **Primary Model**: Early-Fusion Multi-Stream Facial Feature Encoder.
- **Linguistic Markers**: Eyebrow elevation (AU1/2 - polar question), eyebrow furrow (AU4 - wh-question), head shake (negation), head nod (affirmation), mouth aperture/mouthing.
- **Fusion**: Concatenated with manual keypoint embeddings before temporal attention blocks.

### Layer 4: Translation / Language Layer (§8.4)
- **Primary Model**: Gloss-Free Sequence-to-Sequence Transformer with Afrocentric Multilingual Decoder (`AfriBERTa` / `Afro-XLMR`).
- **Target Languages**: Amharic (Ethiopic), Afaan Oromo (Latin), English.
- **Decoding Guardrail**: Constrained beam search rejecting ungrounded entities and enforcing confidence threshold $\tau \ge 0.70$.

### Layer 5: Production Layer — Reverse Channel (§8.5)
- **Primary Model**: Progressive Pose Transformer.
- **Input**: Text or phoneme stream from spoken hearing speech.
- **Output**: Continuous joint trajectory and 52 ARKit-compatible blendshape weights.
- **Rendering Target**: Real-time 3D WebGL / Three.js animated avatar.

### Layer 6: Personalization Layer (§8.6)
- **Primary Model**: Metric Learning (Siamese / Prototypical Embeddings) + On-Device Federated Averaging (Flower).
- **Functionality**: 1-to-5 shot enrollment of family or regional signs into an L2-normalized 128-dimensional metric space.
- **Privacy Guarantee**: Raw video never leaves local client storage.

### Layer 7: Silent-Speech / Neuromotor Output Layer (§8.7)
- **Primary Model**: Non-Invasive Jaw/Face sEMG Subvocalization Classifier (AlterEgo lineage).
- **Hardware**: 6 differential surface electrodes sampled at 1000 Hz.
- **Scope**: Closed-vocabulary commands and emergency phrases (TRL 2–3).

### Layer 8: Companion Wearable & Control Layer (§8.8)
- **Primary Model**: Wrist-worn surface-EMG micro-gesture classifier (Meta Reality Labs lineage).
- **Functionality**: Silent, hands-free control commands (start listening, repeat, clarify, change language).

### Layer 9: Adaptation and Feedback Layer (§8.9)
- **Functionality**: Out-of-distribution (OOD) unknown sign detection, confidence calibration, and interactive clarification loop.
