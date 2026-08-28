# System Architecture Documentation

Detailed architectural specifications, data flow diagrams, and layer contracts for the nine cooperating layers of Tereguwami.

## The Nine Layers

### 1. Perception Layer (§8.1)
- **Role**: High-frame-rate (25–30 FPS) extraction of structured keypoints (hands, pose, facial mesh).
- **Technology**: MediaPipe Holistic pipeline.
- **Output**: 543 normalized 3D keypoints per frame.

### 2. Recognition Layer (§8.2)
- **Role**: Baseline sign unit classification and temporal feature encoding.
- **Technology**: CNN-LSTM / BiLSTM / GRU matching 2025 Ethiopian academic baselines.

### 3. Non-Manual Marker and Facial Semantics Layer (§8.3)
- **Role**: Tracking eyebrow raise/furrow, eye aperture, mouth gestures/mouthing, and head pose/motion.
- **Linguistic Function**: Grammatical questions (yes/no, wh-), negation, and topicalization in ESL.

### 4. Translation / Language Layer (§8.4)
- **Role**: End-to-end gloss-free continuous sequence-to-sequence translation.
- **Technology**: Cross-lingually pretrained transformer encoder with multilingual decoder (Amharic, Afaan Oromo, English).
- **Safety Constraint**: Constrained decoding that preserves recognized sign confidence without hallucinating content.

### 5. Production Layer / Reverse Channel (§8.5)
- **Role**: Generative sign production from text/speech into continuous 3D skeletal and blendshape sequences.
- **Technology**: Progressive Transformer generative pipeline driving avatar performance.

### 6. Personalization Layer (§8.6)
- **Role**: Metric learning (Siamese / prototypical networks) for few-shot sign enrollment + on-device federated learning (Flower) for privacy preservation.

### 7. Silent-Speech / Neuromotor Output Layer (§8.7)
- **Role**: Non-invasive jaw/face sEMG decoding of subvocalization signals into spoken audio.
- **Technology**: AlterEgo-derived architecture, closed-vocabulary personal calibration.

### 8. Companion Wearable and Control Layer (§8.8)
- **Role**: Wrist-worn sEMG band decoding intentional micro-gestures for hands-free UI control.

### 9. Adaptation and Feedback Layer (§8.9)
- **Role**: Out-of-distribution (OOD) unknown-sign detection, confidence calibration, and user correction loops.
