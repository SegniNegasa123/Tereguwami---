# Tereguwami — ተርጓሚ
### An Adaptive Multimodal AI System for Two-Way Communication Between Ethiopian Sign Language Users and Non-Signers

[![License: Research & Open Source](https://img.shields.io/badge/License-MIT%20%2F%20Research-blue.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-success.svg)]()
[![Target: Ethiopian Sign Language](https://img.shields.io/badge/Language-ESL%20%2F%20ETHSL-orange.svg)]()

---

## 1. Overview

**Tereguwami** (ተርጓሚ, Amharic for *"the translator/interpreter"*) is the first open, benchmarked, bidirectional communication system designed for **Ethiopian Sign Language (ESL / ETHSL)**. It bridges communication barriers between Deaf/hard-of-hearing individuals and hearing non-signers across vital public and private domains: healthcare, education, legal proceedings, banking, public broadcasts, and everyday life.

Tereguwami unifies four core research threads into a single deployed accessibility platform:
1. **Continuous, Non-Manual-Marker-Aware Translation**: Sentence-level recognition mapping hands, body pose, and facial grammatical markers (eyebrow, mouth, head movements) into fluent Amharic, Afaan Oromo, and English.
2. **Generative Sign Production (Reverse Channel)**: Continuous avatar performance generated directly from text/speech using sequence transformers.
3. **Non-Invasive Neuromuscular "Silent Speech" Channel**: AlterEgo-derived sEMG decoding from facial/jaw subvocalizations for camera-free communication in low-light, occupied-hand, or privacy-sensitive contexts.
4. **Companion Wearable Control Interface**: Wrist-worn surface-EMG band for hands-free, silent command and control.

---

## 2. Nine-Layer System Architecture

```mermaid
graph TD
    subgraph Input ["Perception & Input Layers"]
        V[Video Feed] --> L1[8.1 Perception Layer<br/>MediaPipe Holistic / Keypoints]
        EMG1[Jaw/Face sEMG] --> L7[8.7 Silent-Speech Layer<br/>Subvocalization Decoder]
        EMG2[Wrist sEMG] --> L8[8.8 Companion Wearable<br/>Silent Control Band]
    end

    subgraph Core ["Recognition & Translation Backbone"]
        L1 --> L2[8.2 Recognition Layer<br/>Temporal Sequence Model]
        L1 --> L3[8.3 Non-Manual Marker Layer<br/>Facial Grammar & Semantics]
        L2 --> L4[8.4 Translation / Language Layer<br/>Gloss-Free Transformer + Multilingual Decoder]
        L3 --> L4
        L7 --> L4
    end

    subgraph Output ["Production & Delivery"]
        L4 --> T_OUT[Amharic / Afaan Oromo / English Text & Audio]
        T_IN[Spoken / Written Text] --> L5[8.5 Production Layer<br/>Generative Avatar Transformer]
        L5 --> AVATAR[3D Animated ESL Avatar]
    end

    subgraph Adaptation ["Personalization & Continuous Learning"]
        L6[8.6 Personalization Layer<br/>Few-Shot Siamese & Federated On-Device]
        L9[8.9 Adaptation & Feedback Layer<br/>Confidence Awareness & Unknown Sign Detection]
        L4 <--> L6
        L4 <--> L9
    end
```

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| **Computer Vision / Perception** | MediaPipe Holistic, OpenCV |
| **Model Development** | PyTorch, HuggingFace Transformers |
| **Personalization / Metric Learning** | Siamese / Prototypical Networks, Flower (Federated Learning) |
| **Backend Services** | Python, FastAPI |
| **Real-Time Transport** | WebRTC, WebSockets |
| **Data Storage** | PostgreSQL (metadata, profiles), Object Storage (video, model artifacts) |
| **Mobile & Edge** | Android & iOS clients, ONNX Runtime Mobile, TensorFlow Lite |
| **Wearable Firmware** | Embedded C/C++ on low-power MCU with analog front-end (AFE) sEMG |
| **Avatar Rendering** | Real-time 3D rendering (WebGL / Unity) driven by generated pose sequences |
| **Evaluation / Leaderboard** | Containerized submission & scoring engine (BLEU-4, signer-independent accuracy) |
| **Infrastructure** | Cloud GPU training cluster + edge-optimized deployment |

---

## 4. Repository Structure

```
tereguwami/
├── docs/                        # Architecture specs, proposals, dataset cards, governance docs
├── data/                        # Datasets, processed keypoints, annotations, and train/val/test splits
├── research/                    # Research notebooks, experiment tracking configs, publication papers
├── models/                      # Deep learning model architectures (perception to silent-speech)
├── backend/                     # FastAPI backend services, streaming WebRTC pipelines, auth, DB
├── mobile/                      # Android, iOS applications and cross-platform shared components
├── firmware/                    # Embedded C/C++ code for the sEMG wristband and AFE hardware
├── avatar/                      # 3D rigging definitions and WebGL/Unity avatar rendering engines
├── evaluation/                  # Benchmark evaluation suites and competitive leaderboard server
├── infra/                       # Edge deployment packaging and cloud GPU orchestration manifests
└── governance/                  # Deaf advisory board charters, consent forms, data-use agreements
```

---

## 5. Development Roadmap & Technology Readiness Levels (TRL)

- **Phase 0 — Foundations (TRL 3–4)**: Perception pipeline, ~10–15 signer pilot data collection, Deaf advisory board establishment.
- **Phase 1 — Isolated Recognition Baseline (TRL 4)**: Small-vocabulary isolated ESL recognizer, reproducing & exceeding 2025 Ethiopian baselines.
- **Phase 2 — Continuous Translation (TRL 4–5)**: Gloss-free transformer with cross-lingual pretraining; sentence-level translation; BLEU-4 evaluation.
- **Phase 3 — Non-Manual Markers (TRL 4)**: High-resolution facial semantics fused as first-class input; ablation validations.
- **Phase 4 — Personalization (TRL 4)**: Few-shot personal gesture enrollment; federated on-device adaptation.
- **Phase 5 — Reverse Channel (TRL 3–4)**: Generative avatar pose sequence generation; bidirectional conversation UI.
- **Phase 6 — Silent-Speech Prototype (TRL 2–3)**: AlterEgo-derived jaw/face sEMG wearable; closed-vocabulary calibration.
- **Phase 7 — Institutional Pilots (TRL 5–6)**: Broadcaster, telecom relay, healthcare/court pilots; public benchmark release.

---

## 6. Community & Ethical Governance

Tereguwami is governed in active partnership with the Ethiopian Deaf community. All data collection follows strict consent and withdrawal protocols under the oversight of a Deaf-led Advisory Board. Raw video data is strictly access-controlled and never committed to version control.
