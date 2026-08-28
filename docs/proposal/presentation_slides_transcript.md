# TEREGUWAMI (ተርጓሚ) — 18 Presentation Slides Transcript & Speaker Notes

**Presentation Context**: EGATE — Gifted & Talented Program | Advanced AI/ML Research Track  
**Author**: Segni Seyoum Negasa, Addis Ababa, Ethiopia (August 2026)  

---

### Slide 1: Title & Identity
- **Header**: EGATE — Gifted & Talented Program | Advanced AI/ML Research Track
- **Title**: TEREGUWAMI (ተሬጉዋሚ / ተርጓሚ) • "the translator / interpreter"
- **Subtitle**: An adaptive multimodal AI system for continuous, two-way communication between Ethiopian Sign Language users and non-signers.

### Slide 2: Problem Statement — A Structural Communication Barrier
- **Key Insight**: Ethiopian Sign Language (ESL) is a complete natural language with its own grammar, distinct from spoken Amharic, Afaan Oromo, or English.
- **Settings Without an Interpreter**:
  1. Hospitals & clinics
  2. Courts & police stations
  3. Banks & service windows
  4. Schools & universities
  5. Government offices
  6. Family & civic life

### Slide 3: Research Gap — What No Prior System Has Attempted
1. Continuous translation (sentence-level, not isolated signs)
2. Non-manual grammar (facial markers modeled as first-class linguistic signal)
3. Public benchmark (signer-independent dataset & leaderboard released openly)
4. Reverse channel (text/speech generated into a signing avatar)
5. Personalization (few-shot adaptation to an individual's signing style)
6. Camera-free channel (non-invasive silent-speech path when signing isn't possible)

### Slide 4: Landscape Review — Where Existing Systems Fall Short
- **SignAll (USA)**: Glove + camera; specialized hardware; not continuous free-signing.
- **Sign Language Transformers (CVPR 2020)**: Gloss-based transformer; German only; requires expensive gloss annotations.
- **Hand Talk / Hugo (Brazil)**: Text-to-Libras avatar (clip-based); one-directional only; not generative.
- **2025 Ethiopian Skeleton Study (Scientific Reports)**: CNN-LSTM / BiLSTM / GRU on MediaPipe keypoints; **94% signer-dependent vs 73% signer-independent (21-point collapse)**.

### Slide 5: Design Philosophy — Five Non-Negotiable Principles
1. Continuous over isolated
2. Grammar is not just hands
3. Generalize to new signers by design
4. Fail safely and stay honest
5. Access doesn't require a camera

### Slide 6: System Architecture — Nine Cooperating Layers
- Layers 1–5: Core Translation Pipeline (Perception, Recognition, Non-Manual Markers, Translation, Production).
- Layers 6–9: Research Extensions (Personalization, Silent-Speech, Companion Wearable, Adaptation & Feedback).

### Slide 7: Perception & Recognition (Layers 1–2)
- **Model Selected**: MediaPipe Holistic + CNN-LSTM / BiLSTM / GRU ensemble.
- **Why**: Real-time on commodity mobile phones (25–30 FPS), matching prior benchmark tooling.

### Slide 8: Non-Manual Marker & Facial Semantics (Layer 3)
- **Model Selected**: Early-fusion multi-stream facial encoder.
- **Why**: Captures eyebrow elevation (questions), eyebrow furrow (wh-questions), head tilt/shake (negation). Early fusion prevents loss of facial grammar.

### Slide 9: Translation / Language Layer (Layer 4)
- **Model Selected**: Gloss-free Transformer with Afrocentric-pretrained multilingual decoder (`AfriBERTa` / `Afro-XLMR`).
- **Why**: ESL has minimal gloss data; Afrocentric decoders achieve 80–91% Macro-F1 compared to 68–82% for generic mBERT.

### Slide 10: Production Layer — Reverse Channel (Layer 5)
- **Model Selected**: Generative pose-sequence transformer (Progressive-Transformer lineage).
- **Why**: Generates genuinely novel, continuous pose sequences with facial blendshapes rather than stitching pre-recorded clips.

### Slide 11: Personalization Layer (Layer 6)
- **Model Selected**: Siamese / Prototypical few-shot networks + Federated Averaging.
- **Why**: Enrolls custom/family signs with 1–5 examples; on-device federated learning guarantees raw video never leaves the user's phone.

### Slide 12: Silent-Speech & Companion Wearable (Layers 7–8)
- **Model Selected**: Non-invasive EMG decoders (AlterEgo-derived jaw/face + Meta-style wrist sEMG).
- **Why**: Consumer-safe, no brain surgery; honest TRL 2–3 research track for low-light or privacy contexts.

### Slide 13: Model Selection Summary
- Comprehensive reference matrix of all 9 layer model choices, alternatives considered, and rationale.

### Slide 14: Dataset & Benchmark Plan — The Flagship Contribution
- 60–100 distinct Deaf signers, continuous natural signing, multi-layer annotation, signer-independent splits, and Deaf-led governance.

### Slide 15: Evaluation Plan & Target Metrics
- BLEU-4, signer-independent accuracy, non-manual ablation gain, latency (<250ms), few-shot accuracy, and human fluency ratings.

### Slide 16: Two-Way Conversation in Practice
- Detailed user journeys for Signer-to-Hearing, Hearing-to-Signer, and Camera-Free Silent-Speech modes.

### Slide 17: Phased Development & Technology Readiness Levels (TRL)
- Phased roadmap from Phase 0 (Foundations, TRL 3–4) to Phase 7 (Institutional Pilots, TRL 5–6).

### Slide 18: Deployment, Scale & Impact
- National broadcaster overlay, telecom relay service, court/hospital pilots, and Horn of Africa regional expansion.
