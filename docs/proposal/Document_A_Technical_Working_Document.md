# TEREGUWAMI (ተርጓሚ)
### An Adaptive Multimodal AI System for Two-Way Communication Between Ethiopian Sign Language Users and Non-Signers
**TECHNICAL & RESEARCH WORKING DOCUMENT (v1.0 — August 2026)**  
*Internal Engineering, Research & Design Reference — Not for External Submission*  
**Prepared by**: Segni Negasa, Addis Ababa, Ethiopia  

---

## Table of Contents
1. Executive Summary
2. Problem Statement
3. Related Work — Global Landscape
4. Related Work — Ethiopian Landscape
5. Research Gap and Contribution Statement
6. Project Identity and Naming
7. System Overview and Design Philosophy
8. Detailed System Architecture (Nine Layers)
9. Complete Feature Catalogue
10. Dataset and Benchmark Plan — the Flagship Research Contribution
11. Technology Stack
12. Repository and Folder Design
13. Engineering Design Process (EDP)
14. Development Roadmap and Technology Readiness Levels (TRL)
15. Evaluation Plan and Metrics
16. Ethics, Privacy, and Community Governance
17. Risk Register
18. Deployment and Government Partnership Strategy
19. Business Model and Scale Pathway
20. Team and Resource Requirements
21. Budget Framework
22. Expected Impact and Broader Significance
23. References

---

## 1. Executive Summary
Tereguwami (ተርጓሚ, Amharic for *"the translator/interpreter"*) is an ambitious research and engineering program to build the first open, benchmarked, bidirectional communication system for Ethiopian Sign Language (ESL / ETHSL). The program combines four research threads that have never previously been combined into a single deployed system:
1. Continuous, non-manual-marker-aware sign language recognition and translation;
2. Generative sign production through an animated 3D avatar;
3. A non-invasive neuromuscular "silent speech" channel that lets a deaf or non-speaking user produce spoken output without vocalizing;
4. A companion wearable input device for hands-free control.

The unifying goal is a single system that lets a deaf person and a hearing person who does not know sign language hold a natural, real-time, two-way conversation, without either party needing special training. Unlike the initial project concept it builds on, Tereguwami is scoped explicitly around a defensible research contribution — a released, signer-independent Ethiopian Sign Language benchmark dataset and translation model — and an explicit government/institutional deployment pathway.

---

## 2. Problem Statement
Deaf and hard-of-hearing Ethiopians communicate primarily through Ethiopian Sign Language, a full natural language with its own grammar, distinct from spoken Amharic, Afaan Oromo, or English, and carried not only through hand shape and movement but through facial expression, eyebrow position, mouth patterns, and head movement — the non-manual markers that encode grammatical features such as questions, negation, and topic marking. The overwhelming majority of hearing Ethiopians do not know ESL. This asymmetry produces a structural communication barrier in hospitals and clinics, courts and police stations, banks, schools, government service windows, and everyday family and civic life.

Ethiopia’s disability-rights framework — as a signatory to the UN Convention on the Rights of Persons with Disabilities (UNCRPD) — formally commits the state to accessible communication, yet no continuous, bidirectional, deployed translation system for ESL exists today.

The problem Tereguwami addresses, precisely stated, is: build a system that:
1. Recognizes continuous, natural ESL signing — not isolated signs performed one at a time — including non-manual grammatical markers;
2. Translates it into fluent Amharic, Afaan Oromo, and English text and speech;
3. Translates the reverse direction, from spoken or written language into a generated ESL avatar performance;
4. Personalizes to an individual signer's style, including family- or community-specific signs, without retraining from scratch;
5. Offers a non-manual, non-camera-dependent communication channel for situations where signing is not possible or a camera is not available.

---

## 3. Related Work — Global Landscape

| System / Research Line | Origin | What it Does | Key Limitation Tereguwami Resolves |
|---|---|---|---|
| **SignAll** | SignAll Inc. (USA) | Camera- and glove-based ASL-to-English translation | Depends on specialized gloves/hardware; not unconstrained continuous signing; no non-manual marker modeling. |
| **Sign Language Transformers** | Camgöz et al., CVPR 2020 | Joint end-to-end sign recognition and translation using Transformer architecture on PHOENIX-2014T | Requires expensive, linguist-produced gloss annotation; German Sign Language only; no East African equivalent. |
| **Gloss-free Transformers** | Academic groups, 2022–2025 | Learns sign-to-text alignment directly from paired video and text | Directly adopted by Tereguwami to bypass gloss annotation bottlenecks. |
| **Hand Talk (Hugo)** | Brazil (Acquired Jan 2025) | Text/speech-to-Libras animated avatar translation | One-directional only; clip-based stitching rather than continuous generative production. |
| **Progressive Transformers** | Saunders et al. (Academic) | Continuous sign pose sequence generation directly from text | Demonstrated on high-resource sign languages; first applied to East African sign language by Tereguwami. |
| **AlterEgo** | MIT Media Lab (Kapur et al., 2018) | Non-invasive jaw/face sEMG neuromuscular subvocalization decoder | Proven ~92% word accuracy on closed vocabulary; adapted for Tereguwami's camera-free alternative channel. |
| **Wrist sEMG Band** | Meta Reality Labs (Nature, 2025) | Reads electrical motor neuron signals to decode finger/hand gestures | Validates non-invasive neuromuscular wristbands for silent device control. |

---

## 4. Related Work — Ethiopian Landscape

Ethiopian Sign Language recognition has a 15-year academic history at Addis Ababa University:
- **Admasu & Raimond (2010)**: Early ANN recognition of isolated signs; small dataset, no continuous signing.
- **Tesfaye (2010)**: Machine translation approach for Amharic text to ESL; rule-based, non-generative.
- **Gimbi (2014) & Tamiru (2018)**: Isolated signs and alphabet fingerspelling recognition.
- **Abeje, Salau, Mengistu, Tamiru (2022)**: Deep CNN for ETHSL alphabet, reporting 98.5% training / 98.3% test accuracy on alphabet letters.
- **Skeleton-based EthSL Recognition (Scientific Reports, Nature Portfolio, 2025)**: 5,600 annotated videos comparing CNN-LSTM, BiLSTM, and GRU on MediaPipe keypoints. Reported **94% accuracy in signer-dependent testing, but collapsed to 73% in signer-independent testing**.

### 4.1 What No Prior Ethiopian Work Has Attempted
1. Continuous, sentence-level sign language translation.
2. Non-manual marker (facial grammar) recognition as part of the model.
3. A publicly released, signer-independent benchmark dataset and leaderboard.
4. A reverse channel — generating ESL avatar output from text or speech.
5. Personalized few-shot adaptation to an individual's signing style.
6. Any wearable, silent-speech, or neuromotor component.
7. Any named institutional deployment pathway.

---

## 5. Research Gap and Contribution Statement
**Core Thesis**: No publicly available system today provides continuous, non-manual-marker-aware, bidirectional translation for any East African sign language, personalizes to individual signers through few-shot learning, or extends communication access beyond the camera through a non-invasive silent-speech channel — Tereguwami is designed to be the first to do all four together, evaluated against a benchmark it will also be the first to publicly release for Ethiopian Sign Language.

### Primary Research Contributions
- **C1**: The first open, signer-independent, continuous Ethiopian Sign Language video-text benchmark dataset.
- **C2**: A gloss-free transformer translation model incorporating non-manual markers as first-class input, cross-lingually pretrained.
- **C3**: A generative (non-clip-based) sign production model driving a 3D animated avatar with facial grammar.
- **C4**: An honestly-scoped non-invasive neuromuscular silent-speech channel adapted from AlterEgo.
- **C5**: A federated, privacy-preserving personalization method for few-shot sign enrollment without raw video leaving the device.

---

## 6. Project Identity and Naming
- **Primary Name**: **Tereguwami (ተርጓሚ)** — Amharic for *"the translator/interpreter"*.
- **Branding Lockup**: Bilingual typography (ተርጓሚ above, *Tereguwami* below).
- **International Descriptor**: *"Tereguwami — the Ethiopian Sign Language AI Bridge"*.

---

## 7. Five Non-Negotiable Design Principles
1. **Continuous over isolated**: Signing is treated as a continuous temporal stream translated at sentence level.
2. **Grammar is not just hands**: Facial expression, eyebrows, mouth shape, and head movements are modeled as first-class linguistic signals.
3. **Generalize to new signers by design**: The 21-point signer-independence drop is treated as the central problem to solve.
4. **Fail safely and stay honest**: Exposes uncertainty, flags unknown signs, and avoids unsubstantiated "mind-reading" claims.
5. **Access does not require a camera**: A neuromuscular silent-speech channel serves low-light, occupied-hand, or privacy-sensitive contexts.

---

## 8. Detailed System Architecture (The Nine Layers)
- **8.1 Perception Layer**: MediaPipe Holistic extraction of 543 3D landmarks (pose, hands, face mesh) at 25-30 FPS.
- **8.2 Recognition Layer**: CNN-LSTM / BiLSTM / GRU baseline reproducing the 2025 Ethiopian benchmark.
- **8.3 Non-Manual Marker Layer**: Early-fusion facial encoder tracking eyebrows, eye aperture, mouth shape, and head pose.
- **8.4 Translation / Language Layer**: Gloss-free transformer with Afrocentric multilingual decoder (Amharic, Afaan Oromo, English) and safety-constrained decoding.
- **8.5 Production Layer (Reverse Channel)**: Progressive Transformer generating continuous pose and blendshape sequences for a 3D avatar.
- **8.6 Personalization Layer**: Few-shot Siamese metric learning and on-device Flower federated adaptation.
- **8.7 Silent-Speech / Neuromotor Layer**: Non-invasive jaw/face sEMG decoding of subvocalization into speech tokens.
- **8.8 Companion Wearable & Control Layer**: Wrist sEMG band for hands-free control commands (start listening, repeat, clarify).
- **8.9 Adaptation & Feedback Layer**: Out-of-distribution unknown-sign detection and user correction integration.

---

## 9–23. Roadmap, Evaluation, Governance & Strategic Outlook
- **Roadmap**: TRL 3 to TRL 6 phased rollout from perception baselines to institutional pilots.
- **Evaluation**: BLEU-4, signer-independent accuracy, non-manual ablation gain, and human fluency ratings.
- **Governance**: Standing Deaf-Led Advisory Board with binding veto power and enforceable participant withdrawal rights.
- **Deployments**: National broadcaster avatar overlay, Ethio Telecom relay service, court/hospital UNCRPD pilots.
