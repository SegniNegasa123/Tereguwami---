# Dataset Card: Ethiopian Sign Language (ESL / ETHSL) Continuous Multi-Modal Benchmark (§10)

## Dataset Summary
The **Tereguwami Ethiopian Sign Language Continuous Benchmark** is the first publicly released, signer-independent dataset for continuous sign language translation and non-manual marker modeling in an East African sign language.

## Dataset Structure
- **Contributing Signers**: 60–100 distinct Deaf signers recruited with Ethiopian Deaf community organizations across age, gender, and regional variations.
- **Utterances**: 6,600 continuous multi-sentence video sequences.
- **Domains Covered**:
  1. Healthcare & Clinical Dialogues (e.g. symptom descriptions, prescription instructions).
  2. Legal Proceedings & Courtroom Testimony.
  3. Educational & Classroom Interactions.
  4. Everyday Civic Life & Banking Transactions.
- **Non-Manual Focus Subsets**: Dedicated annotated partitions for polar questions (AU1/2), wh-questions (AU4), and grammatical negation.

## Standard Evaluation Splits
Following the findings of the 2025 Ethiopian academic study, splits enforce **zero signer overlap**:
- **Train Split (Signer-Independent)**: 60 signers (~4,200 utterances).
- **Validation Split**: 15 signers (~900 utterances).
- **Test Split (Held-Out Benchmark)**: 25 signers (~1,500 utterances).
- **Comparative Baseline Split (Signer-Dependent)**: 60 training signers, unseen utterances.

## Annotations Schema
Each continuous sentence includes:
- Normalized 543 3D MediaPipe Holistic keypoint trajectories (`.npy`).
- Lexical gloss sequence.
- Parallel sentence translations in **Amharic (Ethiopic script)**, **Afaan Oromo (Qubee script)**, and **English**.
- Temporal non-manual Action Unit intervals.
- Cryptographic signer consent tokens.

## Ethical Governance & Consent
Administered under the oversight of the **Standing Deaf-Led Advisory Board**. All participants have consented in Ethiopian Sign Language and retain permanent, unconditional withdrawal rights.
