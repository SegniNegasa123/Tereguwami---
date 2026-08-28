# Benchmark Evaluation Suite

Automated evaluation scripts, scoring protocols, and baseline metrics for Ethiopian Sign Language recognition, translation, and production.

## Standard Evaluation Metrics (§15)
- **Translation Quality**: BLEU-4, ROUGE-L, METEOR, and chrF++ against held-out multi-reference translations in Amharic, Afaan Oromo, and English.
- **Signer Independence**: Accuracy, Precision, Recall, and F1 evaluated strictly on the **held-out signer-independent test split**.
- **Non-Manual Contribution**: Automated ablation scoring comparing hands-only versus multimodal (hands + facial mesh) model variants across question and negation subsets.
- **Latency**: 95th-percentile end-to-end response time (ms) on standard Android/iOS test devices.
- **Few-Shot Personalization**: Classification accuracy on newly enrolled signs as a function of $K$ support shots ($K \in \{1, 3, 5\}$).
- **Human Fluency & Naturalness**: Qualitative scoring rubric by native Deaf Ethiopian reviewers for avatar fidelity.
