# Companion NLP Benchmark Study: Afrocentric vs Multilingual Models for Amharic Text Decoding (§8.4)

**Project Lead**: Segni Seyoum Negasa  
**Program**: EGATE (School of Gifted and Talented) Advanced AI/ML Track  
**Role in Tereguwami**: Informs the Language & Decoder Architecture of Layer 8.4  

---

## 1. Executive Summary
In designing the text decoding head for Tereguwami's gloss-free sign language translation transformer, a critical architectural decision was whether to initialize the decoder with a generic multilingual model (e.g. `mBERT`, `XLM-RoBERTa`) or a specialized Afrocentric model (e.g. `AfriBERTa`, `Afro-XLMR`).

This companion empirical investigation compared classical statistical feature engineering against modern pre-trained contextual representations on standardized Amharic benchmarks (`AfriSenti-Amharic`, ~9,400 annotated posts + 1,500 domain samples):

| Model Family | Representative Model | Amharic Macro-F1 | Ge'ez Token Fertility | CPU Latency | Architectural Fit for Layer 8.4 |
|---|---|---|---|---|---|
| **Classical ML** | TF-IDF + Logistic Regression / SVC | 71.2% | High OOV (>18%) | < 2 ms | Fast baseline, poor syntactic nuance |
| **Generic Multilingual** | mBERT / XLM-R | 68.0% – 82.0% | 3.8 sub-words/word (fragmented) | 30–60 ms | High cross-lingual negative interference on Ge'ez |
| **Afrocentric Transformer** | AfriBERTa / Afro-XLMR / AmRoBERTa | **88.0% – 91.2%** | **1.3 sub-words/word (optimal)** | 15–35 ms | **Selected: Superior morphology capture & higher translation fluency** |

## 2. Key Findings Applied to Tereguwami
1. **WordPiece & BPE Tokenization on Ge'ez**: Generic multilingual models fragment Amharic words into arbitrary byte fragments due to tiny representation in their training corpora. Afrocentric models allocate dedicated sub-word vocabulary for Ethiopian languages, reducing out-of-vocabulary degradation.
2. **Faithful Semantic Grounding**: The higher semantic coherence of Afrocentric representations prevents the translation decoder from hallucinating erroneous sentence completions during constrained beam decoding.
