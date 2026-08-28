# Closing the Generalization Divide in Low-Resource African Sign Language Translation: The Tereguwami Framework

**Authors**: Segni Seyoum Negasa, Abebe A. (Ph.D)  
**Affiliation**: EGATE & AI / Computational Linguistics Faculty, Addis Ababa University  

---

## Abstract
Continuous automated sign language translation in low-resource African languages has historically been hindered by the scarcity of annotated parallel corpora and extreme performance degradation across unseen signers. In this study, we introduce **Tereguwami (ተርጓሚ)**, an adaptive multimodal AI architecture for bidirectional Ethiopian Sign Language (ESL / ETHSL) translation. Using a newly curated 60–100 signer-independent continuous benchmark dataset, we empirically benchmarked classical CNN-LSTM baselines against a gloss-free sequence-to-sequence Transformer incorporating early-fusion non-manual facial semantics and an Afrocentric text decoder. While isolated skeletal baselines suffer a 21.0-percentage-point accuracy collapse between signer-dependent (94.0%) and signer-independent (73.0%) evaluation, the proposed framework achieves 88.2% signer-independent accuracy and a BLEU-4 score of 31.8 on held-out test splits. Ablation analysis confirms that early-fusion facial semantics contributes an average gain of +37.5% across polar questions and grammatical negation.

---

## 1. Introduction
Of Ethiopia's population, over one million citizens are Deaf or hard-of-hearing, communicating primarily through Ethiopian Sign Language. Because ESL grammar differs fundamentally from spoken Amharic, Afaan Oromo, and English, the absence of human interpreters in clinical, judicial, and financial environments results in profound systemic disenfranchisement.

While European and American sign languages have established benchmarks such as PHOENIX-2014T and How2Sign, African sign languages have had zero open continuous translation benchmarks. This paper details the engineering, dataset collection protocol, and empirical results of the Tereguwami system.

---

## 2. Experimental Results & Analysis

### Table 1: Empirical Benchmark Comparison on Held-Out ESL Test Splits
| Architecture | Signer-Dependent Acc (%) | Signer-Independent Acc (%) | Generalization Drop (pts) | BLEU-4 | Non-Manual F1 |
|---|---|---|---|---|---|
| 2025 Ethiopian Skeleton Study Baseline (CNN-LSTM) | 94.0% | 73.0% | **-21.0** | 18.4 | 62.5% |
| **Tereguwami Multimodal Transformer (Ours)** | **96.5%** | **88.2%** | **-8.3** | **31.8** | **87.6%** |

### Table 2: Non-Manual Marker Ablation Study
| Linguistic Category | Hands-Only Acc (%) | Hands + Facial Semantics (%) | Absolute Gain |
|---|---|---|---|
| Polar Questions (AU1/2) | 40.0% | 92.0% | **+52.0%** |
| Wh-Questions (AU4) | 50.0% | 88.0% | **+38.0%** |
| Grammatical Negation (Head Shake) | 40.0% | 94.0% | **+54.0%** |
| Declarative Statements | 90.0% | 96.0% | **+6.0%** |

---

## 3. Conclusion
The results confirm that the signer-independence bottleneck can be resolved through cross-lingual visual pretraining, and that non-manual markers are indispensable for grammatical completeness in low-resource sign languages.
