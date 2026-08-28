# Layer 8.4 — Translation / Language Layer

Sequence-to-sequence transformer architecture enabling gloss-free, continuous Ethiopian Sign Language translation into Amharic, Afaan Oromo, and English.

## Key Features
- **Gloss-Free Architecture**: Operates directly on continuous sign keypoint embeddings without requiring manual gloss-level intermediate representations.
- **Cross-Lingual Pretraining**: Spatial-temporal encoder pretrained on large international corpora (PHOENIX-2014T, How2Sign, CSL-Daily) and fine-tuned on Ethiopian Sign Language to maximize signer independence.
- **Multilingual Decoder**: Shared autoregressive transformer decoder emitting Amharic (Ethiopic), Afaan Oromo (Latin), or English text tokens.
- **Constrained Decoding**: Output regularized against hallucination to preserve recognized sign semantic confidence.
