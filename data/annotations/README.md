# Multilingual Annotations & Linguistic Tags

This directory stores linguistic annotations aligned to video and keypoint sequences.

## Annotation Schema (PHOENIX-2014T & How2Sign Aligned)
Each annotation entry contains:
- `video_id`: Unique recording identifier
- `signer_id`: Anonymized signer ID
- `domain`: `healthcare` | `legal` | `education` | `civic` | `general`
- `sentence_translations`:
  - `amharic`: Ethiopic script ground-truth sentence
  - `afaan_oromo`: Latin/Qubee script sentence
  - `english`: English translation
- `gloss_sequence`: (Optional / where available) Tokenized ESL glosses
- `non_manual_markers`:
  - `eyebrows`: `neutral` | `raised` (yes/no question) | `furrowed` (wh-question)
  - `head`: `neutral` | `nod` (affirmation) | `shake` (negation) | `tilt` (topic)
  - `mouth`: Mouthing / mouth gesture descriptors
