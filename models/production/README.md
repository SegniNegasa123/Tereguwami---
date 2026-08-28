# Layer 8.5 — Production Layer (Reverse Channel: Text/Speech to Sign)

Generative sign language production model synthesizing continuous 3D skeletal animations and facial blendshapes from natural language inputs.

## Methodology
- **Progressive Transformers**: Generates continuous joint trajectories and facial blendshape weights directly from text/speech embeddings, rather than concatenating pre-recorded video clips.
- **Non-Manual Synthesis**: Accurately reproduces grammatical facial movements (eyebrow lift/furrow for questions, head nods/shakes for polarity) alongside manual gestures.
- **Variable Signing Speed**: Speed modulation to support both fluent Deaf signers and beginners.
