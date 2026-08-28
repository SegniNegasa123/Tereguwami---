# Layer 8.1 — Perception & Landmark Extraction

Wrappers for real-time skeletal and dense facial keypoint extraction targeting 25–30 FPS on commodity smartphone and web cameras.

## Components
- `holistic_extractor.py`: MediaPipe Holistic wrapper outputting 543 normalized coordinates.
- `face_mesh_highres.py`: High-resolution facial crop extraction specifically for eyebrow, aperture, and mouth dynamics.
- `normalization.py`: Signer-scale invariance normalization (torso width, shoulder alignment, wrist-relative centering).
