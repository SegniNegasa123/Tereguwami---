# Layer 8.2 — Recognition Layer

Implements baseline temporal sequence models matching the 2025 Ethiopian benchmark for direct reproducibility and baseline benchmarking.

## Architectures
- `cnn_lstm.py`: 1D/2D CNN feature encoder + LSTM temporal classifier.
- `bilstm.py`: Bidirectional LSTM with temporal attention over keypoint tracks.
- `gru.py`: Gated Recurrent Unit network for low-compute mobile inference baselines.
