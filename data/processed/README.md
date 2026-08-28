# Processed Keypoint and Feature Data

This directory stores extracted skeletal and facial landmark time-series derived from raw sign videos.

## Feature Format
- **Landmarks**: 543 normalized coordinates $(x, y, z)$ per frame generated via MediaPipe Holistic.
  - Pose: 33 landmarks
  - Left Hand: 21 landmarks
  - Right Hand: 21 landmarks
  - Face Mesh: 468 landmarks (with dedicated high-resolution sub-regions for eyebrows, eyes, and mouth contours).
- **Storage formats**: HDF5 (`.h5`), compressed NumPy (`.npz`), or Parquet files.
