# Native Android Client

Native Android application engineered for on-device real-time ESL translation and edge inference.

## Technology Stack
- **Languages**: Kotlin, C++ (NDK for high-performance keypoint extraction)
- **ML Runtime**: ONNX Runtime Mobile / TensorFlow Lite (GPU & NPU hardware accelerated via NNAPI)
- **Camera Pipeline**: CameraX streaming at 30 FPS with hardware orientation handling
- **Audio Output**: Text-To-Speech (Amharic & Afaan Oromo vocalization engines)
- **Offline / Low-Connectivity**: Local quantization (INT8) enabling complete edge offline functionality in remote regions.
