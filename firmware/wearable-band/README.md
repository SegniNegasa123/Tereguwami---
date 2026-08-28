# Layer 8.8 — Wearable Companion Band Firmware

Embedded C/C++ firmware running on ultra-low-power microcontrollers (e.g. Nordic nRF52840 / ESP32-S3 / STM32WB) for real-time surface EMG (sEMG) acquisition and silent gesture control.

## System Characteristics
- **Sensors**: Multi-channel differential surface electromyography (sEMG) analog front-end (AFE).
- **Communication**: Bluetooth Low Energy (BLE 5.2) stream transmitting feature vectors / raw micro-voltage readings to mobile/edge host.
- **Control Gestures**:
  - Double finger tap -> "Start / Stop Listening"
  - Wrist flick -> "Repeat Last Translation"
  - Micro-fist squeeze -> "Request Clarification"
  - Hand open hold -> "Switch Output Language"
- **Power Optimization**: Low-power standby with hardware interrupt-driven wake-up for multi-day battery autonomy.
