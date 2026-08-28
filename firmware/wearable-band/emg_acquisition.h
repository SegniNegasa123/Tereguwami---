/**
 * @file emg_acquisition.h
 * @brief Embedded Analog Front-End (AFE) sEMG Acquisition & Filtering Header (§8.8)
 * Part of Tereguwami Companion Wearable Wristband Firmware
 */

#ifndef EMG_ACQUISITION_H
#define EMG_ACQUISITION_H

#include <stdint.h>
#include <stdbool.h>

#define EMG_NUM_CHANNELS       4
#define EMG_SAMPLE_RATE_HZ     1000
#define EMG_WINDOW_SIZE        128

typedef struct {
    float mav;   // Mean Absolute Value
    float wl;    // Waveform Length
    uint16_t zc; // Zero Crossings
    float rms;   // Root Mean Square
} emg_channel_features_t;

typedef struct {
    emg_channel_features_t channels[EMG_NUM_CHANNELS];
    uint32_t timestamp_ms;
} emg_feature_vector_t;

void emg_afe_init(void);
void emg_sample_tick(int16_t raw_adc[EMG_NUM_CHANNELS]);
float emg_apply_notch_50hz(float input, uint8_t channel);
float emg_apply_bandpass(float input, uint8_t channel);
bool emg_is_window_ready(void);
emg_feature_vector_t emg_extract_features(void);

#endif // EMG_ACQUISITION_H
