/**
 * @file emg_acquisition.c
 * @brief Embedded Analog Front-End (AFE) sEMG Acquisition & Filtering Implementation
 * Part of Tereguwami Companion Wearable Wristband Firmware (§8.8, §11)
 */

#include "emg_acquisition.h"
#include <math.h>
#include <string.h>

static float s_window_buffer[EMG_NUM_CHANNELS][EMG_WINDOW_SIZE];
static uint16_t s_buffer_idx = 0;
static bool s_window_full = false;

// 2nd-Order IIR Notch Filter Coefficients for 50Hz @ 1000Hz Sample Rate (Q=30)
static const float NOTCH_B0 =  0.9525f;
static const float NOTCH_B1 = -1.8118f;
static const float NOTCH_B2 =  0.9525f;
static const float NOTCH_A1 = -1.8118f;
static const float NOTCH_A2 =  0.9050f;

static float s_notch_x1[EMG_NUM_CHANNELS] = {0};
static float s_notch_x2[EMG_NUM_CHANNELS] = {0};
static float s_notch_y1[EMG_NUM_CHANNELS] = {0};
static float s_notch_y2[EMG_NUM_CHANNELS] = {0};

void emg_afe_init(void) {
    memset(s_window_buffer, 0, sizeof(s_window_buffer));
    s_buffer_idx = 0;
    s_window_full = false;
}

float emg_apply_notch_50hz(float input, uint8_t ch) {
    float output = NOTCH_B0 * input + NOTCH_B1 * s_notch_x1[ch] + NOTCH_B2 * s_notch_x2[ch]
                   - NOTCH_A1 * s_notch_y1[ch] - NOTCH_A2 * s_notch_y2[ch];
    s_notch_x2[ch] = s_notch_x1[ch];
    s_notch_x1[ch] = input;
    s_notch_y2[ch] = s_notch_y1[ch];
    s_notch_y1[ch] = output;
    return output;
}

float emg_apply_bandpass(float input, uint8_t ch) {
    // 1st-order highpass DC blocking filter at ~20 Hz
    static float s_prev_in[EMG_NUM_CHANNELS] = {0};
    static float s_prev_out[EMG_NUM_CHANNELS] = {0};
    const float alpha = 0.88f;
    float hp_out = alpha * (s_prev_out[ch] + input - s_prev_in[ch]);
    s_prev_in[ch] = input;
    s_prev_out[ch] = hp_out;
    return hp_out;
}

void emg_sample_tick(int16_t raw_adc[EMG_NUM_CHANNELS]) {
    for (uint8_t ch = 0; ch < EMG_NUM_CHANNELS; ch++) {
        // Convert ADC counts (12-bit) to millivolts
        float voltage_mv = ((float)raw_adc[ch] / 4096.0f) * 3300.0f;
        float notch_filtered = emg_apply_notch_50hz(voltage_mv, ch);
        float bandpass_filtered = emg_apply_bandpass(notch_filtered, ch);
        s_window_buffer[ch][s_buffer_idx] = bandpass_filtered;
    }

    s_buffer_idx++;
    if (s_buffer_idx >= EMG_WINDOW_SIZE) {
        s_buffer_idx = 0;
        s_window_full = true;
    }
}

bool emg_is_window_ready(void) {
    return s_window_full;
}

emg_feature_vector_t emg_extract_features(void) {
    emg_feature_vector_t feat;
    feat.timestamp_ms = 0;

    for (uint8_t ch = 0; ch < EMG_NUM_CHANNELS; ch++) {
        float sum_abs = 0.0f;
        float sum_sq = 0.0f;
        float wl = 0.0f;
        uint16_t zc = 0;

        for (uint16_t i = 0; i < EMG_WINDOW_SIZE; i++) {
            float val = s_window_buffer[ch][i];
            sum_abs += fabsf(val);
            sum_sq += val * val;

            if (i > 0) {
                float prev = s_window_buffer[ch][i - 1];
                wl += fabsf(val - prev);
                if ((val * prev < 0.0f) && (fabsf(val - prev) > 5.0f)) {
                    zc++;
                }
            }
        }

        feat.channels[ch].mav = sum_abs / (float)EMG_WINDOW_SIZE;
        feat.channels[ch].rms = sqrtf(sum_sq / (float)EMG_WINDOW_SIZE);
        feat.channels[ch].wl = wl;
        feat.channels[ch].zc = zc;
    }

    s_window_full = false;
    return feat;
}
