/**
 * @file gesture_engine.c
 * @brief On-Device Decision-Tree Gesture Engine (§8.8)
 * Part of Tereguwami Companion Wearable Wristband Firmware
 */

#include "emg_acquisition.h"
#include <stdint.h>

typedef enum {
    GESTURE_NONE = 0,
    GESTURE_INDEX_TAP,
    GESTURE_FIST_CLENCH,
    GESTURE_WRIST_FLICK_UP,
    GESTURE_DOUBLE_TAP
} wearable_gesture_t;

typedef enum {
    CMD_IDLE = 0,
    CMD_START_LISTENING,
    CMD_REPEAT_TRANSLATION,
    CMD_REQUEST_CLARIFICATION,
    CMD_SWITCH_LANGUAGE
} wearable_command_t;

wearable_gesture_t classify_emg_gesture(const emg_feature_vector_t *features) {
    float ch0_mav = features->channels[0].mav;
    float ch1_mav = features->channels[1].mav;
    float ch2_mav = features->channels[2].mav;
    float ch3_mav = features->channels[3].mav;

    float total_energy = ch0_mav + ch1_mav + ch2_mav + ch3_mav;

    // Threshold baseline activation
    if (total_energy < 40.0f) {
        return GESTURE_NONE;
    }

    if (total_energy > 250.0f) {
        return GESTURE_FIST_CLENCH; // Clench = Request clarification
    }

    if (ch0_mav > 2.0f * ch2_mav && ch0_mav > 80.0f) {
        return GESTURE_INDEX_TAP; // Index tap = Start listening
    }

    if (ch1_mav > 2.0f * ch3_mav && ch1_mav > 90.0f) {
        return GESTURE_WRIST_FLICK_UP; // Flick = Switch language
    }

    return GESTURE_DOUBLE_TAP; // Repeat translation
}

wearable_command_t gesture_to_command(wearable_gesture_t gesture) {
    switch (gesture) {
        case GESTURE_INDEX_TAP:      return CMD_START_LISTENING;
        case GESTURE_FIST_CLENCH:    return CMD_REQUEST_CLARIFICATION;
        case GESTURE_WRIST_FLICK_UP: return CMD_SWITCH_LANGUAGE;
        case GESTURE_DOUBLE_TAP:     return CMD_REPEAT_TRANSLATION;
        default:                     return CMD_IDLE;
    }
}
