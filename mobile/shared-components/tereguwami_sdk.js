/**
 * Tereguwami (ተርጓሚ) Client SDK
 * Production JavaScript/TypeScript SDK for Two-Way Multimodal ESL Communication
 * (§8, §11, §13)
 */

class TereguwamiSDK {
    constructor(baseURL = "http://127.0.0.1:8000") {
        this.baseURL = baseURL.replace(/\/$/, "");
        this.wsBaseURL = this.baseURL.replace(/^http/, "ws");
    }

    /**
     * Check system health and active layers.
     */
    async getHealth() {
        const resp = await fetch(`${this.baseURL}/api/v1/health`);
        if (!resp.ok) throw new Error(`Health check failed: ${resp.statusText}`);
        return await resp.json();
    }

    /**
     * Translate continuous 543 3D keypoint sequence into target text.
     * @param {Array} keypoints - Array of frames containing 543 [x, y, z] points.
     * @param {string} targetLang - "am" (Amharic), "om" (Afaan Oromo), "en" (English).
     * @param {string} domainHint - "healthcare", "legal", "education", "civic_banking".
     * @param {boolean} highStakes - Enable clinical/judicial safeguards.
     */
    async translateKeypoints(keypoints, targetLang = "am", domainHint = "healthcare", highStakes = false) {
        const resp = await fetch(`${this.baseURL}/api/v1/translate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                keypoints: keypoints,
                target_language: targetLang,
                domain_hint: domainHint,
                high_stakes_verification: highStakes
            })
        });
        if (!resp.ok) throw new Error(`Translation request failed: ${resp.statusText}`);
        return await resp.json();
    }

    /**
     * Reverse channel: Synthesize 3D avatar pose frames and facial blendshapes from text.
     */
    async produceAvatarAnimation(textPrompt, sourceLang = "am", speed = 1.0) {
        const resp = await fetch(`${this.baseURL}/api/v1/produce`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text_prompt: textPrompt,
                source_language: sourceLang,
                signing_speed: speed
            })
        });
        if (!resp.ok) throw new Error(`Avatar production failed: ${resp.statusText}`);
        return await resp.json();
    }

    /**
     * Enroll a custom family or regional sign (Few-Shot Metric Learning).
     */
    async enrollPersonalSign(signName, exemplarKeypointSequences) {
        const resp = await fetch(`${this.baseURL}/api/v1/personalize/enroll`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sign_name: signName,
                exemplar_keypoints: exemplarKeypointSequences
            })
        });
        if (!resp.ok) throw new Error(`Sign enrollment failed: ${resp.statusText}`);
        return await resp.json();
    }

    /**
     * Query personal sign catalog.
     */
    async queryPersonalSign(keypointSequence, distanceThreshold = 0.5) {
        const resp = await fetch(`${this.baseURL}/api/v1/personalize/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                keypoints: keypointSequence,
                distance_threshold: distanceThreshold
            })
        });
        if (!resp.ok) throw new Error(`Query personalization failed: ${resp.statusText}`);
        return await resp.json();
    }

    /**
     * Decode neuromuscular sEMG subvocalization signals.
     */
    async decodeSilentSpeech(emgSignals) {
        const resp = await fetch(`${this.baseURL}/api/v1/silent-speech/decode`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ emg_signals: emgSignals })
        });
        if (!resp.ok) throw new Error(`Silent speech decoding failed: ${resp.statusText}`);
        return await resp.json();
    }

    /**
     * Start a real-time bidirectional streaming session.
     */
    createStreamingSession(sessionId, onHypothesis, onAvatarReady, onError) {
        const wsUrl = `${this.wsBaseURL}/ws/stream/${sessionId}`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log(`[Tereguwami] Streaming WebSocket connected to session ${sessionId}`);
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "partial_hypothesis" && onHypothesis) {
                    onHypothesis(msg);
                } else if (msg.type === "avatar_production_ready" && onAvatarReady) {
                    onAvatarReady(msg);
                }
            } catch (err) {
                console.error("[Tereguwami] Error parsing WebSocket message", err);
            }
        };

        ws.onerror = (err) => {
            if (onError) onError(err);
        };

        return {
            sendFrame: (frameIndex, landmarks) => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: "frame",
                        frame_index: frameIndex,
                        timestamp_ms: Date.now(),
                        landmarks: landmarks
                    }));
                }
            },
            sendSpeechForAvatar: (text) => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: "speech_to_avatar",
                        text: text
                    }));
                }
            },
            close: () => ws.close()
        };
    }

    /**
     * Vocalize text aloud using Web Speech API (for hearing interlocutor).
     */
    speakAloud(text, lang = "am") {
        if (!("speechSynthesis" in window)) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang === "am" ? "am-ET" : lang === "om" ? "om-ET" : "en-US";
        utterance.rate = 0.95;
        window.speechSynthesis.speak(utterance);
    }
}

// Universal export
if (typeof module !== "undefined" && module.exports) {
    module.exports = { TereguwamiSDK };
} else {
    window.TereguwamiSDK = TereguwamiSDK;
}
