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
     * Vocalize text aloud using multi-tiered Web Speech API + Audio Stream fallback.
     * @param {string} text - Text to vocalize
     * @param {string} lang - "am", "om", or "en"
     * @param {Function} onStart - callback when speech begins
     * @param {Function} onEnd - callback when speech completes
     */
    speakAloud(text, lang = "am", onStart = null, onEnd = null) {
        if (!text || !text.trim()) {
            if (onEnd) onEnd();
            return;
        }

        // Cancel previous audio or speech
        this.stopSpeaking();

        if (onStart) onStart();

        // 1. If language is English and Web Speech is available, use SpeechSynthesis
        if (lang === "en" && "speechSynthesis" in window) {
            try {
                window.speechSynthesis.resume();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = "en-US";
                utterance.rate = 1.0;
                utterance.onend = () => { if (onEnd) onEnd(); };
                utterance.onerror = (e) => {
                    console.warn("[TTS] Web Speech English error, falling back to audio stream:", e);
                    this._playAudioStream(text, "en", onEnd);
                };
                window.speechSynthesis.speak(utterance);
                return;
            } catch (err) {
                console.warn("[TTS] SpeechSynthesis failed:", err);
            }
        }

        // 2. For Amharic ('am') & Afaan Oromoo ('om'), check if a native browser voice actually exists
        if ("speechSynthesis" in window) {
            try {
                const voices = window.speechSynthesis.getVoices() || [];
                const targetCode = lang === "am" ? "am" : lang === "om" ? "om" : "en";
                const matchingVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith(targetCode));

                if (matchingVoice) {
                    window.speechSynthesis.resume();
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.voice = matchingVoice;
                    utterance.lang = matchingVoice.lang;
                    utterance.rate = 0.95;
                    utterance.onend = () => { if (onEnd) onEnd(); };
                    utterance.onerror = () => { this._playAudioStream(text, lang, onEnd); };
                    window.speechSynthesis.speak(utterance);
                    return;
                }
            } catch (e) {
                // Fallthrough to stream
            }
        }

        // 3. High-Quality Audio Stream for Amharic / Afaan Oromoo / Fallback
        this._playAudioStream(text, lang, onEnd);
    }

    _playAudioStream(text, lang, onEnd) {
        const langCode = lang === "om" ? "om" : lang === "am" ? "am" : "en";
        const ttsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&tl=${langCode}&client=tw-ob&q=${encodeURIComponent(text)}`;
        const audio = new Audio(ttsUrl);
        this._currentAudio = audio;

        audio.onended = () => {
            this._currentAudio = null;
            if (onEnd) onEnd();
        };

        audio.onerror = (err) => {
            console.warn("[TTS] Audio stream playback note:", err);
            this._currentAudio = null;
            // 4. Final Fallback: Attempt browser default voice anyway
            if ("speechSynthesis" in window) {
                try {
                    window.speechSynthesis.resume();
                    const fallbackUtterance = new SpeechSynthesisUtterance(text);
                    fallbackUtterance.rate = 0.9;
                    fallbackUtterance.onend = () => { if (onEnd) onEnd(); };
                    fallbackUtterance.onerror = () => { if (onEnd) onEnd(); };
                    window.speechSynthesis.speak(fallbackUtterance);
                    return;
                } catch (e) {}
            }
            if (onEnd) onEnd();
        };

        audio.play().catch(playErr => {
            console.warn("[TTS] Audio stream play note:", playErr);
            if ("speechSynthesis" in window) {
                try {
                    window.speechSynthesis.resume();
                    const fallbackUtterance = new SpeechSynthesisUtterance(text);
                    fallbackUtterance.onend = () => { if (onEnd) onEnd(); };
                    fallbackUtterance.onerror = () => { if (onEnd) onEnd(); };
                    window.speechSynthesis.speak(fallbackUtterance);
                    return;
                } catch (e) {}
            }
            if (onEnd) onEnd();
        });
    }

    /**
     * Cancel any active audio playback or speech synthesis.
     */
    stopSpeaking() {
        if (this._currentAudio) {
            try {
                this._currentAudio.pause();
                this._currentAudio.currentTime = 0;
            } catch (e) {}
            this._currentAudio = null;
        }
        if ("speechSynthesis" in window) {
            try {
                window.speechSynthesis.cancel();
            } catch (e) {}
        }
    }
}

// Universal export
if (typeof module !== "undefined" && module.exports) {
    module.exports = { TereguwamiSDK };
} else {
    window.TereguwamiSDK = TereguwamiSDK;
}
