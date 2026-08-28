/**
 * Reactive Application State Store for Tereguwami (ተርጓሚ) Client
 */

class StateStore {
    constructor() {
        this.state = {
            targetLanguage: "am", // "am", "om", "en"
            activeDomain: "healthcare", // "healthcare", "legal", "education", "civic_banking"
            highStakesMode: false,
            autoVocalize: true,
            isStreaming: false,
            currentHypothesis: "",
            confidence: 0.0,
            conversationHistory: [],
            enrolledSigns: [],
            telemetry: {
                eyebrowAU: 0.0,
                mouthAU: 0.0,
                headTiltDeg: 0.0,
                latencyMs: 0.0
            }
        };
        this.listeners = [];
    }

    getState() {
        return this.state;
    }

    update(partial) {
        this.state = { ...this.state, ...partial };
        this.listeners.forEach(fn => fn(this.state));
    }

    subscribe(listener) {
        this.listeners.push(listener);
        listener(this.state);
        return () => {
            this.listeners = this.listeners.filter(l => l !== listener);
        };
    }

    addMessage(sender, text, lang, confidence = 1.0) {
        const msg = {
            id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
            sender: sender, // "deaf_signer" or "hearing_user"
            text: text,
            lang: lang,
            confidence: confidence,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        };
        this.update({
            conversationHistory: [...this.state.conversationHistory, msg]
        });
        return msg;
    }
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { StateStore };
} else {
    window.StateStore = StateStore;
}
