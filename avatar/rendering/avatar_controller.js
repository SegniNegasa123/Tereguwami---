/**
 * SignAvatars 3D Motion Stream Controller (§8.5, §11)
 * Adapted from SignAvatars (Zhengdi Yu et al., ECCV 2024 / SMPL-X Holistic Motion Framework)
 *
 * Drives continuous SMPL-X body joints, MANO articulated fingers, and FLAME facial expressions.
 */

class SignAvatarsController {
    constructor() {
        this.currentFrameIndex = 0;
        this.animationFrames = [];
        this.isPlaying = false;
        this.fps = 30;
        this.speed = 1.0;
        this.smoothingFactor = 0.25;
        this.onFrameCallback = null;
        this.onCompleteCallback = null;
        this.timer = null;
    }

    loadSequence(frames, fps = 30, speed = 1.0) {
        this.animationFrames = frames || [];
        this.fps = fps;
        this.speed = speed;
        this.currentFrameIndex = 0;
    }

    play(onComplete = null) {
        if (!this.animationFrames.length) {
            if (onComplete) onComplete();
            return;
        }
        this.onCompleteCallback = onComplete;
        this.isPlaying = true;
        if (this.timer) clearTimeout(this.timer);
        this._tick();
    }

    pause() {
        this.isPlaying = false;
        if (this.timer) clearTimeout(this.timer);
    }

    reset() {
        this.currentFrameIndex = 0;
        this.isPlaying = false;
        if (this.timer) clearTimeout(this.timer);
    }

    _tick() {
        if (!this.isPlaying) return;

        const frame = this.animationFrames[this.currentFrameIndex];
        if (this.onFrameCallback && frame) {
            this.onFrameCallback(frame, this.currentFrameIndex);
        }

        this.currentFrameIndex++;
        if (this.currentFrameIndex >= this.animationFrames.length) {
            this.isPlaying = false;
            this.currentFrameIndex = 0;
            if (this.onCompleteCallback) {
                this.onCompleteCallback();
            }
            return;
        }

        const interval = Math.max(10, 1000 / (this.fps * this.speed));
        this.timer = setTimeout(() => this._tick(), interval);
    }
}

// Global & CommonJS export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SignAvatarsController, AvatarController: SignAvatarsController };
} else {
    window.SignAvatarsController = SignAvatarsController;
    window.AvatarController = SignAvatarsController;
}
