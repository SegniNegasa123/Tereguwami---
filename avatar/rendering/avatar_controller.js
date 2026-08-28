/**
 * Tereguwami 3D Avatar Animation Controller (§8.5, §11)
 * Manages skeletal joint smoothing, facial blendshape blending, and real-time playback.
 */

class AvatarController {
    constructor() {
        this.currentFrameIndex = 0;
        this.animationFrames = [];
        this.isPlaying = false;
        this.fps = 30;
        this.speed = 1.0;
        this.smoothingFactor = 0.25; // Spline interpolation weight
        this.onFrameCallback = null;
    }

    loadSequence(frames, fps = 30) {
        this.animationFrames = frames;
        this.fps = fps;
        this.currentFrameIndex = 0;
    }

    play() {
        if (!this.animationFrames.length) return;
        this.isPlaying = true;
        this._tick();
    }

    pause() {
        this.isPlaying = false;
    }

    reset() {
        this.currentFrameIndex = 0;
        this.isPlaying = false;
    }

    _tick() {
        if (!this.isPlaying) return;

        const frame = this.animationFrames[this.currentFrameIndex];
        if (this.onFrameCallback && frame) {
            this.onFrameCallback(frame);
        }

        this.currentFrameIndex++;
        if (this.currentFrameIndex >= this.animationFrames.length) {
            this.currentFrameIndex = 0; // Loop or pause
            this.isPlaying = false;
            return;
        }

        const interval = (1000 / (this.fps * this.speed));
        setTimeout(() => this._tick(), interval);
    }
}

if (typeof module !== 'undefined') {
    module.exports = { AvatarController };
}
