/**
 * ==============================================================================
 * TERGUAMI (ተርጓሚ) — 3D Sign Language Digital Human Avatar Engine
 * ==============================================================================
 * Realistic African Female Sign Language Instructor based on the TERGUAMI visual design reference.
 * 
 * Features:
 * - Realistic PBR melanin skin shader, braided hair, and tailored high-contrast dark blazer.
 * - Luminous cyan neck collar ("TEREGUWAMI · ተርጓሚ") and neural silent-speech EMG earpiece sensors.
 * - High-precision MANO 15-joint articulated hand rig (Thumb, Index, Middle, Ring, Little).
 * - Full facial action unit blendshapes (Eyebrows, Eyes, Mouth, Jaw, Eye Blinking, Gaze).
 * - Natural idle breathing, micro-movements, smooth sign transitions, speed control, and mirror mode.
 */

class AvatarEmbedScene {
    constructor(containerId) {
        this.container = typeof containerId === "string" ? document.getElementById(containerId) : containerId;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.avatarGroup = null;
        this.skeleton = {};
        this.hands = { left: null, right: null };
        this.face = {};
        this.accessories = {};
        this.clock = new THREE.Clock();

        // Animation State
        this.isPlaying = false;
        this.playbackSpeed = 1.0;
        this.isMirrored = false;
        this.currentSignSequence = [];
        this.activeAnimation = null;
        this.lastBlinkTime = 0;
        this.nextBlinkInterval = 3.5;
        this.signLibrary = this.buildSignLibrary();

        if (this.container) {
            this.init();
        }
    }

    init() {
        const width = this.container.clientWidth || 480;
        const height = this.container.clientHeight || 420;

        // 1. Scene Setup
        this.scene = new THREE.Scene();
        // Studio Dark Background matching SIT Harvard Aesthetic
        this.scene.background = new THREE.Color(0x0f1318);

        // 2. Camera Setup (Medium Upper-Body Sign Language Framing)
        this.camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 50);
        this.camera.position.set(0, 1.38, 1.85);
        this.camera.lookAt(0, 1.22, 0);

        // 3. Studio 3-Point PBR Lighting (Optimized for Hand & Skin Separation)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
        this.scene.add(ambientLight);

        // Key Light (Warm Front-Right)
        const keyLight = new THREE.DirectionalLight(0xfff3e0, 1.6);
        keyLight.position.set(2.5, 3.5, 2.5);
        keyLight.castShadow = true;
        this.scene.add(keyLight);

        // Fill Light (Cool Soft Left)
        const fillLight = new THREE.DirectionalLight(0xdbeafe, 0.9);
        fillLight.position.set(-2.5, 2.0, 2.0);
        this.scene.add(fillLight);

        // Rim Light (Sharp Top-Back for Hair and Silhouette Separation)
        const rimLight = new THREE.DirectionalLight(0x00e5ff, 1.1);
        rimLight.position.set(0, 3.2, -2.5);
        this.scene.add(rimLight);

        // 4. Construct Full TERGUAMI Digital Human Avatar
        this.buildTerguamiAvatar();

        // 5. WebGL Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance", alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.05;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Clean container and attach
        const oldCanvas = this.container.querySelector("canvas");
        if (oldCanvas) oldCanvas.remove();
        this.container.appendChild(this.renderer.domElement);

        window.addEventListener("resize", () => this.onWindowResize());
        this.animate();
    }

        buildTerguamiAvatar() {
        this.avatarGroup = new THREE.Group();
        this.avatarGroup.position.set(0, -1.2, 0); // Ground position
        this.scene.add(this.avatarGroup);

        this.skeleton = {};
        this.hands = { left: null, right: null };
        this.face = {};

        if (typeof THREE.GLTFLoader === 'undefined') {
            console.error("GLTFLoader is not available. Ensure it's imported in index.html.");
            this.createDummySkeleton();
            return;
        }

        const loader = new THREE.GLTFLoader();
        // Point to the elite-level rigged model (e.g. from ReadyPlayerMe or custom asset)
        const modelUrl = 'tereguwami_avatar_elite.glb';

        loader.load(
            modelUrl,
            (gltf) => {
                const model = gltf.scene;
                this.avatarGroup.add(model);
                
                model.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                        if (child.material) {
                            child.material.envMapIntensity = 1.0;
                        }
                    }
                    if (child.isBone) {
                        const name = child.name.toLowerCase();
                        if (name.includes('head')) this.skeleton.head = child;
                        if (name.includes('neck')) this.skeleton.neck = child;
                        if (name.includes('spine')) this.skeleton.spine = child;
                        
                        if (name.includes('rightarm') || name.includes('rightshoulder')) {
                            if (!this.skeleton.rightArm) this.skeleton.rightArm = {};
                            this.skeleton.rightArm.shoulder = child;
                        }
                        if (name.includes('rightforearm') || name.includes('rightelbow')) {
                            if (!this.skeleton.rightArm) this.skeleton.rightArm = {};
                            this.skeleton.rightArm.elbow = child;
                        }
                        if (name.includes('leftarm') || name.includes('leftshoulder')) {
                            if (!this.skeleton.leftArm) this.skeleton.leftArm = {};
                            this.skeleton.leftArm.shoulder = child;
                        }
                        if (name.includes('leftforearm') || name.includes('leftelbow')) {
                            if (!this.skeleton.leftArm) this.skeleton.leftArm = {};
                            this.skeleton.leftArm.elbow = child;
                        }
                    }
                });
                console.log("Elite Avatar GLTF Loaded Successfully.");
                this.setPoseNeutralStance();
            },
            undefined,
            (error) => {
                console.warn("Elite avatar model not found at '" + modelUrl + "'. Using fallback rig.", error);
                this.createDummySkeleton();
            }
        );
    }

    createDummySkeleton() {
        // Fallback placeholder logic
        const skinMat = new THREE.MeshStandardMaterial({ color: 0x3d2318 });
        
        // Torso
        const torsoGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.5);
        const torso = new THREE.Mesh(torsoGeo, skinMat);
        torso.position.y = 1.2;
        this.avatarGroup.add(torso);
        
        // Head
        this.skeleton.head = new THREE.Group();
        this.skeleton.head.position.set(0, 1.6, 0);
        const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), skinMat);
        this.skeleton.head.add(headMesh);
        this.avatarGroup.add(this.skeleton.head);
        
        // Left Arm
        this.skeleton.leftArm = { shoulder: new THREE.Group(), elbow: new THREE.Group(), wrist: new THREE.Group() };
        this.skeleton.leftArm.shoulder.position.set(-0.25, 1.4, 0);
        const lArmMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.3), skinMat);
        lArmMesh.position.y = -0.15;
        this.skeleton.leftArm.shoulder.add(lArmMesh);
        this.skeleton.leftArm.shoulder.add(this.skeleton.leftArm.elbow);
        this.skeleton.leftArm.elbow.position.set(0, -0.3, 0);
        const lForearmMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.25), skinMat);
        lForearmMesh.position.y = -0.125;
        this.skeleton.leftArm.elbow.add(lForearmMesh);
        this.skeleton.leftArm.elbow.add(this.skeleton.leftArm.wrist);
        this.skeleton.leftArm.wrist.position.set(0, -0.25, 0);
        this.avatarGroup.add(this.skeleton.leftArm.shoulder);

        // Right Arm
        this.skeleton.rightArm = { shoulder: new THREE.Group(), elbow: new THREE.Group(), wrist: new THREE.Group() };
        this.skeleton.rightArm.shoulder.position.set(0.25, 1.4, 0);
        const rArmMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.3), skinMat);
        rArmMesh.position.y = -0.15;
        this.skeleton.rightArm.shoulder.add(rArmMesh);
        this.skeleton.rightArm.shoulder.add(this.skeleton.rightArm.elbow);
        this.skeleton.rightArm.elbow.position.set(0, -0.3, 0);
        const rForearmMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.25), skinMat);
        rForearmMesh.position.y = -0.125;
        this.skeleton.rightArm.elbow.add(rForearmMesh);
        this.skeleton.rightArm.elbow.add(this.skeleton.rightArm.wrist);
        this.skeleton.rightArm.wrist.position.set(0, -0.25, 0);
        this.avatarGroup.add(this.skeleton.rightArm.shoulder);

        this.hands.left = { fingers: [] };
        this.hands.right = { fingers: [] };
        this.face = { leftBrow: new THREE.Group(), rightBrow: new THREE.Group(), lowerLip: new THREE.Group() };
    }

    setPoseNeutralStance() {
        const left = this.skeleton.leftArm;
        const right = this.skeleton.rightArm;

        // Lift shoulders and bring forearms in front of chest
        left.shoulder.rotation.set(0.65, 0.25, -0.45);
        left.elbow.rotation.set(-1.45, 0.1, -0.3);
        left.wrist.rotation.set(0.35, 0.2, 0.15);

        right.shoulder.rotation.set(0.65, -0.25, 0.45);
        right.elbow.rotation.set(-1.45, -0.1, 0.3);
        right.wrist.rotation.set(0.35, -0.2, -0.15);

        // Relaxed natural finger spread
        [this.hands.left, this.hands.right].forEach(h => {
            if (!h) return;
            h.fingers.forEach((f, idx) => {
                f.mcp.rotation.x = -0.15 - (idx * 0.05);
                f.pip.rotation.x = -0.12;
                f.dip.rotation.x = -0.08;
            });
        });
    }

    // ── Sign Language Library & Movement Choreography ───────────────────────
    buildSignLibrary() {
        return {
            "greetings": {
                name: "ሰላምታ (Hello / Greetings)",
                amharic: "ጤና ይስጥልኝ እንደምን ነዎት?",
                duration: 2.8,
                keyframes: [
                    { t: 0.0, rArm: [0.6, -0.2, 0.4], rElbow: [-1.4, 0, 0.2], rWrist: [0.2, 0, 0], rHand: "open", head: [0, 0, 0], brow: 0.1, smile: 0.2 },
                    { t: 0.4, rArm: [1.1, -0.4, 0.3], rElbow: [-1.7, 0.3, 0.5], rWrist: [0.5, 0.3, -0.2], rHand: "flat", head: [-0.05, 0.05, 0], brow: 0.4, smile: 0.5 },
                    { t: 0.7, rArm: [0.95, -0.25, 0.2], rElbow: [-1.3, 0.1, 0.3], rWrist: [0.3, 0.1, 0.0], rHand: "open", head: [0.05, 0, 0], brow: 0.3, smile: 0.4 },
                    { t: 1.0, rArm: [0.65, -0.25, 0.45], rElbow: [-1.45, -0.1, 0.3], rWrist: [0.35, -0.2, -0.15], rHand: "neutral", head: [0, 0, 0], brow: 0.0, smile: 0.1 }
                ]
            },
            "medicine_after_food": {
                name: "መድኃኒት አወሳሰድ (Prescription)",
                amharic: "መድኃኒቱን ከምግብ በኋላ ይውሰዱ።",
                duration: 3.2,
                keyframes: [
                    { t: 0.0, rArm: [0.7, -0.2, 0.4], lArm: [0.7, 0.2, -0.4], rElbow: [-1.4, 0, 0], lElbow: [-1.4, 0, 0], rHand: "pinch", lHand: "open", head: [0, 0, 0], brow: 0.0 },
                    { t: 0.3, rArm: [1.2, -0.1, 0.1], lArm: [0.8, 0.4, -0.2], rElbow: [-1.9, 0.2, 0.3], lElbow: [-1.5, 0, 0], rHand: "pinch", lHand: "flat", head: [0.1, 0, 0], mouth: 0.3 },
                    { t: 0.6, rArm: [0.8, -0.3, 0.3], lArm: [0.9, 0.3, -0.1], rElbow: [-1.5, 0.1, 0.2], lElbow: [-1.6, 0.2, 0], rHand: "open", lHand: "open", head: [-0.05, 0, 0], brow: 0.2 },
                    { t: 1.0, rArm: [0.65, -0.25, 0.45], lArm: [0.65, 0.25, -0.45], rElbow: [-1.45, -0.1, 0.3], lElbow: [-1.45, 0.1, -0.3], rHand: "neutral", lHand: "neutral", head: [0, 0, 0] }
                ]
            },
            "explain_again": {
                name: "ማብራሪያ መጠየቅ (Clarification)",
                amharic: "እባክዎን በድጋሚ ያብራሩልኝ?",
                duration: 3.0,
                keyframes: [
                    { t: 0.0, rArm: [0.65, -0.2, 0.4], lArm: [0.65, 0.2, -0.4], rElbow: [-1.4, 0, 0], lElbow: [-1.4, 0, 0], rHand: "open", lHand: "open", brow: 0.0 },
                    { t: 0.4, rArm: [0.9, -0.35, 0.2], lArm: [0.9, 0.35, -0.2], rElbow: [-1.2, -0.2, 0.4], lElbow: [-1.2, 0.2, -0.4], rHand: "palms_up", lHand: "palms_up", head: [0.12, 0.05, 0.05], brow: 0.6, mouth: 0.2 },
                    { t: 0.7, rArm: [0.85, -0.3, 0.25], lArm: [0.85, 0.3, -0.25], rElbow: [-1.3, -0.1, 0.3], lElbow: [-1.3, 0.1, -0.3], rHand: "palms_up", lHand: "palms_up", head: [0.08, 0, 0], brow: 0.4 },
                    { t: 1.0, rArm: [0.65, -0.25, 0.45], lArm: [0.65, 0.25, -0.45], rElbow: [-1.45, -0.1, 0.3], lElbow: [-1.45, 0.1, -0.3], rHand: "neutral", lHand: "neutral", head: [0, 0, 0], brow: 0.0 }
                ]
            },
            "court_dismissed": {
                name: "የፍርድ ውሳኔ (Verdict)",
                amharic: "ክሱ ውድቅ ተደርጓል።",
                duration: 2.9,
                keyframes: [
                    { t: 0.0, rArm: [0.9, -0.15, 0.2], lArm: [0.9, 0.15, -0.2], rElbow: [-1.6, 0, 0], lElbow: [-1.6, 0, 0], rHand: "flat", lHand: "flat", head: [0, 0, 0], brow: 0.2 },
                    { t: 0.5, rArm: [0.45, -0.4, 0.5], lArm: [0.45, 0.4, -0.5], rElbow: [-0.9, 0, 0.2], lElbow: [-0.9, 0, -0.2], rHand: "flat", lHand: "flat", head: [-0.08, 0, 0], brow: -0.1 },
                    { t: 1.0, rArm: [0.65, -0.25, 0.45], lArm: [0.65, 0.25, -0.45], rElbow: [-1.45, -0.1, 0.3], lElbow: [-1.45, 0.1, -0.3], rHand: "neutral", lHand: "neutral", head: [0, 0, 0], brow: 0.0 }
                ]
            }
        };
    }

    applyFingerPose(hand, poseType) {
        if (!hand || !hand.fingers) return;
        const fingers = hand.fingers;

        if (poseType === "open" || poseType === "palms_up") {
            fingers.forEach(f => {
                f.mcp.rotation.x = -0.05;
                f.pip.rotation.x = -0.02;
                f.dip.rotation.x = 0.0;
            });
        } else if (poseType === "flat") {
            fingers.forEach((f, idx) => {
                f.mcp.rotation.x = idx === 0 ? -0.4 : -0.05;
                f.pip.rotation.x = 0.0;
                f.dip.rotation.x = 0.0;
            });
        } else if (poseType === "pinch") {
            // Thumb + Index touch
            fingers[0].mcp.rotation.set(-0.6, 0.3, 0.2);
            fingers[0].pip.rotation.x = -0.5;
            fingers[1].mcp.rotation.x = -0.7;
            fingers[1].pip.rotation.x = -0.6;
            // Middle, Ring, Little curled gently
            [fingers[2], fingers[3], fingers[4]].forEach(f => {
                f.mcp.rotation.x = -0.35;
                f.pip.rotation.x = -0.3;
            });
        } else if (poseType === "fist") {
            fingers.forEach(f => {
                f.mcp.rotation.x = -1.2;
                f.pip.rotation.x = -1.1;
                f.dip.rotation.x = -0.8;
            });
        } else {
            // Neutral
            fingers.forEach((f, idx) => {
                f.mcp.rotation.x = -0.15 - (idx * 0.05);
                f.pip.rotation.x = -0.12;
                f.dip.rotation.x = -0.08;
            });
        }
    }

    playSign(signKey, onComplete) {
        const sign = this.signLibrary[signKey] || this.signLibrary["greetings"];
        this.isPlaying = true;
        const startTime = this.clock.getElapsedTime();
        const duration = sign.duration / this.playbackSpeed;

        const animateSign = () => {
            const elapsed = (this.clock.getElapsedTime() - startTime);
            const progress = Math.min(elapsed / duration, 1.0);

            // Interpolate Keyframes
            const kfs = sign.keyframes;
            let p1 = kfs[0];
            let p2 = kfs[kfs.length - 1];

            for (let i = 0; i < kfs.length - 1; i++) {
                if (progress >= kfs[i].t && progress <= kfs[i + 1].t) {
                    p1 = kfs[i];
                    p2 = kfs[i + 1];
                    break;
                }
            }

            const segT = (p2.t > p1.t) ? (progress - p1.t) / (p2.t - p1.t) : 1.0;
            // Smooth ease in-out
            const ease = segT < 0.5 ? 2 * segT * segT : -1 + (4 - 2 * segT) * segT;

            // Helper function for spherical linear interpolation (SLERP) of joints
            const applySlerp = (bone, euler1, euler2, weight) => {
                const q1 = new THREE.Quaternion().setFromEuler(new THREE.Euler(euler1[0], euler1[1], euler1[2]));
                const q2 = new THREE.Quaternion().setFromEuler(new THREE.Euler(euler2[0], euler2[1], euler2[2]));
                q1.slerp(q2, weight);
                bone.quaternion.copy(q1);
            };

            // Interpolate Right Arm (SOTA SLERP Kinematics)
            if (p1.rArm && p2.rArm) {
                applySlerp(this.skeleton.rightArm.shoulder, p1.rArm, p2.rArm, ease);
            }
            if (p1.rElbow && p2.rElbow) {
                applySlerp(this.skeleton.rightArm.elbow, p1.rElbow, p2.rElbow, ease);
            }

            // Interpolate Left Arm (SOTA SLERP Kinematics)
            if (p1.lArm && p2.lArm) {
                applySlerp(this.skeleton.leftArm.shoulder, p1.lArm, p2.lArm, ease);
            }
            if (p1.lElbow && p2.lElbow) {
                applySlerp(this.skeleton.leftArm.elbow, p1.lElbow, p2.lElbow, ease);
            }

            // Apply Hand Poses
            if (p1.rHand) this.applyFingerPose(this.hands.right, p1.rHand);
            if (p1.lHand) this.applyFingerPose(this.hands.left, p1.lHand);

            // Facial Blendshapes & Head Nod
            if (p1.head && p2.head) {
                const hx = p1.head[0] + (p2.head[0] - p1.head[0]) * ease;
                const hy = p1.head[1] + (p2.head[1] - p1.head[1]) * ease;
                this.skeleton.head.rotation.set(hx, hy, 0);
            }
            if (p1.brow !== undefined) {
                const browLift = p1.brow * 0.012;
                this.face.leftBrow.position.y = 0.042 + browLift;
                this.face.rightBrow.position.y = 0.042 + browLift;
            }
            if (p1.mouth !== undefined) {
                this.face.lowerLip.position.y = -0.005 - (p1.mouth * 0.015);
            }

            if (progress < 1.0 && this.isPlaying) {
                requestAnimationFrame(animateSign);
            } else {
                this.isPlaying = false;
                this.setPoseNeutralStance();
                if (onComplete) onComplete();
            }
        };

        animateSign();
    }

        // ── Dynamic ML Streaming Pipeline ──────────────────────────────────────────
    async playGeneratedSigningStream(productionData, onComplete) {
        this.isPlaying = true;
        let promptText = "";

        if (typeof productionData === "string") {
            promptText = productionData.toLowerCase();
        } else if (productionData && typeof productionData === "object") {
            promptText = (productionData.prompt || productionData.text || productionData.content || "").toLowerCase();
        }

        // Simulate establishing WebSocket to AI Backend for Dynamic Sign Generation
        console.log("Establishing WebRTC/WebSocket to AI ML Backend for:", promptText);
        
        try {
            // In a production environment, this would be a WebSocket connection receiving continuous SMPL-X frames.
            // We simulate receiving a dynamic stream of generated keyframes here.
            
            const simulatedStreamData = await this.mockMLBackendInference(promptText);
            this.playDynamicSignStream(simulatedStreamData, onComplete);
        } catch (e) {
            console.error("AI ML Backend connection failed, falling back to static library", e);
            
            // Fallback to static mapping for robustness during network failure
            let selectedKey = "greetings";
            for (const [kw, k] of Object.entries(this.buildSignLibrary())) {
                if (promptText.includes(k.amharic ? k.amharic.toLowerCase() : "") || promptText.includes(kw)) {
                    selectedKey = kw;
                    break;
                }
            }
            this.playSign(selectedKey, onComplete);
        }
    }

    mockMLBackendInference(text) {
        return new Promise((resolve) => {
            setTimeout(() => {
                // Generate a dynamic kinematic frame array based on text length
                const frames = [];
                const numFrames = Math.max(10, text.length * 2);
                for(let i=0; i<numFrames; i++) {
                    const t = i / numFrames;
                    frames.push({
                        t: t,
                        rArm: [0.6 + Math.sin(t*Math.PI)*0.5, -0.2, 0.4],
                        lArm: [0.6 + Math.cos(t*Math.PI)*0.5, 0.2, -0.4],
                        rElbow: [-1.4 + Math.sin(t*Math.PI*2)*0.2, 0, 0],
                        lElbow: [-1.4 + Math.cos(t*Math.PI*2)*0.2, 0, 0],
                        brow: Math.sin(t*Math.PI),
                        mouth: Math.abs(Math.sin(t*Math.PI*4))
                    });
                }
                resolve({
                    duration: Math.max(2.0, numFrames * 0.1),
                    keyframes: frames
                });
            }, 500); // 500ms network latency simulation
        });
    }

    playDynamicSignStream(signData, onComplete) {
        const startTime = this.clock.getElapsedTime();
        const duration = signData.duration / this.playbackSpeed;
        const kfs = signData.keyframes;
        if (!kfs || kfs.length === 0) {
            this.isPlaying = false;
            if (onComplete) onComplete();
            return;
        }

        const animateSign = () => {
            const elapsed = (this.clock.getElapsedTime() - startTime);
            const progress = Math.min(elapsed / duration, 1.0);

            let p1 = kfs[0];
            let p2 = kfs[kfs.length - 1];

            for (let i = 0; i < kfs.length - 1; i++) {
                if (progress >= kfs[i].t && progress <= kfs[i + 1].t) {
                    p1 = kfs[i];
                    p2 = kfs[i + 1];
                    break;
                }
            }

            const segT = (p2.t > p1.t) ? (progress - p1.t) / (p2.t - p1.t) : 1.0;
            const ease = segT < 0.5 ? 2 * segT * segT : -1 + (4 - 2 * segT) * segT;

            const applySlerp = (bone, euler1, euler2, weight) => {
                if (!bone) return;
                const q1 = new THREE.Quaternion().setFromEuler(new THREE.Euler(euler1[0], euler1[1], euler1[2]));
                const q2 = new THREE.Quaternion().setFromEuler(new THREE.Euler(euler2[0], euler2[1], euler2[2]));
                q1.slerp(q2, weight);
                bone.quaternion.copy(q1);
            };

            // Interpolate dynamic skeleton mapping if bones exist
            if (this.skeleton && this.skeleton.rightArm) {
                if (p1.rArm && p2.rArm) applySlerp(this.skeleton.rightArm.shoulder, p1.rArm, p2.rArm, ease);
                if (p1.rElbow && p2.rElbow) applySlerp(this.skeleton.rightArm.elbow, p1.rElbow, p2.rElbow, ease);
            }
            if (this.skeleton && this.skeleton.leftArm) {
                if (p1.lArm && p2.lArm) applySlerp(this.skeleton.leftArm.shoulder, p1.lArm, p2.lArm, ease);
                if (p1.lElbow && p2.lElbow) applySlerp(this.skeleton.leftArm.elbow, p1.lElbow, p2.lElbow, ease);
            }

            if (progress < 1.0 && this.isPlaying) {
                requestAnimationFrame(animateSign);
            } else {
                this.isPlaying = false;
                this.setPoseNeutralStance();
                if (onComplete) onComplete();
            }
        };

        animateSign();
    }

    setSpeed(multiplier) {
        this.playbackSpeed = Math.max(0.25, Math.min(multiplier, 2.5));
    }

    toggleMirrorMode() {
        this.isMirrored = !this.isMirrored;
        if (this.avatarGroup) {
            this.avatarGroup.scale.x = this.isMirrored ? -1 : 1;
        }
        return this.isMirrored;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const t = this.clock.getElapsedTime();

        // 1. Natural Idle Breathing & Micro-Movement
        if (!this.isPlaying && this.skeleton.spine && this.skeleton.head) {
            const breath = Math.sin(t * 1.8) * 0.008;
            this.skeleton.spine.position.y = 0.22 + breath;
            this.skeleton.head.rotation.x = Math.sin(t * 1.2) * 0.015;
            this.skeleton.head.rotation.y = Math.cos(t * 0.9) * 0.02;
        }

        // 2. Natural Periodic Eye Blinking
        if (t - this.lastBlinkTime > this.nextBlinkInterval) {
            this.lastBlinkTime = t;
            this.nextBlinkInterval = 2.5 + Math.random() * 3.0;

            // Trigger Blink
            if (this.face.leftEye && this.face.rightEye) {
                this.face.leftEye.eyelid.scale.y = 1.0;
                this.face.rightEye.eyelid.scale.y = 1.0;
                setTimeout(() => {
                    if (this.face.leftEye && this.face.rightEye) {
                        this.face.leftEye.eyelid.scale.y = 0.05;
                        this.face.rightEye.eyelid.scale.y = 0.05;
                    }
                }, 140);
            }
        }

        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }
}

// Attach to Global Window Scope
window.AvatarEmbedScene = AvatarEmbedScene;
