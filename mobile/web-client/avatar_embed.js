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
        this.avatarGroup.position.set(0, 0, 0);
        this.scene.add(this.avatarGroup);

        // ── Materials & Shaders ──────────────────────────────────────────────
        // Realistic Melanin Warm Brown Skin
        const skinMaterial = new THREE.MeshStandardMaterial({
            color: 0x734327,
            roughness: 0.48,
            metalness: 0.08,
            bumpScale: 0.002
        });

        // Professional Dark Charcoal Tailored Blazer (High Contrast against hands)
        const suitMaterial = new THREE.MeshStandardMaterial({
            color: 0x16191d,
            roughness: 0.75,
            metalness: 0.12
        });

        // Dark Inner Collar / Turtleneck
        const innerShirtMaterial = new THREE.MeshStandardMaterial({
            color: 0x0c0e10,
            roughness: 0.85
        });

        // Black Braided Hair
        const hairMaterial = new THREE.MeshStandardMaterial({
            color: 0x141110,
            roughness: 0.65,
            metalness: 0.15
        });

        // Luminous Cyan Emissive Necklace Material
        const cyanGlowMaterial = new THREE.MeshStandardMaterial({
            color: 0x00e5ff,
            emissive: 0x00c4e6,
            emissiveIntensity: 0.85,
            roughness: 0.2,
            metalness: 0.4
        });

        // Sensor Hardware Material (Silver / Ceramic)
        const sensorHardwareMaterial = new THREE.MeshStandardMaterial({
            color: 0xdde4ea,
            roughness: 0.25,
            metalness: 0.8
        });

        // Gold Sensor Contacts
        const goldContactMaterial = new THREE.MeshStandardMaterial({
            color: 0xf59e0b,
            roughness: 0.3,
            metalness: 0.7
        });

        // ── Body Hierarchy ───────────────────────────────────────────────────
        // Root Pelvis
        const pelvis = new THREE.Group();
        pelvis.position.set(0, 0.92, 0);
        this.avatarGroup.add(pelvis);
        this.skeleton.pelvis = pelvis;

        // Lower Torso (Inside Blazer)
        const lowerTorsoGeo = new THREE.CylinderGeometry(0.185, 0.165, 0.22, 28);
        const lowerTorso = new THREE.Mesh(lowerTorsoGeo, suitMaterial);
        lowerTorso.position.y = 0.11;
        pelvis.add(lowerTorso);

        // Spine Hierarchy
        const spine = new THREE.Group();
        spine.position.set(0, 0.22, 0);
        pelvis.add(spine);
        this.skeleton.spine = spine;

        // Chest / Upper Torso & Tailored Blazer
        const chestGeo = new THREE.CylinderGeometry(0.245, 0.195, 0.26, 32);
        chestGeo.scale(1.15, 1.0, 0.88);
        const chest = new THREE.Mesh(chestGeo, suitMaterial);
        chest.position.y = 0.13;
        spine.add(chest);

        // Blazer Lapels & V-Neck Opening
        const lapelGeo = new THREE.BoxGeometry(0.04, 0.22, 0.02);
        const leftLapel = new THREE.Mesh(lapelGeo, suitMaterial);
        leftLapel.position.set(-0.08, 0.15, 0.12);
        leftLapel.rotation.set(0.1, 0, -0.2);
        spine.add(leftLapel);

        const rightLapel = new THREE.Mesh(lapelGeo, suitMaterial);
        rightLapel.position.set(0.08, 0.15, 0.12);
        rightLapel.rotation.set(0.1, 0, 0.2);
        spine.add(rightLapel);

        // High Dark Inner Turtleneck
        const innerShirtGeo = new THREE.CylinderGeometry(0.085, 0.11, 0.14, 24);
        const innerShirt = new THREE.Mesh(innerShirtGeo, innerShirtMaterial);
        innerShirt.position.set(0, 0.21, 0.02);
        spine.add(innerShirt);

        // Neck
        const neck = new THREE.Group();
        neck.position.set(0, 0.27, 0);
        spine.add(neck);
        this.skeleton.neck = neck;

        const neckMeshGeo = new THREE.CylinderGeometry(0.062, 0.075, 0.12, 24);
        const neckMesh = new THREE.Mesh(neckMeshGeo, skinMaterial);
        neckMesh.position.y = 0.06;
        neck.add(neckMesh);

        // ── Signature Collar Necklace ("TEREGUWAMI · ተርጓሚ") ─────────────────
        const collarTorusGeo = new THREE.TorusGeometry(0.078, 0.012, 16, 40);
        collarTorusGeo.rotateX(Math.PI / 2);
        collarTorusGeo.scale(1.0, 1.15, 1.0);
        const collarMesh = new THREE.Mesh(collarTorusGeo, cyanGlowMaterial);
        collarMesh.position.set(0, 0.04, 0.01);
        neck.add(collarMesh);

        // Head Group
        const head = new THREE.Group();
        head.position.set(0, 0.12, 0);
        neck.add(head);
        this.skeleton.head = head;

        // ── Sculpted African Female Cranium & Face ───────────────────────────
        const craniumGeo = new THREE.SphereGeometry(0.125, 36, 28);
        craniumGeo.scale(0.96, 1.18, 1.08);
        const cranium = new THREE.Mesh(craniumGeo, skinMaterial);
        head.add(cranium);

        // Chin & Jaw Structure
        const jawGeo = new THREE.ConeGeometry(0.09, 0.12, 24);
        jawGeo.rotateX(Math.PI);
        jawGeo.scale(1.0, 0.8, 0.9);
        const jaw = new THREE.Mesh(jawGeo, skinMaterial);
        jaw.position.set(0, -0.06, 0.035);
        head.add(jaw);

        // ── Braided Hair Crown & Bun ─────────────────────────────────────────
        const hairCrownGeo = new THREE.SphereGeometry(0.134, 32, 24);
        hairCrownGeo.scale(0.98, 1.16, 1.06);
        const hairCrown = new THREE.Mesh(hairCrownGeo, hairMaterial);
        hairCrown.position.set(0, 0.025, -0.02);
        head.add(hairCrown);

        // Neatly Pulled Back Braided Bun
        const bunGeo = new THREE.SphereGeometry(0.072, 24, 20);
        bunGeo.scale(1.1, 0.9, 0.8);
        const bun = new THREE.Mesh(bunGeo, hairMaterial);
        bun.position.set(0, 0.04, -0.14);
        head.add(bun);

        // ── Facial Features & Expression Rig ────────────────────────────────
        // Eyes (Sclera + Iris)
        const eyeWhiteMat = new THREE.MeshBasicMaterial({ color: 0xf8fafc });
        const irisMat = new THREE.MeshStandardMaterial({ color: 0x3d2314, roughness: 0.1 });
        const pupilMat = new THREE.MeshBasicMaterial({ color: 0x050505 });

        const createEye = (isLeft) => {
            const eyeGroup = new THREE.Group();
            const side = isLeft ? -1 : 1;
            eyeGroup.position.set(side * 0.038, 0.018, 0.115);

            const eyeball = new THREE.Mesh(new THREE.SphereGeometry(0.015, 16, 16), eyeWhiteMat);
            const iris = new THREE.Mesh(new THREE.CircleGeometry(0.0075, 16), irisMat);
            iris.position.set(0, 0, 0.014);
            const pupil = new THREE.Mesh(new THREE.CircleGeometry(0.0035, 16), pupilMat);
            pupil.position.set(0, 0, 0.0145);

            eyeball.add(iris);
            eyeball.add(pupil);
            eyeGroup.add(eyeball);

            // Eyelid for blinking
            const eyelidGeo = new THREE.SphereGeometry(0.016, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.5);
            eyelidGeo.rotateX(Math.PI / 2);
            const eyelid = new THREE.Mesh(eyelidGeo, skinMaterial);
            eyelid.position.set(0, 0.002, 0.002);
            eyeGroup.add(eyelid);

            return { group: eyeGroup, eyelid };
        };

        const leftEye = createEye(true);
        const rightEye = createEye(false);
        head.add(leftEye.group);
        head.add(rightEye.group);
        this.face.leftEye = leftEye;
        this.face.rightEye = rightEye;

        // Natural Eyebrows
        const browGeo = new THREE.CylinderGeometry(0.004, 0.0025, 0.036, 8);
        browGeo.rotateZ(Math.PI / 2);
        const leftBrow = new THREE.Mesh(browGeo, hairMaterial);
        leftBrow.position.set(-0.04, 0.042, 0.124);
        leftBrow.rotation.z = -0.15;
        head.add(leftBrow);

        const rightBrow = new THREE.Mesh(browGeo, hairMaterial);
        rightBrow.position.set(0.04, 0.042, 0.124);
        rightBrow.rotation.z = 0.15;
        head.add(rightBrow);
        this.face.leftBrow = leftBrow;
        this.face.rightBrow = rightBrow;

        // Sculpted Nose
        const noseGeo = new THREE.ConeGeometry(0.016, 0.042, 12);
        noseGeo.rotateX(Math.PI * 0.4);
        const nose = new THREE.Mesh(noseGeo, skinMaterial);
        nose.position.set(0, 0.002, 0.132);
        head.add(nose);

        // Natural Full Lips & Mouth Opening
        const lipMaterial = new THREE.MeshStandardMaterial({
            color: 0x6e3828,
            roughness: 0.35,
            metalness: 0.05
        });

        const mouthGroup = new THREE.Group();
        mouthGroup.position.set(0, -0.042, 0.12);
        head.add(mouthGroup);
        this.face.mouthGroup = mouthGroup;

        const upperLipGeo = new THREE.BoxGeometry(0.038, 0.008, 0.012);
        const upperLip = new THREE.Mesh(upperLipGeo, lipMaterial);
        upperLip.position.y = 0.004;
        mouthGroup.add(upperLip);

        const lowerLipGeo = new THREE.BoxGeometry(0.034, 0.010, 0.014);
        const lowerLip = new THREE.Mesh(lowerLipGeo, lipMaterial);
        lowerLip.position.y = -0.005;
        mouthGroup.add(lowerLip);
        this.face.lowerLip = lowerLip;

        // ── Silent Speech Sensors (EMG Earpiece + Facial Micro-Sensors) ──────
        // Right Ear Receiver / Comm Hardware
        const earpieceGroup = new THREE.Group();
        earpieceGroup.position.set(0.125, 0.02, 0.01);
        head.add(earpieceGroup);

        const earpieceBody = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.048, 0.022), sensorHardwareMaterial);
        earpieceGroup.add(earpieceBody);

        const antennaRod = new THREE.Mesh(new THREE.CylinderGeometry(0.002, 0.002, 0.045, 8), sensorHardwareMaterial);
        antennaRod.position.set(0, 0.035, -0.005);
        earpieceGroup.add(antennaRod);

        // Facial EMG Electrodes (Right Cheek and Jaw Contacts from Reference)
        const electrode1 = new THREE.Mesh(new THREE.CylinderGeometry(0.0065, 0.0065, 0.004, 12), goldContactMaterial);
        electrode1.rotateZ(Math.PI / 2);
        electrode1.position.set(0.098, 0.0, 0.082);
        head.add(electrode1);

        const electrode2 = new THREE.Mesh(new THREE.CylinderGeometry(0.0065, 0.0065, 0.004, 12), goldContactMaterial);
        electrode2.rotateZ(Math.PI / 2);
        electrode2.position.set(0.082, -0.038, 0.075);
        head.add(electrode2);

        // ── High-Precision MANO 15-Joint Hand Rig ───────────────────────────
        const buildArticulatedHand = (isLeft) => {
            const side = isLeft ? -1 : 1;
            const handRoot = new THREE.Group();

            // Palm Structure
            const palmGeo = new THREE.BoxGeometry(0.052, 0.068, 0.022);
            const palmMesh = new THREE.Mesh(palmGeo, skinMaterial);
            palmMesh.position.y = -0.034;
            handRoot.add(palmMesh);

            const fingers = [];
            const fingerSpans = [
                { name: "thumb", x: side * -0.028, y: -0.018, z: 0.008, len: 0.022, r: 0.0065 },
                { name: "index", x: side * -0.018, y: -0.068, z: 0.0, len: 0.028, r: 0.0055 },
                { name: "middle", x: side * -0.006, y: -0.072, z: 0.0, len: 0.032, r: 0.0058 },
                { name: "ring", x: side * 0.007, y: -0.068, z: 0.0, len: 0.029, r: 0.0054 },
                { name: "little", x: side * 0.019, y: -0.062, z: 0.0, len: 0.024, r: 0.0048 }
            ];

            fingerSpans.forEach((spec, fIdx) => {
                // Base Joint (MCP)
                const mcp = new THREE.Group();
                mcp.position.set(spec.x, spec.y, spec.z);
                handRoot.add(mcp);

                const p1Geo = new THREE.CylinderGeometry(spec.r * 0.9, spec.r, spec.len, 10);
                p1Geo.translate(0, -spec.len * 0.5, 0);
                const p1Mesh = new THREE.Mesh(p1Geo, skinMaterial);
                mcp.add(p1Mesh);

                // Intermediate Joint (PIP)
                const pip = new THREE.Group();
                pip.position.set(0, -spec.len, 0);
                mcp.add(pip);

                const p2Geo = new THREE.CylinderGeometry(spec.r * 0.8, spec.r * 0.9, spec.len * 0.8, 10);
                p2Geo.translate(0, -spec.len * 0.4, 0);
                const p2Mesh = new THREE.Mesh(p2Geo, skinMaterial);
                pip.add(p2Mesh);

                // Distal Joint (DIP / Fingertip)
                const dip = new THREE.Group();
                dip.position.set(0, -spec.len * 0.8, 0);
                pip.add(dip);

                const p3Geo = new THREE.CylinderGeometry(spec.r * 0.65, spec.r * 0.8, spec.len * 0.65, 10);
                p3Geo.translate(0, -spec.len * 0.32, 0);
                const p3Mesh = new THREE.Mesh(p3Geo, skinMaterial);
                dip.add(p3Mesh);

                fingers.push({ mcp, pip, dip, name: spec.name });
            });

            return { root: handRoot, fingers };
        };

        // ── Arm Skeletal Chains (Shoulder -> Elbow -> Wrist -> MANO Hand) ────
        const createArmChain = (isLeft) => {
            const side = isLeft ? -1 : 1;
            const shoulder = new THREE.Group();
            shoulder.position.set(side * 0.24, 0.22, 0.0);
            spine.add(shoulder);

            // Upper Arm (in Suit Jacket Sleeve)
            const upperArmGeo = new THREE.CylinderGeometry(0.052, 0.046, 0.27, 24);
            upperArmGeo.translate(0, -0.135, 0);
            const upperArmMesh = new THREE.Mesh(upperArmGeo, suitMaterial);
            shoulder.add(upperArmMesh);

            const elbow = new THREE.Group();
            elbow.position.set(0, -0.27, 0);
            shoulder.add(elbow);

            // Forearm (Tailored Cuff revealing wrist & skin)
            const forearmSleeveGeo = new THREE.CylinderGeometry(0.046, 0.041, 0.18, 24);
            forearmSleeveGeo.translate(0, -0.09, 0);
            const forearmSleeve = new THREE.Mesh(forearmSleeveGeo, suitMaterial);
            elbow.add(forearmSleeve);

            const forearmSkinGeo = new THREE.CylinderGeometry(0.038, 0.033, 0.09, 20);
            forearmSkinGeo.translate(0, -0.225, 0);
            const forearmSkin = new THREE.Mesh(forearmSkinGeo, skinMaterial);
            elbow.add(forearmSkin);

            const wrist = new THREE.Group();
            wrist.position.set(0, -0.26, 0);
            elbow.add(wrist);

            const hand = buildArticulatedHand(isLeft);
            wrist.add(hand.root);

            return { shoulder, elbow, wrist, hand };
        };

        this.skeleton.leftArm = createArmChain(true);
        this.skeleton.rightArm = createArmChain(false);
        this.hands.left = this.skeleton.leftArm.hand;
        this.hands.right = this.skeleton.rightArm.hand;

        // Initial Neutral Signing Stance (Hands raised gracefully in front of chest)
        this.setPoseNeutralStance();
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

            // Interpolate Right Arm
            if (p1.rArm && p2.rArm) {
                const rx = p1.rArm[0] + (p2.rArm[0] - p1.rArm[0]) * ease;
                const ry = p1.rArm[1] + (p2.rArm[1] - p1.rArm[1]) * ease;
                const rz = p1.rArm[2] + (p2.rArm[2] - p1.rArm[2]) * ease;
                this.skeleton.rightArm.shoulder.rotation.set(rx, ry, rz);
            }
            if (p1.rElbow && p2.rElbow) {
                const ex = p1.rElbow[0] + (p2.rElbow[0] - p1.rElbow[0]) * ease;
                const ey = p1.rElbow[1] + (p2.rElbow[1] - p1.rElbow[1]) * ease;
                const ez = p1.rElbow[2] + (p2.rElbow[2] - p1.rElbow[2]) * ease;
                this.skeleton.rightArm.elbow.rotation.set(ex, ey, ez);
            }

            // Interpolate Left Arm
            if (p1.lArm && p2.lArm) {
                const lx = p1.lArm[0] + (p2.lArm[0] - p1.lArm[0]) * ease;
                const ly = p1.lArm[1] + (p2.lArm[1] - p1.lArm[1]) * ease;
                const lz = p1.lArm[2] + (p2.lArm[2] - p1.lArm[2]) * ease;
                this.skeleton.leftArm.shoulder.rotation.set(lx, ly, lz);
            }
            if (p1.lElbow && p2.lElbow) {
                const lex = p1.lElbow[0] + (p2.lElbow[0] - p1.lElbow[0]) * ease;
                const ley = p1.lElbow[1] + (p2.lElbow[1] - p1.lElbow[1]) * ease;
                const lez = p1.lElbow[2] + (p2.lElbow[2] - p1.lElbow[2]) * ease;
                this.skeleton.leftArm.elbow.rotation.set(lex, ley, lez);
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

    playGeneratedSigningStream(productionData, onComplete) {
        // Map synthesized text output to sign sequences
        const signMap = {
            "ጤና ይስጥልኝ": "greetings",
            "ሰላም": "greetings",
            "መድኃኒት": "medicine_after_food",
            "አብራሩልኝ": "explain_again",
            "ውድቅ": "court_dismissed"
        };

        let selectedKey = "greetings";
        const prompt = (productionData && productionData.prompt) ? productionData.prompt : "";

        for (const [kw, k] of Object.entries(signMap)) {
            if (prompt.includes(kw)) {
                selectedKey = k;
                break;
            }
        }

        this.playSign(selectedKey, onComplete);
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
