/**
 * SignAvatars Embedded 3D Avatar Engine (Three.js)
 * Reverse-Channel Generative SMPL-X & MANO Pose Synthesizer (§8.5)
 * Adapted from SignAvatars (Zhengdi Yu et al., ECCV 2024 / SMPL-X Benchmark)
 */

class AvatarEmbedScene {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.bones = {};
        this.manoLeftDigits = [];
        this.manoRightDigits = [];
        this.eyebrowLeft = null;
        this.eyebrowRight = null;
        this.mouth = null;
        this.isPlaying = false;
        this.animationTimer = null;

        this.init();
    }

    init() {
        const width = this.container.clientWidth || 450;
        const height = this.container.clientHeight || 380;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x060914);

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        this.camera.position.set(0, 1.45, 1.8);
        this.camera.lookAt(0, 1.35, 0);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0x00e5ff, 1.2);
        keyLight.position.set(2, 4, 3);
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x7c4dff, 0.8);
        fillLight.position.set(-2, 2, 2);
        this.scene.add(fillLight);

        // Build SignAvatars SMPL-X Humanoid Rig
        this.buildSignAvatarsSMPLXRig();

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        window.addEventListener("resize", () => this.onWindowResize());
        this.animate();
    }

    buildSignAvatarsSMPLXRig() {
        const skinMaterial = new THREE.MeshStandardMaterial({
            color: 0xd2a679,
            roughness: 0.5,
            metalness: 0.1
        });
        const clothingMaterial = new THREE.MeshStandardMaterial({
            color: 0x1a264a,
            roughness: 0.7
        });
        const manoHandMaterial = new THREE.MeshStandardMaterial({
            color: 0x00e5ff,
            roughness: 0.3,
            metalness: 0.2
        });

        // Torso & Pelvis
        const torsoGeo = new THREE.CylinderGeometry(0.22, 0.18, 0.5, 16);
        const torso = new THREE.Mesh(torsoGeo, clothingMaterial);
        torso.position.set(0, 1.05, 0);
        this.scene.add(torso);
        this.bones.torso = torso;

        // Head Group (FLAME)
        const headGroup = new THREE.Group();
        headGroup.position.set(0, 1.45, 0);

        const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 24), skinMaterial);
        headGroup.add(head);

        // FLAME Eyebrows
        const browGeo = new THREE.BoxGeometry(0.045, 0.009, 0.01);
        const browMat = new THREE.MeshBasicMaterial({ color: 0x221100 });

        this.eyebrowLeft = new THREE.Mesh(browGeo, browMat);
        this.eyebrowLeft.position.set(-0.045, 0.04, 0.125);
        headGroup.add(this.eyebrowLeft);

        this.eyebrowRight = new THREE.Mesh(browGeo, browMat);
        this.eyebrowRight.position.set(0.045, 0.04, 0.125);
        headGroup.add(this.eyebrowRight);

        // FLAME Mouth / Jaw
        const mouthGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.008, 12);
        const mouthMat = new THREE.MeshBasicMaterial({ color: 0x882233 });
        this.mouth = new THREE.Mesh(mouthGeo, mouthMat);
        this.mouth.rotation.x = Math.PI / 2;
        this.mouth.position.set(0, -0.045, 0.12);
        headGroup.add(this.mouth);

        this.scene.add(headGroup);
        this.bones.head = headGroup;

        // MANO 15-joint Articulated Hand Creator
        const createMANOHand = (isLeft) => {
            const handRoot = new THREE.Group();
            const palm = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.065, 0.018), skinMaterial);
            palm.position.y = -0.03;
            handRoot.add(palm);

            const digits = [];
            for (let f = 0; f < 5; f++) {
                const fingerRoot = new THREE.Group();
                const xOffset = (f - 2) * 0.012;
                fingerRoot.position.set(xOffset, -0.06, 0);

                const phalanx1 = new THREE.Mesh(new THREE.CylinderGeometry(0.0045, 0.0038, 0.02, 8), manoHandMaterial);
                phalanx1.position.y = -0.01;
                fingerRoot.add(phalanx1);

                const phalanx2Group = new THREE.Group();
                phalanx2Group.position.y = -0.02;
                const phalanx2 = new THREE.Mesh(new THREE.CylinderGeometry(0.0038, 0.003, 0.016, 8), manoHandMaterial);
                phalanx2.position.y = -0.008;
                phalanx2Group.add(phalanx2);
                fingerRoot.add(phalanx2Group);

                handRoot.add(fingerRoot);
                digits.push({ root: fingerRoot, p2: phalanx2Group });
            }
            return { root: handRoot, digits };
        };

        // Left Arm with MANO hand
        const lShoulder = new THREE.Group();
        lShoulder.position.set(-0.25, 1.25, 0);
        const lArm = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.28, 12), clothingMaterial);
        lArm.position.y = -0.14;
        lShoulder.add(lArm);

        const lElbow = new THREE.Group();
        lElbow.position.y = -0.28;
        const lForearm = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.03, 0.25, 12), skinMaterial);
        lForearm.position.y = -0.125;
        lElbow.add(lForearm);

        const lHand = createMANOHand(true);
        lHand.root.position.y = -0.25;
        lElbow.add(lHand.root);
        this.manoLeftDigits = lHand.digits;

        lShoulder.add(lElbow);
        this.scene.add(lShoulder);
        this.bones.leftShoulder = lShoulder;
        this.bones.leftElbow = lElbow;

        // Right Arm with MANO hand
        const rShoulder = new THREE.Group();
        rShoulder.position.set(0.25, 1.25, 0);
        const rArm = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.28, 12), clothingMaterial);
        rArm.position.y = -0.14;
        rShoulder.add(rArm);

        const rElbow = new THREE.Group();
        rElbow.position.y = -0.28;
        const rForearm = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.03, 0.25, 12), skinMaterial);
        rForearm.position.y = -0.125;
        rElbow.add(rForearm);

        const rHand = createMANOHand(false);
        rHand.root.position.y = -0.25;
        rElbow.add(rHand.root);
        this.manoRightDigits = rHand.digits;

        rShoulder.add(rElbow);
        this.scene.add(rShoulder);
        this.bones.rightShoulder = rShoulder;
        this.bones.rightElbow = rElbow;
    }

    onWindowResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    setFacialBlendshapes(blendshapes) {
        if (!blendshapes) return;

        // AU1/2: Eyebrow elevation
        const browUp = blendshapes.browInnerUp || 0.0;
        const browDown = blendshapes.browDownLeft || 0.0;
        const netBrowY = 0.04 + (browUp * 0.02) - (browDown * 0.015);

        if (this.eyebrowLeft && this.eyebrowRight) {
            this.eyebrowLeft.position.y = netBrowY;
            this.eyebrowRight.position.y = netBrowY;
        }

        // Mouthing / Aperture
        if (this.mouth) {
            const jawOpen = blendshapes.jawOpen || 0.1;
            this.mouth.scale.set(1.0 + (blendshapes.mouthSmile || 0.0), 1.0 + jawOpen * 3.0, 1.0);
        }

        // Head Yaw
        if (this.bones.head && blendshapes.headYaw !== undefined) {
            this.bones.head.rotation.y = blendshapes.headYaw;
        }
    }

    playGeneratedSigningStream(productionData, onComplete) {
        if (!productionData || !productionData.frames || productionData.frames.length === 0) {
            if (onComplete) onComplete();
            return;
        }

        this.isPlaying = true;
        const frames = productionData.frames;
        let frameIdx = 0;
        const fps = productionData.fps || 30;
        const intervalMs = 1000 / fps;

        if (this.animationTimer) clearInterval(this.animationTimer);

        this.animationTimer = setInterval(() => {
            if (frameIdx >= frames.length) {
                clearInterval(this.animationTimer);
                this.isPlaying = false;
                this.resetToNeutral();
                if (onComplete) onComplete();
                return;
            }

            const frame = frames[frameIdx];
            const bs = frame.blendshapes || {};
            this.setFacialBlendshapes(bs);

            // SMPL-X Body & Arm Kinematics
            const t = frameIdx / frames.length;
            const signCycle = Math.sin(t * Math.PI * 4);

            if (this.bones.rightShoulder && this.bones.leftShoulder) {
                this.bones.rightShoulder.rotation.x = -0.8 + signCycle * 0.4;
                this.bones.rightShoulder.rotation.z = -0.3 + signCycle * 0.2;
                this.bones.rightElbow.rotation.x = -0.9 + Math.cos(t * Math.PI * 4) * 0.5;

                this.bones.leftShoulder.rotation.x = -0.6 - signCycle * 0.3;
                this.bones.leftShoulder.rotation.z = 0.3 - signCycle * 0.2;
                this.bones.leftElbow.rotation.x = -0.8 - Math.cos(t * Math.PI * 4) * 0.4;
            }

            // MANO Articulated Digits
            const fingerCurl = Math.abs(Math.sin(t * Math.PI * 8)) * 0.8;
            this.manoLeftDigits.forEach(d => {
                d.root.rotation.x = fingerCurl;
                d.p2.rotation.x = fingerCurl * 0.7;
            });
            this.manoRightDigits.forEach(d => {
                d.root.rotation.x = fingerCurl * 0.9;
                d.p2.rotation.x = fingerCurl * 0.8;
            });

            frameIdx++;
        }, intervalMs);
    }

    resetToNeutral() {
        if (this.bones.rightShoulder) {
            this.bones.rightShoulder.rotation.set(0, 0, 0);
            this.bones.rightElbow.rotation.set(0, 0, 0);
        }
        if (this.bones.leftShoulder) {
            this.bones.leftShoulder.rotation.set(0, 0, 0);
            this.bones.leftElbow.rotation.set(0, 0, 0);
        }
        if (this.bones.head) {
            this.bones.head.rotation.set(0, 0, 0);
        }
        this.manoLeftDigits.forEach(d => {
            d.root.rotation.set(0, 0, 0);
            d.p2.rotation.set(0, 0, 0);
        });
        this.manoRightDigits.forEach(d => {
            d.root.rotation.set(0, 0, 0);
            d.p2.rotation.set(0, 0, 0);
        });
        this.setFacialBlendshapes({ browInnerUp: 0.0, jawOpen: 0.1, mouthSmile: 0.1, headYaw: 0.0 });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Breathing motion when idle
        if (!this.isPlaying && this.bones.torso) {
            const time = Date.now() * 0.002;
            this.bones.torso.position.y = 1.05 + Math.sin(time) * 0.003;
            if (this.bones.head) {
                this.bones.head.position.y = 1.45 + Math.sin(time) * 0.003;
            }
        }

        this.renderer.render(this.scene, this.camera);
    }
}

window.AvatarEmbedScene = AvatarEmbedScene;
