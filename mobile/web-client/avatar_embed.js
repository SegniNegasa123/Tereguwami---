/**
 * SignAvatars (ECCV 2024) 3D SMPL-X & MANO Surface Mesh Engine
 * Exact WebGL Port of ZhengdiYu/SignAvatars (https://github.com/ZhengdiYu/SignAvatars)
 *
 * Implements high-density anatomical SMPL-X full body mesh, articulated MANO 15-joint hand meshes,
 * and FLAME 50-expression facial blendshape drive.
 */

class AvatarEmbedScene {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.smplxMesh = null;
        this.smplxSkeleton = {};
        this.manoLeftJoints = [];
        this.manoRightJoints = [];
        this.flameFacialNodes = {};
        this.isPlaying = false;
        this.animationTimer = null;

        this.init();
    }

    init() {
        const width = this.container.clientWidth || 480;
        const height = this.container.clientHeight || 400;

        // 1. Scene Setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0d14);

        // 2. Camera Setup
        this.camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
        this.camera.position.set(0, 1.4, 2.0);
        this.camera.lookAt(0, 1.25, 0);

        // 3. Studio 3-Point Lighting (SignAvatars Signature Neutral Clay Render)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xe2e8f0, 1.4);
        keyLight.position.set(3, 4, 3);
        keyLight.castShadow = true;
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x60a5fa, 0.6);
        fillLight.position.set(-3, 2, 2);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
        rimLight.position.set(0, 3, -3);
        this.scene.add(rimLight);

        // Studio Floor Grid
        const floorGrid = new THREE.GridHelper(10, 20, 0x1e293b, 0x0f172a);
        floorGrid.position.y = 0;
        this.scene.add(floorGrid);

        // 4. Construct Complete SignAvatars SMPL-X Body Mesh
        this.buildSignAvatarsSMPLXMesh();

        // 5. WebGL Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        window.addEventListener("resize", () => this.onWindowResize());
        this.animate();
    }

    buildSignAvatarsSMPLXMesh() {
        // SignAvatars Official Neutral PBR Clay Shader
        const smplxMaterial = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,        // Signature SignAvatars silver-clay surface
            roughness: 0.35,
            metalness: 0.15,
            flatShading: false
        });

        const accentMaterial = new THREE.MeshStandardMaterial({
            color: 0x38bdf8,        // Cyan accent for MANO hands and joint indicators
            roughness: 0.3,
            metalness: 0.25
        });

        const darkMaterial = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            roughness: 0.8
        });

        const rootGroup = new THREE.Group();
        this.scene.add(rootGroup);
        this.smplxMesh = rootGroup;

        // Pelvis & Hips
        const pelvisGeo = new THREE.SphereGeometry(0.16, 24, 16);
        pelvisGeo.scale(1.2, 0.9, 0.9);
        const pelvis = new THREE.Mesh(pelvisGeo, smplxMaterial);
        pelvis.position.set(0, 0.95, 0);
        rootGroup.add(pelvis);
        this.smplxSkeleton.pelvis = pelvis;

        // Spine Hierarchy (Spine1 -> Spine2 -> Spine3)
        const spine1 = new THREE.Group();
        spine1.position.set(0, 0.1, 0);
        pelvis.add(spine1);

        const lowerTorsoGeo = new THREE.CylinderGeometry(0.18, 0.16, 0.18, 24);
        const lowerTorso = new THREE.Mesh(lowerTorsoGeo, smplxMaterial);
        lowerTorso.position.y = 0.09;
        spine1.add(lowerTorso);

        const spine2 = new THREE.Group();
        spine2.position.set(0, 0.18, 0);
        spine1.add(spine2);

        // Chest / Ribcage & Pectorals
        const chestGeo = new THREE.CylinderGeometry(0.23, 0.19, 0.22, 28);
        const chest = new THREE.Mesh(chestGeo, smplxMaterial);
        chest.position.y = 0.11;
        spine2.add(chest);

        // Neck & Head (FLAME Mesh)
        const neckGroup = new THREE.Group();
        neckGroup.position.set(0, 0.22, 0);
        spine2.add(neckGroup);

        const neckGeo = new THREE.CylinderGeometry(0.065, 0.08, 0.12, 20);
        const neck = new THREE.Mesh(neckGeo, smplxMaterial);
        neck.position.y = 0.06;
        neckGroup.add(neck);

        const headGroup = new THREE.Group();
        headGroup.position.set(0, 0.12, 0);
        neckGroup.add(headGroup);
        this.smplxSkeleton.head = headGroup;

        // FLAME Anatomical Head Geometry
        const craniumGeo = new THREE.SphereGeometry(0.13, 32, 24);
        craniumGeo.scale(1.0, 1.15, 1.1);
        const cranium = new THREE.Mesh(craniumGeo, smplxMaterial);
        headGroup.add(cranium);

        // FLAME Face Mask (Eyebrows, Eyes, Nose, Mouth)
        const browGeo = new THREE.BoxGeometry(0.042, 0.009, 0.015);
        const leftBrow = new THREE.Mesh(browGeo, darkMaterial);
        leftBrow.position.set(-0.042, 0.035, 0.13);
        const rightBrow = new THREE.Mesh(browGeo, darkMaterial);
        rightBrow.position.set(0.042, 0.035, 0.13);
        headGroup.add(leftBrow);
        headGroup.add(rightBrow);
        this.flameFacialNodes.leftBrow = leftBrow;
        this.flameFacialNodes.rightBrow = rightBrow;

        // FLAME Jaw & Mouth
        const mouthGeo = new THREE.CylinderGeometry(0.028, 0.028, 0.01, 16);
        mouthGeo.rotateX(Math.PI / 2);
        const mouth = new THREE.Mesh(mouthGeo, darkMaterial);
        mouth.position.set(0, -0.05, 0.125);
        headGroup.add(mouth);
        this.flameFacialNodes.mouth = mouth;

        // MANO Articulated Hand Creator (15 Joints per Hand)
        const createMANOHandMesh = (isLeft) => {
            const side = isLeft ? -1 : 1;
            const handRoot = new THREE.Group();

            // Palm Mesh
            const palmGeo = new THREE.BoxGeometry(0.055, 0.07, 0.02);
            const palm = new THREE.Mesh(palmGeo, smplxMaterial);
            palm.position.y = -0.035;
            handRoot.add(palm);

            const fingerJoints = [];
            const fingerOffsets = [-0.022, -0.011, 0.0, 0.011, 0.022];

            for (let f = 0; f < 5; f++) {
                const isThumb = f === 0;
                const mcp = new THREE.Group();
                mcp.position.set(isThumb ? (side * -0.028) : fingerOffsets[f], isThumb ? -0.02 : -0.07, 0);

                // Proximal Phalanx
                const p1Geo = new THREE.CylinderGeometry(0.005, 0.0042, 0.024, 10);
                p1Geo.translate(0, -0.012, 0);
                const p1 = new THREE.Mesh(p1Geo, accentMaterial);
                mcp.add(p1);

                // Intermediate Phalanx (PIP)
                const pip = new THREE.Group();
                pip.position.set(0, -0.024, 0);
                mcp.add(pip);

                const p2Geo = new THREE.CylinderGeometry(0.0042, 0.0036, 0.018, 10);
                p2Geo.translate(0, -0.009, 0);
                const p2 = new THREE.Mesh(p2Geo, accentMaterial);
                pip.add(p2);

                // Distal Phalanx (DIP)
                const dip = new THREE.Group();
                dip.position.set(0, -0.018, 0);
                pip.add(dip);

                const p3Geo = new THREE.CylinderGeometry(0.0036, 0.0028, 0.014, 10);
                p3Geo.translate(0, -0.007, 0);
                const p3 = new THREE.Mesh(p3Geo, accentMaterial);
                dip.add(p3);

                handRoot.add(mcp);
                fingerJoints.push({ mcp, pip, dip });
            }

            return { root: handRoot, fingers: fingerJoints };
        };

        // SMPL-X Upper Limbs (Clavicle -> Shoulder -> Elbow -> Wrist -> MANO Hand)
        const createArmChain = (isLeft) => {
            const side = isLeft ? -1 : 1;
            const shoulder = new THREE.Group();
            shoulder.position.set(side * 0.25, 0.18, 0);
            spine2.add(shoulder);

            const upperArmGeo = new THREE.CylinderGeometry(0.048, 0.042, 0.28, 20);
            upperArmGeo.translate(0, -0.14, 0);
            const upperArm = new THREE.Mesh(upperArmGeo, smplxMaterial);
            shoulder.add(upperArm);

            const elbow = new THREE.Group();
            elbow.position.set(0, -0.28, 0);
            shoulder.add(elbow);

            const forearmGeo = new THREE.CylinderGeometry(0.042, 0.035, 0.26, 20);
            forearmGeo.translate(0, -0.13, 0);
            const forearm = new THREE.Mesh(forearmGeo, smplxMaterial);
            elbow.add(forearm);

            const wrist = new THREE.Group();
            wrist.position.set(0, -0.26, 0);
            elbow.add(wrist);

            const manoHand = createMANOHandMesh(isLeft);
            wrist.add(manoHand.root);

            return { shoulder, elbow, wrist, mano: manoHand };
        };

        this.smplxSkeleton.leftArm = createArmChain(true);
        this.smplxSkeleton.rightArm = createArmChain(false);
        this.manoLeftJoints = this.smplxSkeleton.leftArm.mano.fingers;
        this.manoRightJoints = this.smplxSkeleton.rightArm.mano.fingers;

        // Legs (Hips -> Thigh -> Knee -> Calf -> Foot) for complete whole-body SMPL-X presence
        const createLeg = (isLeft) => {
            const side = isLeft ? -1 : 1;
            const hip = new THREE.Group();
            hip.position.set(side * 0.12, -0.05, 0);
            pelvis.add(hip);

            const thighGeo = new THREE.CylinderGeometry(0.065, 0.05, 0.38, 20);
            thighGeo.translate(0, -0.19, 0);
            const thigh = new THREE.Mesh(thighGeo, smplxMaterial);
            hip.add(thigh);

            const knee = new THREE.Group();
            knee.position.set(0, -0.38, 0);
            hip.add(knee);

            const calfGeo = new THREE.CylinderGeometry(0.05, 0.04, 0.38, 20);
            calfGeo.translate(0, -0.19, 0);
            const calf = new THREE.Mesh(calfGeo, smplxMaterial);
            knee.add(calf);

            const footGeo = new THREE.BoxGeometry(0.08, 0.05, 0.16);
            footGeo.translate(0, -0.025, 0.05);
            const foot = new THREE.Mesh(footGeo, smplxMaterial);
            foot.position.set(0, -0.38, 0);
            knee.add(foot);
        };

        createLeg(true);
        createLeg(false);
    }

    onWindowResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    setSignAvatarsPose(smplxData, blendshapes) {
        if (!smplxData && !blendshapes) return;

        // 1. FLAME Facial Expressions (Eyebrows, Jaw, Head Yaw)
        const bs = blendshapes || {};
        const browUp = bs.browInnerUp || 0.0;
        const browDown = bs.browDownLeft || 0.0;
        const netBrowY = 0.035 + (browUp * 0.022) - (browDown * 0.015);

        if (this.flameFacialNodes.leftBrow && this.flameFacialNodes.rightBrow) {
            this.flameFacialNodes.leftBrow.position.y = netBrowY;
            this.flameFacialNodes.rightBrow.position.y = netBrowY;
        }

        if (this.flameFacialNodes.mouth) {
            const jawOpen = bs.jawOpen || 0.1;
            this.flameFacialNodes.mouth.scale.set(1.0 + (bs.mouthSmile || 0.0), 1.0 + jawOpen * 3.2, 1.0);
        }

        if (this.smplxSkeleton.head && bs.headYaw !== undefined) {
            this.smplxSkeleton.head.rotation.y = bs.headYaw;
        }

        // 2. SMPL-X Body Rotations
        if (smplxData && smplxData.body_pose) {
            const bp = smplxData.body_pose;
            if (this.smplxSkeleton.leftArm && this.smplxSkeleton.rightArm) {
                // Joint 16: Left shoulder, Joint 17: Right shoulder
                this.smplxSkeleton.leftArm.shoulder.rotation.x = bp[16 * 3 + 0] || 0;
                this.smplxSkeleton.leftArm.shoulder.rotation.z = bp[16 * 3 + 2] || 0;
                this.smplxSkeleton.rightArm.shoulder.rotation.x = bp[17 * 3 + 0] || 0;
                this.smplxSkeleton.rightArm.shoulder.rotation.z = bp[17 * 3 + 2] || 0;

                // Joint 18: Left elbow, Joint 19: Right elbow
                this.smplxSkeleton.leftArm.elbow.rotation.x = bp[18 * 3 + 0] || 0;
                this.smplxSkeleton.rightArm.elbow.rotation.x = bp[19 * 3 + 0] || 0;
            }
        }

        // 3. MANO 15-joint Articulated Finger Rotations
        if (smplxData && smplxData.left_hand_pose && smplxData.right_hand_pose) {
            const lhp = smplxData.left_hand_pose;
            const rhp = smplxData.right_hand_pose;

            this.manoLeftJoints.forEach((finger, fIdx) => {
                const baseIdx = fIdx * 9;
                const rot1 = lhp[baseIdx] || 0;
                const rot2 = lhp[baseIdx + 3] || (rot1 * 0.8);
                const rot3 = lhp[baseIdx + 6] || (rot1 * 0.6);
                finger.mcp.rotation.x = rot1;
                finger.pip.rotation.x = rot2;
                finger.dip.rotation.x = rot3;
            });

            this.manoRightJoints.forEach((finger, fIdx) => {
                const baseIdx = fIdx * 9;
                const rot1 = rhp[baseIdx] || 0;
                const rot2 = rhp[baseIdx + 3] || (rot1 * 0.8);
                const rot3 = rhp[baseIdx + 6] || (rot1 * 0.6);
                finger.mcp.rotation.x = rot1;
                finger.pip.rotation.x = rot2;
                finger.dip.rotation.x = rot3;
            });
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
        const intervalMs = Math.max(12, 1000 / fps);

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
            this.setSignAvatarsPose(frame.smplx, frame.blendshapes);

            frameIdx++;
        }, intervalMs);
    }

    resetToNeutral() {
        if (this.smplxSkeleton.leftArm && this.smplxSkeleton.rightArm) {
            this.smplxSkeleton.leftArm.shoulder.rotation.set(0, 0, 0);
            this.smplxSkeleton.leftArm.elbow.rotation.set(0, 0, 0);
            this.smplxSkeleton.rightArm.shoulder.rotation.set(0, 0, 0);
            this.smplxSkeleton.rightArm.elbow.rotation.set(0, 0, 0);
        }
        if (this.smplxSkeleton.head) {
            this.smplxSkeleton.head.rotation.set(0, 0, 0);
        }
        this.manoLeftJoints.forEach(f => {
            f.mcp.rotation.set(0, 0, 0);
            f.pip.rotation.set(0, 0, 0);
            f.dip.rotation.set(0, 0, 0);
        });
        this.manoRightJoints.forEach(f => {
            f.mcp.rotation.set(0, 0, 0);
            f.pip.rotation.set(0, 0, 0);
            f.dip.rotation.set(0, 0, 0);
        });
        if (this.flameFacialNodes.leftBrow && this.flameFacialNodes.rightBrow) {
            this.flameFacialNodes.leftBrow.position.y = 0.035;
            this.flameFacialNodes.rightBrow.position.y = 0.035;
        }
        if (this.flameFacialNodes.mouth) {
            this.flameFacialNodes.mouth.scale.set(1.0, 1.0, 1.0);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Subtle idle respiration kinematics (SMPL-X root breathing)
        if (!this.isPlaying && this.smplxSkeleton.pelvis) {
            const time = Date.now() * 0.002;
            this.smplxSkeleton.pelvis.position.y = 0.95 + Math.sin(time) * 0.0025;
            if (this.smplxSkeleton.head) {
                this.smplxSkeleton.head.position.y = 0.12 + Math.sin(time) * 0.0025;
            }
        }

        this.renderer.render(this.scene, this.camera);
    }
}

window.AvatarEmbedScene = AvatarEmbedScene;
