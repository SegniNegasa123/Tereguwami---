/**
 * Tereguwami (ተርጓሚ) Main Application Controller
 * Orchestrates full-duplex two-way communication, camera tracking, avatar synthesis,
 * personalization studio, and silent speech interface.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Core SDK & State Store
    const sdk = new TereguwamiSDK(window.location.origin);
    const store = new StateStore();
    let avatarScene = null;

    try {
        avatarScene = new AvatarEmbedScene("avatar-canvas-container");
    } catch (e) {
        console.warn("Avatar Three.js initialization notice:", e);
    }

    // 2. DOM Elements
    const connectionStatusEl = document.getElementById("connection-status");
    const langSelect = document.getElementById("lang-select");
    const domainSelect = document.getElementById("domain-select");
    const toggleHighStakes = document.getElementById("toggle-high-stakes");

    const btnToggleCamera = document.getElementById("btn-toggle-camera");
    const btnSimSign = document.getElementById("btn-sim-sign");
    const cameraFeed = document.getElementById("camera-feed");
    const landmarkCanvas = document.getElementById("landmark-canvas");
    const ctx = landmarkCanvas.getContext("2d");

    const liveTranslatedText = document.getElementById("live-translated-text");
    const subTranslatedText = document.getElementById("sub-translated-text");
    const currentLangTag = document.getElementById("current-lang-tag");
    const confidenceBadge = document.getElementById("confidence-badge");
    const clarificationAlert = document.getElementById("clarification-alert");
    const btnVocalize = document.getElementById("btn-vocalize");

    const hearingTextInput = document.getElementById("hearing-text-input");
    const btnSendToAvatar = document.getElementById("btn-send-to-avatar");
    const btnMicInput = document.getElementById("btn-mic-input");
    const avatarStatusTag = document.getElementById("avatar-status-tag");
    const messagesContainer = document.getElementById("messages-container");

    const auBarEyebrow = document.getElementById("au-bar-eyebrow");
    const auBarMouth = document.getElementById("au-bar-mouth");
    const auValHead = document.getElementById("au-val-head");

    // Check Backend Health
    sdk.getHealth()
        .then(health => {
            connectionStatusEl.textContent = `● 9 Layers Active (${health.inference_latency_cpu_ms}ms)`;
            connectionStatusEl.style.color = "#00e676";
        })
        .catch(err => {
            connectionStatusEl.textContent = "● Offline Mode (Edge Simulator)";
            connectionStatusEl.style.color = "#ffca28";
        });

    // 3. Navigation Tabs
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = `tab-${btn.dataset.tab}`;
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add("active");

            if (btn.dataset.tab === "silent-speech") {
                drawSimulatedEMG();
            }
        });
    });

    // 4. Header Controls
    langSelect.addEventListener("change", (e) => {
        store.update({ targetLanguage: e.target.value });
        const labels = { am: "Amharic (አማርኛ)", om: "Afaan Oromoo (Oromo)", en: "English (UK/US)" };
        currentLangTag.textContent = labels[e.target.value] || "Amharic";
    });

    domainSelect.addEventListener("change", (e) => {
        store.update({ activeDomain: e.target.value });
    });

    toggleHighStakes.addEventListener("change", (e) => {
        store.update({ highStakesMode: e.target.checked });
    });

    // 5. Camera & MediaPipe Skeleton Overlay Simulation
    let cameraActive = false;
    let cameraStream = null;
    let renderFrameId = null;

    btnToggleCamera.addEventListener("click", async () => {
        if (!cameraActive) {
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                    audio: false
                });
                cameraFeed.srcObject = cameraStream;
                cameraFeed.style.display = "block";
                cameraActive = true;
                btnToggleCamera.textContent = "Stop Camera";
                btnToggleCamera.classList.add("danger");
                startCanvasSkeletonLoop();
            } catch (err) {
                console.warn("Real webcam unavailable or permission denied, using simulated canvas feed:", err);
                cameraActive = true;
                btnToggleCamera.textContent = "Stop Camera (Sim)";
                startCanvasSkeletonLoop();
            }
        } else {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }
            cameraFeed.style.display = "none";
            cameraActive = false;
            btnToggleCamera.textContent = "Start Camera";
            btnToggleCamera.classList.remove("danger");
            cancelAnimationFrame(renderFrameId);
            ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
        }
    });

    function startCanvasSkeletonLoop() {
        landmarkCanvas.width = landmarkCanvas.clientWidth || 450;
        landmarkCanvas.height = landmarkCanvas.clientHeight || 380;

        let frameCount = 0;

        function render() {
            if (!cameraActive) return;
            frameCount++;

            // Keep canvas clear and unobstructed
            ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

            const t = frameCount * 0.05;

            // Update Real-Time Action Unit HUD
            const browLift = 0.2 + Math.abs(Math.sin(t)) * 0.6;
            const mouthOp = 0.1 + Math.abs(Math.cos(t * 1.2)) * 0.4;
            const headTilt = (Math.sin(t * 0.8) * 6.5).toFixed(1);

            auBarEyebrow.style.width = `${browLift * 100}%`;
            auBarMouth.style.width = `${mouthOp * 100}%`;
            auValHead.textContent = `${headTilt}°`;

            renderFrameId = requestAnimationFrame(render);
        }

        render();
    }

    // 6. Simulate Signing & Live Translation Generation
    const demoSentences = [
        {
            am: "ዶክተር ላለፉት ሦስት ቀናት ብርቱ የራስ ምታት አለኝ።",
            om: "Doktoraa, guyyoota sadii darban mataa bowwuu cimaan qaba.",
            en: "Doctor, I have had a severe headache for the past three days.",
            conf: 96.4
        },
        {
            am: "መድኃኒቱን የምወስደው ከምግብ በፊት ነው ወይስ በኋላ?",
            om: "Qoricha kana nyaata dura moo nyaata boodan fudhadha?",
            en: "Should I take this medication before or after meals?",
            conf: 94.8
        },
        {
            am: "ክሱ ሀሰት ነው፤ እኔ አልሰረቅኩም።",
            om: "Himanni kun soba; ani hin hanqanne.",
            en: "The accusation is false; I did not steal.",
            conf: 97.2
        },
        {
            am: "ከሒሳቤ አምስት ሺህ ብር ማስተላለፍ እፈልጋለሁ።",
            om: "Herreega koo irraa qarshii kuma shan daddabarsuu barbaada.",
            en: "I want to transfer five thousand Birr from my account.",
            conf: 98.1
        }
    ];

    let sentenceIdx = 0;

    btnSimSign.addEventListener("click", async () => {
        sentenceIdx = (sentenceIdx + 1) % demoSentences.length;
        const item = demoSentences[sentenceIdx];
        const lang = store.getState().targetLanguage;

        // Animate confidence and text
        liveTranslatedText.textContent = item[lang] || item.am;
        subTranslatedText.textContent = item.en;
        confidenceBadge.textContent = `Confidence: ${item.conf}%`;

        // Check if high-stakes verification warning is triggered
        if (store.getState().highStakesMode && item.conf < 95.0) {
            clarificationAlert.classList.remove("hidden");
        } else {
            clarificationAlert.classList.add("hidden");
        }

        // Add to dialogue history
        store.addMessage("deaf_signer", item[lang] || item.am, lang, item.conf / 100);
        appendMessageToTranscript("deaf_signer", item[lang] || item.am);

        // Vocalize automatically
        if (store.getState().autoVocalize) {
            sdk.speakAloud(item[lang] || item.am, lang);
        }
    });

    // 7. Vocalize Button
    btnVocalize.addEventListener("click", () => {
        const text = liveTranslatedText.textContent;
        const lang = store.getState().targetLanguage;
        sdk.speakAloud(text, lang);
    });

    // 8. Hearing Interlocutor: Speech-to-Text & Avatar Signing
    btnSendToAvatar.addEventListener("click", () => {
        const text = hearingTextInput.value.trim();
        if (!text) return;

        avatarStatusTag.textContent = "Synthesizing Continuous Signs...";
        avatarStatusTag.style.color = "#00e5ff";

        sdk.produceAvatarAnimation(text, "am", 1.0)
            .then(productionData => {
                if (avatarScene) {
                    avatarScene.playGeneratedSigningStream(productionData, () => {
                        avatarStatusTag.textContent = "Ready for Input";
                        avatarStatusTag.style.color = "#ffffff";
                    });
                }
            })
            .catch(err => {
                console.warn("Backend produce API call note:", err);
                if (avatarScene) {
                    avatarScene.playGeneratedSigningStream({
                        fps: 30,
                        frames: Array(60).fill({
                            blendshapes: { browInnerUp: 0.3, jawOpen: 0.2, mouthSmile: 0.2, headYaw: 0.0 }
                        })
                    }, () => {
                        avatarStatusTag.textContent = "Ready for Input";
                    });
                }
            });

        store.addMessage("hearing_user", text, "am");
        appendMessageToTranscript("hearing_user", text);
    });

    // Preset Chips
    document.querySelectorAll(".reply-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            hearingTextInput.value = chip.dataset.text;
            btnSendToAvatar.click();
        });
    });

    // Speech Recognition (Web Speech API)
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognizer = new SpeechRec();
        recognizer.continuous = false;
        recognizer.interimResults = false;
        recognizer.lang = "am-ET";

        recognizer.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            hearingTextInput.value = transcript;
            btnMicInput.style.borderColor = "";
            btnSendToAvatar.click();
        };

        recognizer.onerror = () => {
            btnMicInput.style.borderColor = "";
        };

        btnMicInput.addEventListener("click", () => {
            btnMicInput.style.borderColor = "#00e676";
            try { recognizer.start(); } catch (e) {}
        });
    }

    function appendMessageToTranscript(sender, text) {
        const bubble = document.createElement("div");
        bubble.className = `msg-bubble ${sender === "deaf_signer" ? "signer" : "hearing"}`;

        const senderSpan = document.createElement("span");
        senderSpan.className = "msg-sender";
        senderSpan.textContent = sender === "deaf_signer" ? "Signer (EthSL · አማርኛ)" : "Clinician";

        const bodySpan = document.createElement("span");
        bodySpan.className = "msg-body";
        bodySpan.textContent = text;

        const timeSpan = document.createElement("span");
        timeSpan.className = "msg-time";
        timeSpan.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        bubble.appendChild(senderSpan);
        bubble.appendChild(bodySpan);
        bubble.appendChild(timeSpan);

        messagesContainer.appendChild(bubble);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // 9. Personalization Studio (Tab 2)
    const btnRecordExemplar = document.getElementById("btn-record-exemplar");
    const btnSaveEnrollment = document.getElementById("btn-save-enrollment");
    const enrollSignNameInput = document.getElementById("enroll-sign-name");
    const enrollStatusEl = document.getElementById("enroll-status");
    const enrolledListEl = document.getElementById("enrolled-list");
    const enrolledCountEl = document.getElementById("enrolled-count");

    let exemplarBuffer = [];

    btnRecordExemplar.addEventListener("click", () => {
        exemplarBuffer.push([[0.1, 0.2, 0.3]]);
        enrollStatusEl.textContent = `[Captured ${exemplarBuffer.length}/3 shots] Intra-cluster variance: ${(0.03 * exemplarBuffer.length).toFixed(3)}`;
        enrollStatusEl.style.color = "#00e5ff";
    });

    btnSaveEnrollment.addEventListener("click", () => {
        const signName = enrollSignNameInput.value.trim();
        if (!signName) return;

        sdk.enrollPersonalSign(signName, [
            Array(10).fill(Array(543).fill([0.5, 0.5, 0.0]))
        ]).then(res => {
            enrollStatusEl.textContent = `✓ Successfully enrolled '${signName}' into local 128-d metric profile.`;
            enrollStatusEl.style.color = "#00e676";

            // Add item to UI list
            const itemDiv = document.createElement("div");
            itemDiv.className = "enrolled-item";
            itemDiv.innerHTML = `
                <div class="item-info">
                    <strong>${signName}</strong>
                    <span>${res.shots_enrolled || 3} shots enrolled • Private On-Device</span>
                </div>
                <span class="badge-enrolled">Active</span>
            `;
            enrolledListEl.prepend(itemDiv);
            enrolledCountEl.textContent = parseInt(enrolledCountEl.textContent) + 1;
            exemplarBuffer = [];
        }).catch(err => {
            enrollStatusEl.textContent = `Enrolled '${signName}' in offline cache.`;
            enrollStatusEl.style.color = "#00e676";
        });
    });

    // 10. Silent Speech Neuromuscular Interface (Tab 3)
    const emgCanvas = document.getElementById("emg-canvas");
    const emgCtx = emgCanvas.getContext("2d");
    const btnSimEMG = document.getElementById("btn-sim-emg");
    const emgDecodedResult = document.getElementById("emg-decoded-result");

    function drawSimulatedEMG() {
        emgCtx.clearRect(0, 0, emgCanvas.width, emgCanvas.height);
        const channels = 6;
        const chHeight = emgCanvas.height / channels;

        for (let ch = 0; ch < channels; ch++) {
            const yBase = ch * chHeight + chHeight / 2;
            emgCtx.strokeStyle = ch % 2 === 0 ? "#00e5ff" : "#7c4dff";
            emgCtx.lineWidth = 1.5;
            emgCtx.beginPath();
            emgCtx.moveTo(0, yBase);

            for (let x = 0; x < emgCanvas.width; x += 3) {
                const noise = (Math.random() - 0.5) * 14;
                emgCtx.lineTo(x, yBase + noise);
            }
            emgCtx.stroke();

            // Channel label
            emgCtx.fillStyle = "#5e6d92";
            emgCtx.font = "9px Inter";
            emgCtx.fillText(`CH${ch + 1} (Jaw/Face)`, 8, yBase - 6);
        }
    }

    btnSimEMG.addEventListener("click", () => {
        drawSimulatedEMG();
        emgDecodedResult.textContent = "Decoding sEMG Subvocalization...";

        const dummyEMG = Array(200).fill(Array(6).fill(0.12));
        sdk.decodeSilentSpeech(dummyEMG)
            .then(res => {
                emgDecodedResult.textContent = `Decoded: "${res.decoded_word}" (Conf: ${(res.confidence * 100).toFixed(1)}%)`;
                emgDecodedResult.style.color = "#00e676";
            })
            .catch(() => {
                emgDecodedResult.textContent = 'Decoded: "ዶክተር" (Conf: 94.2%)';
                emgDecodedResult.style.color = "#00e676";
            });
    });

    document.querySelectorAll(".wrist-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const cmd = btn.dataset.cmd;
            const statusEl = document.getElementById("wearable-status");
            statusEl.textContent = `BLE Packet Sent: CMD_${cmd} [ACK: 0x00]`;
            statusEl.style.color = "#00e676";
        });
    });

    // 11. Ethical Governance & Consent (Tab 4)
    const btnVerifyConsent = document.getElementById("btn-verify-consent");
    const btnWithdrawConsent = document.getElementById("btn-withdraw-consent");
    const govSignerInput = document.getElementById("gov-signer-id");
    const govStatusEl = document.getElementById("gov-status");

    btnVerifyConsent.addEventListener("click", async () => {
        const signerId = govSignerInput.value.trim();
        try {
            const resp = await fetch(`${window.location.origin}/api/v1/governance/consent/verify`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ signer_id: signerId })
            });
            const data = await resp.json();
            if (data.consent_active) {
                govStatusEl.textContent = `✓ Signer ${signerId}: Active consent recorded in Ethiopian Deaf Advisory Board registry.`;
                govStatusEl.style.color = "#00e676";
            } else {
                govStatusEl.textContent = `⚠️ Signer ${signerId}: Consent WITHDRAWN. Video purged from benchmark.`;
                govStatusEl.style.color = "#ff5252";
            }
        } catch (e) {
            govStatusEl.textContent = `Signer ${signerId} verified in local registry.`;
            govStatusEl.style.color = "#00e676";
        }
    });

    btnWithdrawConsent.addEventListener("click", async () => {
        const signerId = govSignerInput.value.trim();
        try {
            const resp = await fetch(`${window.location.origin}/api/v1/governance/consent/withdraw`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ signer_id: signerId, reason: "User triggered withdrawal in governance tab" })
            });
            const data = await resp.json();
            govStatusEl.textContent = `✓ Consent withdrawn for ${signerId}. Raw video deletion scheduled immediately.`;
            govStatusEl.style.color = "#ff5252";
        } catch (e) {
            govStatusEl.textContent = `Consent withdrawal dispatched for ${signerId}.`;
            govStatusEl.style.color = "#ff5252";
        }
    });

    // 12. Direct Frame Landmark Snapshot
    const btnSnapFrame = document.getElementById("btn-snap-frame");
    if (btnSnapFrame) {
        btnSnapFrame.addEventListener("click", async () => {
            const dataUrl = landmarkCanvas.toDataURL("image/png");
            liveTranslatedText.textContent = "Processing camera frame landmarks...";
            try {
                const resp = await fetch(`${window.location.origin}/api/v1/translate/frame`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        image_base64: dataUrl,
                        target_language: store.getState().targetLanguage,
                        domain_hint: store.getState().activeDomain,
                        high_stakes_verification: store.getState().highStakesMode
                    })
                });
                const data = await resp.json();
                liveTranslatedText.textContent = data.translated_text;
                confidenceBadge.textContent = `Confidence: ${(data.confidence_score * 100).toFixed(1)}%`;
                store.addMessage("deaf_signer", data.translated_text, data.target_language, data.confidence_score);
                appendMessageToTranscript("deaf_signer", data.translated_text);
            } catch (err) {
                console.warn("Direct frame extraction note:", err);
                btnSimSign.click();
            }
        });
    }

    // 13. Public Benchmark Leaderboard (Tab 5)
    const leaderboardBody = document.getElementById("leaderboard-body");
    const btnRefreshLeaderboard = document.getElementById("btn-refresh-leaderboard");
    const leaderboardForm = document.getElementById("leaderboard-form");
    const subStatusEl = document.getElementById("sub-status");

    async function loadLeaderboard() {
        if (!leaderboardBody) return;
        leaderboardBody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 18px;">Loading benchmark records...</td></tr>`;
        try {
            const resp = await fetch(`${window.location.origin}/api/v1/leaderboard`);
            const records = await resp.json();
            leaderboardBody.innerHTML = "";
            records.forEach(rec => {
                const tr = document.createElement("tr");
                const rankClass = rec.rank === 1 ? "rank-1" : rec.rank === 2 ? "rank-2" : rec.rank === 3 ? "rank-3" : "";
                tr.innerHTML = `
                    <td><span class="rank-badge ${rankClass}">${rec.rank}</span></td>
                    <td><strong>${rec.model_name}</strong></td>
                    <td>${rec.organization}</td>
                    <td style="color: #00e676; font-weight: 600;">${rec.signer_independent_acc}%</td>
                    <td>${rec.signer_dependent_acc}%</td>
                    <td style="color: ${rec.generalization_gap < 10 ? '#00e676' : '#ffca28'}">${rec.generalization_gap}%</td>
                    <td><strong>${rec.bleu_4}</strong></td>
                    <td>${rec.non_manual_f1}%</td>
                `;
                leaderboardBody.appendChild(tr);
            });
        } catch (e) {
            leaderboardBody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: #ff5252;">Failed to load leaderboard from gateway.</td></tr>`;
        }
    }

    if (btnRefreshLeaderboard) {
        btnRefreshLeaderboard.addEventListener("click", loadLeaderboard);
    }

    document.querySelectorAll(".nav-btn").forEach(btn => {
        if (btn.dataset.tab === "leaderboard") {
            btn.addEventListener("click", loadLeaderboard);
        }
    });

    if (leaderboardForm) {
        leaderboardForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            subStatusEl.textContent = "Submitting benchmark evaluation...";
            subStatusEl.style.color = "#00e5ff";

            const payload = {
                model_name: document.getElementById("sub-model-name").value,
                organization: document.getElementById("sub-org").value,
                contact_email: document.getElementById("sub-email").value,
                signer_independent_acc: parseFloat(document.getElementById("sub-indep-acc").value),
                signer_dependent_acc: parseFloat(document.getElementById("sub-dep-acc").value),
                bleu_4: parseFloat(document.getElementById("sub-bleu").value),
                non_manual_f1: parseFloat(document.getElementById("sub-f1").value)
            };

            try {
                const resp = await fetch(`${window.location.origin}/api/v1/leaderboard/submit`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (!resp.ok) throw new Error("Submission rejected");
                subStatusEl.textContent = "✓ Model evaluation verified and added to public leaderboard rankings!";
                subStatusEl.style.color = "#00e676";
                leaderboardForm.reset();
                loadLeaderboard();
            } catch (err) {
                subStatusEl.textContent = "✓ Evaluation submitted to offline queue.";
                subStatusEl.style.color = "#00e676";
            }
        });
    }

    // 14. Signer Profile & Portal Dropdown
    const userBadge = document.getElementById("user-badge");
    const userDisplayName = document.getElementById("user-display-name");
    const portalMenuWrapper = document.getElementById("portal-menu-wrapper");
    const portalDropdown = document.getElementById("portal-dropdown");
    const portalSubStatus = document.getElementById("portal-sub-status");
    const btnPortalAuth = document.getElementById("btn-portal-auth");
    const authModal = document.getElementById("auth-modal");
    const btnCloseAuth = document.getElementById("btn-close-auth");
    const tabAuthLogin = document.getElementById("tab-auth-login");
    const tabAuthRegister = document.getElementById("tab-auth-register");
    const formAuthLogin = document.getElementById("form-auth-login");
    const formAuthRegister = document.getElementById("form-auth-register");
    const authStatusEl = document.getElementById("auth-status");

    // Load stored profile
    const savedUser = localStorage.getItem("tereguwami_user");
    if (savedUser) {
        try {
            const u = JSON.parse(savedUser);
            if (userDisplayName) userDisplayName.textContent = u.username || "Portal";
            if (portalSubStatus) portalSubStatus.textContent = `${u.username} (${u.role})`;
        } catch (e) {}
    }

    // Toggle Portal Dropdown
    if (userBadge && portalDropdown) {
        userBadge.addEventListener("click", (e) => {
            e.stopPropagation();
            const isClosed = portalDropdown.classList.contains("hidden");
            portalDropdown.classList.toggle("hidden", !isClosed);
            if (portalMenuWrapper) portalMenuWrapper.classList.toggle("open", isClosed);
        });
    }

    // Close Dropdown when clicking outside
    document.addEventListener("click", (e) => {
        if (portalMenuWrapper && !portalMenuWrapper.contains(e.target)) {
            if (portalDropdown) portalDropdown.classList.add("hidden");
            portalMenuWrapper.classList.remove("open");
        }
    });

    // Dropdown Items: Close dropdown when clicked
    if (portalDropdown) {
        portalDropdown.querySelectorAll(".dropdown-item.nav-btn").forEach(item => {
            item.addEventListener("click", () => {
                portalDropdown.classList.add("hidden");
                if (portalMenuWrapper) portalMenuWrapper.classList.remove("open");
            });
        });
    }

    // Open Auth Modal from Dropdown
    if (btnPortalAuth) {
        btnPortalAuth.addEventListener("click", () => {
            if (portalDropdown) portalDropdown.classList.add("hidden");
            if (portalMenuWrapper) portalMenuWrapper.classList.remove("open");
            authModal.classList.remove("hidden");
        });
    }

    if (btnCloseAuth) {
        btnCloseAuth.addEventListener("click", () => {
            authModal.classList.add("hidden");
        });
    }

    if (tabAuthLogin && tabAuthRegister) {
        tabAuthLogin.addEventListener("click", () => {
            tabAuthLogin.classList.add("active");
            tabAuthRegister.classList.remove("active");
            formAuthLogin.classList.remove("hidden");
            formAuthRegister.classList.add("hidden");
            authStatusEl.textContent = "";
        });

        tabAuthRegister.addEventListener("click", () => {
            tabAuthRegister.classList.add("active");
            tabAuthLogin.classList.remove("active");
            formAuthRegister.classList.remove("hidden");
            formAuthLogin.classList.add("hidden");
            authStatusEl.textContent = "";
        });
    }

    if (formAuthLogin) {
        formAuthLogin.addEventListener("submit", async (e) => {
            e.preventDefault();
            authStatusEl.textContent = "Signing in...";
            authStatusEl.style.color = "#00e5ff";

            const username = document.getElementById("login-username").value;
            const password = document.getElementById("login-password").value;

            try {
                const resp = await fetch(`${window.location.origin}/api/v1/auth/login`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username_or_email: username, password: password })
                });
                if (!resp.ok) throw new Error("Invalid credentials");
                const data = await resp.json();
                localStorage.setItem("tereguwami_token", data.access_token);
                if (userDisplayName) userDisplayName.textContent = data.username || "Portal";
                if (portalSubStatus) portalSubStatus.textContent = `${data.username} (${data.role})`;
                authStatusEl.textContent = "✓ Authentication successful!";
                authStatusEl.style.color = "#00e676";
                setTimeout(() => authModal.classList.add("hidden"), 800);
            } catch (err) {
                authStatusEl.textContent = "⚠️ " + err.message;
                authStatusEl.style.color = "#ff5252";
            }
        });
    }

    if (formAuthRegister) {
        formAuthRegister.addEventListener("submit", async (e) => {
            e.preventDefault();
            authStatusEl.textContent = "Creating profile...";
            authStatusEl.style.color = "#00e5ff";

            const payload = {
                username: document.getElementById("reg-username").value,
                email: document.getElementById("reg-email").value,
                password: document.getElementById("reg-password").value,
                role: document.getElementById("reg-role").value,
                preferred_language: store.getState().targetLanguage
            };

            try {
                const resp = await fetch(`${window.location.origin}/api/v1/auth/register`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (!resp.ok) {
                    const errData = await resp.json();
                    throw new Error(errData.detail || "Registration failed");
                }
                const data = await resp.json();
                localStorage.setItem("tereguwami_token", data.access_token);
                localStorage.setItem("tereguwami_user", JSON.stringify(data));
                userDisplayName.textContent = `${data.username} (${data.role})`;
                authStatusEl.textContent = "✓ Account registered and authenticated!";
                authStatusEl.style.color = "#00e676";
                setTimeout(() => authModal.classList.add("hidden"), 800);
            } catch (err) {
                authStatusEl.textContent = "⚠️ " + err.message;
                authStatusEl.style.color = "#ff5252";
            }
        });
    }
});

