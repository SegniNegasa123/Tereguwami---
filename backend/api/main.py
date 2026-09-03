"""
Tereguwami (ተርጓሚ) FastAPI Gateway Entrypoint
Part of Tereguwami Multimodal Ethiopian Sign Language AI Platform (§8, §11)
"""

import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.db.migrations import run_migrations_and_seed
from backend.api.routes_auth import router as auth_router
from backend.api.routes_translation import router as translation_router
from backend.api.routes_production import router as production_router
from backend.api.routes_personalization import router as personalization_router
from backend.api.routes_silent_speech import router as silent_speech_router
from backend.api.routes_governance import router as governance_router
from backend.api.routes_leaderboard import router as leaderboard_router
from backend.api.routes_health import router as health_router
from backend.streaming.websocket_manager import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and seed database migrations on startup
    run_migrations_and_seed()
    yield


app = FastAPI(
    title="Tereguwami (ተርጓሚ) API",
    description=(
        "Production API Gateway for Ethiopian Sign Language (ESL / ETHSL) "
        "continuous bidirectional translation, generative 3D avatar synthesis, "
        "few-shot personalization, and silent-speech sEMG neuromotor decoding."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth_router)
app.include_router(translation_router)
app.include_router(production_router)
app.include_router(personalization_router)
app.include_router(silent_speech_router)
app.include_router(governance_router)
app.include_router(leaderboard_router)
app.include_router(health_router)


# Real-time WebSocket Streaming Endpoint (§8.1, §11)
@app.websocket("/ws/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Bidirectional streaming WebSocket endpoint for real-time two-way communication:
    - Ingests streaming normalized 543 3D landmark keypoints from camera
    - Emits incremental translation hypotheses and low-confidence clarification alerts
    - Broadcasts generative 3D avatar pose frames
    """
    await ws_manager.connect(websocket, session_id)
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                data = json.loads(raw_text)
                msg_type = data.get("type", "frame")
                if msg_type == "frame":
                    await ws_manager.handle_incoming_frame(session_id, data)
                elif msg_type == "speech_to_avatar":
                    text_content = data.get("text", "")
                    if text_content:
                        await ws_manager.handle_speech_to_avatar(session_id, text_content)
                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)


# Mount static assets for web-client and shared-components
web_client_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mobile", "web-client"))
shared_comp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mobile", "shared-components"))
avatar_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "avatar"))

if os.path.exists(web_client_dir):
    app.mount("/client", StaticFiles(directory=web_client_dir), name="client")
if os.path.exists(shared_comp_dir):
    app.mount("/shared-components", StaticFiles(directory=shared_comp_dir), name="shared_components")
if os.path.exists(avatar_dir):
    app.mount("/avatar", StaticFiles(directory=avatar_dir, html=True), name="avatar")


@app.get("/app", response_class=HTMLResponse)
async def get_product_app():
    """Serves the complete Tereguwami (ተርጓሚ) Two-Way Conversation Web & Mobile App."""
    index_path = os.path.join(web_client_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
            import time
            content = content.replace('href="styles.css"', f'href="/client/styles.css?v={int(time.time())}"')
            content = content.replace('src="avatar_embed.js"', f'src="/client/avatar_embed.js?v={int(time.time())}"')
            content = content.replace('src="app.js"', f'src="/client/app.js?v={int(time.time())}"')
            content = content.replace('src="../shared-components/tereguwami_sdk.js"', f'src="/shared-components/tereguwami_sdk.js?v={int(time.time())}"')
            content = content.replace('src="../shared-components/state_store.js"', f'src="/shared-components/state_store.js?v={int(time.time())}"')
            return HTMLResponse(
                content=content,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    return HTMLResponse(content="<h1>Product app not found</h1>", status_code=404)


@app.get("/avatar", response_class=HTMLResponse)
async def get_avatar_page():
    """Serves the interactive 3D WebGL ESL Avatar Engine preview."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "rendering", "webgl_avatar_engine.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Avatar HTML not found</h1>", status_code=404)


@app.get("/")
async def root():
    return {
        "project": "Tereguwami (ተርጓሚ)",
        "message": "Ethiopian Sign Language AI Platform API Gateway",
        "app": "/app",
        "docs": "/docs",
        "health": "/api/v1/health",
        "avatar": "/avatar"
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
