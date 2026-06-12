"""
CV Analyzer AI — FastAPI application entry point.

All business logic lives in services/. This file only:
  • Creates the FastAPI app and configures middleware
  • Registers the three routers under /api
  • Exposes a /health endpoint
  • Creates DB tables on startup
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from backend.database import engine  # SessionLocal is not used here; routers use get_db()
import backend.models as models

# ── Lifespan (startup / shutdown hooks) ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Keep startup light so the web UI and health check are available immediately."""
    print("[*] Starting CV Analyzer AI server...")
    # Create DB tables inside lifespan to avoid blocking import-time execution.
    try:
        models.Base.metadata.create_all(bind=engine)
        print("[+] Database tables initialized successfully")
    except Exception as e:
        print(f"[!] Database initialization failed: {e}")
    print("[+] Web app ready. AI models will load on the first analysis request.")
    yield
    print("[*] Shutting down...")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CV Analyzer AI",
    description="AI-powered CV analysis: role prediction, ATS scoring, and improvement tips.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in development.
# In production, replace "*" with your actual frontend domain.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
allow_creds = True
if "*" in ALLOWED_ORIGINS:
    allow_creds = False  # Starlette rejects wildcard origins with credentials=True

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ────────────────────────────────────────────────────────────────────
from backend.routes.analyze import router as analyze_router
from backend.routes.history import router as history_router
from backend.routes.users import router as users_router
from backend.routes.rewrite import router as rewrite_router

app.include_router(analyze_router, prefix="/api/analyze", tags=["Analysis"])
app.include_router(history_router, prefix="/api/history", tags=["History"])
app.include_router(users_router,   prefix="/api/users",   tags=["Users"])
app.include_router(rewrite_router, prefix="/api/rewrite", tags=["Rewrite"])


# ── Static frontend ───────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "Frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the frontend app."""
    return RedirectResponse(url="/app/index.html")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    """Quick liveness probe."""
    return {"status": "ok", "version": app.version}
