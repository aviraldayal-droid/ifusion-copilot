"""
TBG AI Copilot — FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.auth_routes import router as auth_router
from app.api.conv_routes import router as conv_router
from app.api.excel_routes import router as excel_router
from app.config.settings import settings

_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION}")
    if settings.is_ollama_cloud:
        print(f"LLM: {settings.OLLAMA_MODEL} (OLLAMA CLOUD)")
    else:
        print(f"LLM: {settings.OLLAMA_MODEL} (LOCAL at {settings.OLLAMA_BASE_URL})")
    print(f"LangSmith tracing: {'enabled' if settings.LANGSMITH_API_KEY else 'disabled'}")
    yield
    # Shutdown — nothing to clean up (sessions are in-memory)


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "Agentic AI backend for the TBG (Tableau de Bord de Gestion) financial reports "
        "of Moov Benin. Powered by LangGraph + Gemini 2.5 Flash."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(auth_router)
app.include_router(conv_router)
app.include_router(excel_router)

# Serve the chat UI at /
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(_STATIC / "index.html")
