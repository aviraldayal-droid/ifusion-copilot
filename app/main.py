"""
TBG AI Copilot — FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
import logging.config
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
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": str(_LOG_DIR / "app.log"),
            "maxBytes": 10 * 1024 * 1024,   # 10 MB per file
            "backupCount": 5,                # keep app.log + 5 rotated copies
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
    "loggers": {
        "tbg": {"level": "DEBUG", "propagate": True},
        "uvicorn.access": {"level": "WARNING", "propagate": True},
    },
})

log = logging.getLogger("tbg.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s v%s", settings.APP_TITLE, settings.APP_VERSION)
    if settings.is_ollama_cloud:
        log.info("LLM: %s (OLLAMA CLOUD)", settings.OLLAMA_MODEL)
    else:
        log.info("LLM: %s (LOCAL at %s)", settings.OLLAMA_MODEL, settings.OLLAMA_BASE_URL)
    log.info("LangSmith tracing: %s", "enabled" if settings.LANGSMITH_API_KEY else "disabled")
    yield
    log.info("Shutting down %s", settings.APP_TITLE)


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
