import os
from contextvars import ContextVar
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Per-request Ollama API key — set by route handlers, read by _make_llm().
# Falls back to settings.OLLAMA_API_KEY when not set.
request_api_key: ContextVar[str] = ContextVar("request_api_key", default="")

# Compute .env search paths at module level so Docker shallow paths work.
# Local: /home/.../tbg_copilot/tb/app/config/settings.py → parents 2,3,4 all valid.
# Docker: /app/app/config/settings.py → only parents 0-3 exist; skip parent[4].
_file_parents = list(Path(__file__).resolve().parents)
_ENV_FILES = [
    str(_file_parents[i] / ".env")
    for i in [4, 3, 2]
    if i < len(_file_parents) and (_file_parents[i] / ".env").exists()
]


class Settings(BaseSettings):
    DB_USER: str = "digiwise_rw"
    DB_PASSWORD: str = "Digi@3456rw$"
    DB_HOST: str = "197.230.47.51"
    DB_PORT: str = "5432"
    DB_NAME: str = "digiwise"
    # Ollama — supports both local and cloud
    # OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_BASE_URL: str = "https://api.ollama.com"
    OLLAMA_MODEL: str = "nemotron-3-nano:30b"
    OLLAMA_API_KEY: str = ""  # Set for Ollama Cloud, leave empty for local

    # Narration model for format_answer — defaults to OLLAMA_MODEL if empty.
    # Use a fluent language model here (e.g. mistral-large-3, qwen3-next:80b).
    OLLAMA_NARRATION_MODEL: str = "ministral-3:14b"

    # Embedding model for schema RAG — defaults to OLLAMA_MODEL if empty.
    # Pull a dedicated model for better quality: ollama pull nomic-embed-text
    OLLAMA_EMBEDDING_MODEL: str = ""

    @property
    def DATABASE_URL(self) -> str:
        from urllib.parse import quote_plus
        return f"postgresql://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    # PostgreSQL
    # DATABASE_URL: str = "postgresql://digiwise:digiwise_secret@localhost:5432/digiwise"

    # LangSmith
    LANGSMITH_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "tbg-ai-copilot"
    LANGCHAIN_TRACING_V2: str = "true"

    # Admin key required to call POST /api/v1/auth/register.
    # Set a strong secret here; leave empty to disable all registration.
    ADMIN_KEY: str = ""

    MAX_SESSIONS: int = 50
    SESSION_TTL_HOURS: int = 24
    APP_TITLE: str = "TBG AI Copilot"
    APP_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        extra="ignore",
    )

    @property
    def OLLAMA_CLIENT_KWARGS(self) -> dict[str, dict[str, str]]:
        """Return a dict of kwargs to initialize the Ollama client, based on current settings."""
        if self.OLLAMA_API_KEY:
            return {
                "headers": {'Authorization': f"Bearer {self.OLLAMA_API_KEY}"},
            }
        return {}
    @property
    def is_ollama_cloud(self) -> bool:
        """Return True if using Ollama Cloud, False if using local Ollama."""
        return bool(self.OLLAMA_API_KEY)


settings = Settings()

# Configure LangSmith tracing — set env vars immediately so every
# LangChain/LangGraph call is captured, including tool calls.
if settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
