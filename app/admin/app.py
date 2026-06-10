"""
iFusion Admin — standalone FastAPI app on port 8001.

Start with:
    uvicorn app.admin.app:app --host 0.0.0.0 --port 8001
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.admin.routes import router as admin_router

_STATIC = Path(__file__).parent.parent / "static"

app = FastAPI(title="iFusion Admin", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def serve_admin_ui():
    return FileResponse(_STATIC / "admin.html")
