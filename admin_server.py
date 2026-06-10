"""
Entry point for the iFusion Admin dashboard (port 8001).

Usage:
    python admin_server.py
    # or
    uvicorn app.admin.app:app --host 0.0.0.0 --port 8001 --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.admin.app:app", host="0.0.0.0", port=8001, reload=True)
