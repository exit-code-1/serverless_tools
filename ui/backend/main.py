# -*- coding: utf-8 -*-
"""FastAPI entry point.

Run with:
    uvicorn ui.backend.main:app --host 0.0.0.0 --port 8000

The backend is intentionally self-contained: it injects the predictor project
root into sys.path on startup and reuses the existing optimization modules.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_UI_DIR = os.path.dirname(_BACKEND_DIR)
_PROJECT_ROOT = os.path.dirname(_UI_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ui.backend.api import datasets as datasets_api  # noqa: E402
from ui.backend.api import execute as execute_api  # noqa: E402
from ui.backend.api import history as history_api  # noqa: E402
from ui.backend.api import optimize as optimize_api  # noqa: E402
from ui.backend.api import results as results_api  # noqa: E402
from ui.backend.services.task_manager import get_task_manager  # noqa: E402
from ui.backend.settings import get_settings  # noqa: E402


app = FastAPI(title="Serverless Predictor UI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(datasets_api.router)
app.include_router(optimize_api.router)
app.include_router(execute_api.router)
app.include_router(results_api.router)
app.include_router(history_api.router)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "predictor_root": settings.predictor_root,
        "query_txt_path": settings.query_txt_path,
        "data_file_dir": settings.data_file_dir,
        "gauss_data_dir": settings.gauss_data_dir,
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    """Expose non-sensitive config knobs to the frontend."""
    settings = get_settings()
    return {
        "predictor_root": settings.predictor_root,
        "query_txt_path": settings.query_txt_path,
        "data_file_dir": settings.data_file_dir,
        "gauss_data_dir": settings.gauss_data_dir,
        "databases": settings.databases,
        "sql_dirs": settings.sql_dirs,
        "optimize_defaults": settings.optimize_defaults,
    }


@app.get("/api/tasks/{task_id}/stream")
async def stream_any_task(task_id: str, request: Request) -> StreamingResponse:
    """Generic SSE endpoint usable by both optimize and execute tasks."""
    task_manager = get_task_manager()
    if task_manager.get(task_id) is None:
        raise HTTPException(status_code=404, detail="task not found")

    async def event_source():
        async for event in task_manager.subscribe(task_id):
            if await request.is_disconnected():
                break
            event_name = event.get("event", "message")
            data = event.get("data", "")
            formatted = "\n".join(f"data: {chunk}" for chunk in data.splitlines() or [""])
            yield f"event: {event_name}\n{formatted}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/api/tasks/{task_id}/logs")
def get_task_logs(task_id: str) -> Dict[str, Any]:
    task_manager = get_task_manager()
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task_id,
        "status": record.status,
        "logs": task_manager.get_logs(task_id),
    }


# ----- Static frontend serving -------------------------------------------------

_FRONTEND_DIST = os.path.join(_UI_DIR, "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "message": "Frontend build not found. Run `npm install && npm run build` inside ui/frontend or hit /api/* endpoints directly.",
                "api_docs": "/docs",
            }
        )
