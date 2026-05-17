# -*- coding: utf-8 -*-
"""openGauss execution endpoints."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models.schemas import ExecuteRequest, TaskCreatedResponse, TaskInfo
from ..services import gauss_runner, optimizer_runner
from ..services.task_manager import TaskRecord, get_task_manager
from ..settings import get_settings


router = APIRouter(prefix="/api/execute", tags=["execute"])


@router.post("", response_model=TaskCreatedResponse)
def submit_execute(request: ExecuteRequest) -> TaskCreatedResponse:
    settings = get_settings()
    csv_path = request.optimization_csv_path
    if not csv_path:
        from ..services.optimizer_runner import _resolve_algorithm_paths

        train_mode = settings.optimize_defaults.get("train_mode", "exact_train")
        _, csv_path = _resolve_algorithm_paths(settings, request.dataset, train_mode, "pipeline")
    if not csv_path:
        raise HTTPException(status_code=400, detail="optimization_csv_path missing and cannot be inferred")

    base_dop = int(request.base_dop or settings.optimize_defaults.get("base_dop", 64))

    task_manager = get_task_manager()
    record = task_manager.create("execute", metadata={
        "dataset": request.dataset,
        "query_id": request.query_id,
        "optimization_csv_path": csv_path,
        "base_dop": base_dop,
    })

    def _worker() -> None:
        try:
            result = gauss_runner.run_execution(
                task_manager=task_manager,
                task_id=record.task_id,
                dataset=request.dataset,
                query_id=request.query_id,
                optimization_csv_path=csv_path,
                base_dop=base_dop,
                restart_gauss=request.restart_gauss,
            )
            task_manager.update_metadata(record.task_id, **result)
            task_manager.mark_completed(record.task_id)
        except Exception as exc:  # noqa: BLE001
            task_manager.mark_failed(record.task_id, str(exc))

    threading.Thread(target=_worker, daemon=True, name=f"execute-{record.task_id}").start()
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.get("/{task_id}", response_model=TaskInfo)
def get_execute_status(task_id: str) -> TaskInfo:
    record = _require_task(task_id)
    return TaskInfo(**record.to_public_dict())


@router.get("/{task_id}/logs")
def get_execute_logs(task_id: str) -> Dict[str, Any]:
    record = _require_task(task_id)
    task_manager = get_task_manager()
    return {
        "task_id": task_id,
        "status": record.status,
        "logs": task_manager.get_logs(task_id),
    }


@router.get("/{task_id}/stream")
async def stream_execute(task_id: str, request: Request) -> StreamingResponse:
    record = _require_task(task_id)
    task_manager = get_task_manager()

    async def event_source():
        async for event in task_manager.subscribe(task_id):
            if await request.is_disconnected():
                break
            yield _format_sse(event)
        yield _format_sse({"event": "close", "data": "1"})

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.get("/tasks/all", response_model=List[TaskInfo])
def list_tasks(kind: Optional[str] = None) -> List[TaskInfo]:
    task_manager = get_task_manager()
    records = task_manager.list_all()
    if kind:
        records = [r for r in records if r.kind == kind]
    records.sort(key=lambda r: r.created_at, reverse=True)
    return [TaskInfo(**r.to_public_dict()) for r in records]


def _format_sse(event: Dict[str, str]) -> str:
    event_name = event.get("event", "message")
    data = event.get("data", "")
    # SSE multiline data must be prefixed line by line.
    formatted_data = "\n".join(f"data: {chunk}" for chunk in data.splitlines() or [""])
    return f"event: {event_name}\n{formatted_data}\n\n"


def _require_task(task_id: str) -> TaskRecord:
    record = get_task_manager().get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return record
