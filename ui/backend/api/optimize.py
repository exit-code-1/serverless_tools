# -*- coding: utf-8 -*-
"""Optimization task endpoints."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models.schemas import (
    OptimizeRequest,
    OptimizeResultPayload,
    TaskCreatedResponse,
    TaskInfo,
)
from ..services import optimizer_runner
from ..services.task_manager import TaskRecord, get_task_manager


router = APIRouter(prefix="/api/optimize", tags=["optimize"])


@router.post("", response_model=TaskCreatedResponse)
def submit_optimize(request: OptimizeRequest) -> TaskCreatedResponse:
    task_manager = get_task_manager()
    record = task_manager.create("optimize", metadata={
        "dataset": request.dataset,
        "query_id": request.query_id,
        "algorithm": request.algorithm,
    })

    params = request.dict(exclude_unset=False)
    params.pop("dataset", None)
    params.pop("query_id", None)
    params.pop("algorithm", None)
    params.pop("force_rerun", None)

    def _worker() -> None:
        try:
            payload = optimizer_runner.run_optimization(
                task_manager=task_manager,
                task_id=record.task_id,
                dataset=request.dataset,
                query_id=request.query_id,
                algorithm=request.algorithm,
                params=params,
                force_rerun=request.force_rerun,
            )
            task_manager.update_metadata(
                record.task_id,
                optimization_csv_path=payload["optimization_csv_path"],
                optimization_json_path=payload["optimization_json_path"],
                total_cpu_time=payload.get("total_cpu_time"),
                max_dop=payload.get("max_dop"),
            )
            task_manager.mark_completed(record.task_id)
        except Exception as exc:  # noqa: BLE001
            task_manager.mark_failed(record.task_id, str(exc))

    threading.Thread(target=_worker, daemon=True, name=f"optimize-{record.task_id}").start()
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.get("/{task_id}", response_model=TaskInfo)
def get_optimize_status(task_id: str) -> TaskInfo:
    record = _require_task(task_id)
    return TaskInfo(**record.to_public_dict())


@router.get("/{task_id}/result", response_model=OptimizeResultPayload)
def get_optimize_result(task_id: str) -> OptimizeResultPayload:
    record = _require_task(task_id)
    if record.status != "completed":
        raise HTTPException(status_code=409, detail=f"Task not completed yet: {record.status}")
    payload = optimizer_runner.load_persisted_payload(task_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No persisted payload for this task")
    return OptimizeResultPayload(**payload)


def _require_task(task_id: str) -> TaskRecord:
    record = get_task_manager().get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return record
