# -*- coding: utf-8 -*-
"""History records endpoints."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    HistoryCreateRequest,
    HistoryRunDetail,
    HistoryRunSummary,
)
from ..services.history_store import get_history_store


router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=List[HistoryRunSummary])
def list_history(
    dataset: Optional[str] = None,
    query_id: Optional[int] = None,
    algorithm: Optional[str] = None,
    limit: int = 200,
) -> List[HistoryRunSummary]:
    rows = get_history_store().list_runs(
        dataset=dataset,
        query_id=query_id,
        algorithm=algorithm,
        limit=limit,
    )
    return [HistoryRunSummary(**row) for row in rows]


@router.get("/{run_id}", response_model=HistoryRunDetail)
def get_history(run_id: int) -> HistoryRunDetail:
    row = get_history_store().get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return HistoryRunDetail(**row)


@router.post("", response_model=HistoryRunDetail)
def add_history(request: HistoryCreateRequest) -> HistoryRunDetail:
    store = get_history_store()
    new_id = store.add_run(request.dict())
    row = store.get_run(new_id)
    if row is None:
        raise HTTPException(status_code=500, detail="insert failed")
    return HistoryRunDetail(**row)


@router.delete("/{run_id}")
def delete_history(run_id: int) -> dict:
    ok = get_history_store().delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="run not found")
    return {"deleted": run_id}
