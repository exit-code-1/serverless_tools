# -*- coding: utf-8 -*-
"""Read structured actual execution results from data_file CSVs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models.schemas import ActualResult
from ..services import results_loader


router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{dataset}/{query_id}", response_model=ActualResult)
def get_result(dataset: str, query_id: int) -> ActualResult:
    payload = results_loader.read_latest_result(dataset, query_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="no actual result available")
    return ActualResult(**payload)
