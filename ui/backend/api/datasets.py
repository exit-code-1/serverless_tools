# -*- coding: utf-8 -*-
"""Endpoints for dataset/query browsing."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from ..models.schemas import DatasetInfo, QueryDetail, QuerySummary
from ..services import dataset_loader


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=List[DatasetInfo])
def list_datasets() -> List[DatasetInfo]:
    return [DatasetInfo(**item) for item in dataset_loader.list_datasets()]


@router.get("/{dataset}/queries", response_model=List[QuerySummary])
def list_queries(dataset: str) -> List[QuerySummary]:
    try:
        rows = dataset_loader.list_queries(dataset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return [QuerySummary(**row) for row in rows]


@router.get("/{dataset}/queries/{query_id}", response_model=QueryDetail)
def get_query_detail(dataset: str, query_id: int) -> QueryDetail:
    try:
        detail = dataset_loader.get_query_detail(dataset, query_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return QueryDetail(**detail)
