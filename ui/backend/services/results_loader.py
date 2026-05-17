# -*- coding: utf-8 -*-
"""Reads openGauss output files (plan_info.csv, query_info.csv) and returns a
structured per-query view for the UI. The execution flow only ever runs one
query at a time, so the most recent rows for the requested query_id are
sufficient."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from ..settings import Settings, get_settings


def read_latest_result(
    dataset: str,
    query_id: int,
    *,
    settings: Optional[Settings] = None,
) -> Optional[Dict[str, Any]]:
    settings = settings or get_settings()
    plan_info_path = os.path.join(settings.data_file_dir, "plan_info.csv")
    query_info_path = os.path.join(settings.data_file_dir, "query_info.csv")

    query_df = _read_csv(query_info_path)
    plan_df = _read_csv(plan_info_path)

    if query_df is None or query_df.empty:
        return None

    query_df["query_id"] = pd.to_numeric(query_df["query_id"], errors="coerce")
    matches = query_df[query_df["query_id"].astype("Int64") == int(query_id)]
    if matches.empty:
        return None
    query_row = matches.iloc[-1]

    operators: List[Dict[str, Any]] = []
    if plan_df is not None and not plan_df.empty:
        plan_df["query_id"] = pd.to_numeric(plan_df["query_id"], errors="coerce")
        plan_rows = plan_df[plan_df["query_id"].astype("Int64") == int(query_id)]
        if not plan_rows.empty:
            plan_rows = plan_rows.sort_values("plan_id")
            parent_lookup: Dict[int, int] = {}
            child_lookup: Dict[int, int] = {}
            for _, row in plan_rows.iterrows():
                plan_id = _to_int(row.get("plan_id"))
                child = _to_int(row.get("child_plan"))
                if plan_id is None:
                    continue
                if child is not None and child != plan_id and child >= 0:
                    child_lookup[plan_id] = child
                    parent_lookup.setdefault(child, plan_id)
            for _, row in plan_rows.iterrows():
                plan_id = _to_int(row.get("plan_id")) or 0
                operators.append({
                    "plan_id": plan_id,
                    "operator_type": str(row.get("operator_type", "")),
                    "dop": _to_int(row.get("dop")) or 0,
                    "query_dop": _to_int(row.get("query_dop")),
                    "execution_time": _to_float(row.get("execution_time")),
                    "total_time": _to_float(row.get("total_time")),
                    "actual_rows": _to_float(row.get("actual_rows")),
                    "peak_mem": _to_float(row.get("peak_mem")),
                    "width": _to_int(row.get("width")),
                    "parent_child": parent_lookup.get(plan_id, -1),
                    "left_child": child_lookup.get(plan_id, -1),
                })

    return {
        "dataset": dataset,
        "query_id": int(query_id),
        "query_dop": _to_int(query_row.get("dop")),
        "execution_time": _to_float(query_row.get("execution_time")),
        "executor_start_time": _to_float(query_row.get("executor_start_time")),
        "query_used_mem": _to_float(query_row.get("query_used_mem")),
        "operator_mem": _to_float(query_row.get("operator_mem")),
        "process_used_mem": _to_float(query_row.get("process_used_mem")),
        "cpu_time": _to_float(query_row.get("cpu_time")),
        "io_time": _to_float(query_row.get("io_time")),
        "total_costs": _to_float(query_row.get("total_costs")),
        "operator_num": _to_int(query_row.get("operator_num")),
        "table_names": _to_str(query_row.get("table_names")),
        "operators": operators,
        "plan_info_path": plan_info_path,
        "query_info_path": query_info_path,
        "sourced_at": time.time(),
    }


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    if "query_id" not in df.columns:
        return None
    return df


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None
