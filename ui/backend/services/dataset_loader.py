# -*- coding: utf-8 -*-
"""Adapter around the predictor's DatasetLoader to expose UI-friendly views."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..settings import get_settings


def _ensure_predictor_on_path() -> None:
    settings = get_settings()
    root = settings.predictor_root
    if root not in sys.path:
        sys.path.insert(0, root)


def list_datasets() -> List[Dict[str, Any]]:
    _ensure_predictor_on_path()
    from config.main_config import DATASETS

    settings = get_settings()
    sql_dirs = settings.sql_dirs
    databases = settings.databases

    items: List[Dict[str, Any]] = []
    for name, cfg in DATASETS.items():
        try:
            paths = _resolve_paths(name)
        except Exception:
            continue
        query_count = 0
        if os.path.exists(paths["query_info"]):
            try:
                df = _read_query_info(paths["query_info"])
                query_count = int(df["query_id"].nunique()) if not df.empty else 0
            except Exception:
                query_count = 0
        items.append({
            "name": name,
            "display_name": _display_name(name),
            "plan_info_path": paths["plan_info"],
            "query_info_path": paths["query_info"],
            "query_count": query_count,
            "database": databases.get(name),
            "sql_dir": sql_dirs.get(name),
        })
    return items


def list_queries(dataset: str) -> List[Dict[str, Any]]:
    paths = _resolve_paths(dataset)
    df = _read_query_info(paths["query_info"])
    if df.empty:
        return []
    aggregated = (
        df.sort_values(["query_id", "dop"]).groupby("query_id", as_index=False).first()
    )
    rows: List[Dict[str, Any]] = []
    for _, row in aggregated.iterrows():
        rows.append({
            "query_id": int(row["query_id"]),
            "dop": _to_int(row.get("dop")),
            "execution_time": _to_float(row.get("execution_time")),
            "query_used_mem": _to_float(row.get("query_used_mem")),
            "operator_num": _to_int(row.get("operator_num")),
            "table_names": _to_str(row.get("table_names")),
        })
    rows.sort(key=lambda x: x["query_id"])
    return rows


def get_query_detail(dataset: str, query_id: int) -> Dict[str, Any]:
    paths = _resolve_paths(dataset)
    query_df = _read_query_info(paths["query_info"])
    if query_df.empty:
        raise KeyError(f"query_info empty for dataset {dataset}")

    query_rows = query_df[query_df["query_id"].astype(int) == int(query_id)]
    if query_rows.empty:
        raise KeyError(f"query_id {query_id} not in dataset {dataset}")
    query_row = query_rows.sort_values("dop").iloc[0]

    plan_df = _read_plan_info(paths["plan_info"])
    plan_rows = plan_df[plan_df["query_id"].astype(int) == int(query_id)]
    if plan_rows.empty:
        raise KeyError(f"query_id {query_id} has no plan_info rows")
    plan_rows = plan_rows.sort_values("plan_id")

    base_dop = _to_int(query_row.get("dop")) or 0
    base_rows = plan_rows[plan_rows["query_dop"].astype(int) == base_dop] if "query_dop" in plan_rows.columns else plan_rows
    if base_rows.empty:
        base_rows = plan_rows

    # plan_info.csv only carries `child_plan` (downstream child). The query.txt
    # convention exposes both `left_child` and `parent_child`, so reverse map
    # parents from the child references collected above.
    parent_lookup: Dict[int, int] = {}
    child_lookup: Dict[int, int] = {}
    for _, row in base_rows.iterrows():
        plan_id = _to_int(row.get("plan_id"))
        child = _to_int(row.get("child_plan"))
        if plan_id is None:
            continue
        if child is not None and child != plan_id and child >= 0:
            child_lookup[plan_id] = child
            parent_lookup.setdefault(child, plan_id)

    operators: List[Dict[str, Any]] = []
    for _, row in base_rows.iterrows():
        plan_id = int(row["plan_id"])
        operators.append({
            "plan_id": plan_id,
            "operator_type": str(row.get("operator_type", "")),
            "width": _to_int(row.get("width")) or 0,
            "dop": _to_int(row.get("dop")) or 0,
            "parent_child": parent_lookup.get(plan_id, -1),
            "left_child": child_lookup.get(plan_id, -1),
        })

    summary = {
        "query_id": int(query_id),
        "dop": _to_int(query_row.get("dop")),
        "execution_time": _to_float(query_row.get("execution_time")),
        "query_used_mem": _to_float(query_row.get("query_used_mem")),
        "operator_num": _to_int(query_row.get("operator_num")),
        "table_names": _to_str(query_row.get("table_names")),
    }
    sql_text, sql_path = _try_load_sql(dataset, int(query_id), query_row)
    return {
        "dataset": dataset,
        "query_id": int(query_id),
        "summary": summary,
        "operators": operators,
        "sql_text": sql_text,
        "sql_path": sql_path,
    }


def _try_load_sql(dataset: str, query_id: int, query_row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    settings = get_settings()
    sql_dir = settings.sql_dirs.get(dataset)
    if sql_dir:
        candidate = os.path.join(sql_dir, f"{query_id}.sql")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fp:
                    return fp.read(), candidate
            except OSError:
                pass
    text = _to_str(query_row.get("query_string"))
    return (text or None), None


def _resolve_paths(dataset: str) -> Dict[str, str]:
    _ensure_predictor_on_path()
    from config.main_config import DATASETS, PROJECT_ROOT

    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset}")
    cfg = DATASETS[dataset]
    test_dir = os.path.join(PROJECT_ROOT, "data_kunpeng", cfg["test_dir"])
    return {
        "plan_info": os.path.join(test_dir, cfg["plan_info_file"]),
        "query_info": os.path.join(test_dir, cfg["query_info_file"]),
        "dir": test_dir,
    }


def _read_query_info(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    if "query_id" not in df.columns:
        return pd.DataFrame()
    df["query_id"] = pd.to_numeric(df["query_id"], errors="coerce")
    df = df.dropna(subset=["query_id"])
    df["query_id"] = df["query_id"].astype(int)
    return df


def _read_plan_info(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    if "query_id" not in df.columns:
        return pd.DataFrame()
    df["query_id"] = pd.to_numeric(df["query_id"], errors="coerce")
    df = df.dropna(subset=["query_id"])
    df["query_id"] = df["query_id"].astype(int)
    if "plan_id" in df.columns:
        df["plan_id"] = pd.to_numeric(df["plan_id"], errors="coerce").fillna(0).astype(int)
    if "query_dop" in df.columns:
        df["query_dop"] = pd.to_numeric(df["query_dop"], errors="coerce").fillna(0).astype(int)
    return df


def _display_name(name: str) -> str:
    return {"tpch": "TPC-H", "tpcds": "TPC-DS", "job": "JOB"}.get(name, name)


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
