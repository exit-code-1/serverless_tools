# -*- coding: utf-8 -*-
"""Convert a per-query slice of pipeline_optimization.csv into the query.txt
file consumed by openGauss. Mirrors the helper in the user's batch script."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


COLUMNS_TO_KEEP = ["plan_id", "operator_type", "width", "dop", "left_child", "parent_child"]


def generate_query_file(input_csv: str, query_id: int, output_path: str, dop_cap: int = 0) -> str:
    """Filter the input CSV to rows matching `query_id`, drop the `query_id`
    column and write the result to `output_path`. Returns the resolved path."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Pipeline optimization CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Pipeline optimization CSV is empty: {input_csv}")

    df.columns = [c.strip() for c in df.columns]
    if "query_id" not in df.columns:
        raise ValueError("CSV does not have a query_id column")

    filtered = df[df["query_id"].astype(int) == int(query_id)]
    if filtered.empty:
        raise ValueError(f"No rows for query_id={query_id} in {input_csv}")

    missing = [c for c in COLUMNS_TO_KEEP if c not in filtered.columns]
    if missing:
        raise ValueError(f"Pipeline CSV missing required columns: {missing}")

    out_df = filtered[COLUMNS_TO_KEEP].copy()
    if dop_cap > 0 and "dop" in out_df.columns:
        out_df["dop"] = pd.to_numeric(out_df["dop"], errors="coerce").fillna(1).astype(int).clip(upper=dop_cap)
    _force_root_pipeline_dop_one(out_df)
    out_df = out_df.sort_values("plan_id")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def preview_query_file(input_csv: str, query_id: int, dop_cap: int = 0) -> str:
    """Return the would-be content as a string without writing to disk."""
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    filtered = df[df["query_id"].astype(int) == int(query_id)][COLUMNS_TO_KEEP]
    if dop_cap > 0 and "dop" in filtered.columns:
        filtered = filtered.copy()
        filtered["dop"] = pd.to_numeric(filtered["dop"], errors="coerce").fillna(1).astype(int).clip(upper=dop_cap)
    _force_root_pipeline_dop_one(filtered)
    return filtered.sort_values("plan_id").to_csv(index=False)


def read_query_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def _force_root_pipeline_dop_one(df: pd.DataFrame) -> None:
    """Keep the top/root pipeline serial before the first exchange boundary."""
    if df.empty or not {"plan_id", "dop", "left_child", "parent_child", "operator_type"}.issubset(df.columns):
        return

    df["plan_id"] = pd.to_numeric(df["plan_id"], errors="coerce").fillna(-1).astype(int)
    df["left_child"] = pd.to_numeric(df["left_child"], errors="coerce").fillna(-1).astype(int)
    df["parent_child"] = pd.to_numeric(df["parent_child"], errors="coerce").fillna(-1).astype(int)

    by_plan_id = {int(row["plan_id"]): idx for idx, row in df.iterrows()}
    root_ids = [
        int(row["plan_id"])
        for _, row in df.iterrows()
        if int(row["parent_child"]) < 0
    ]

    for root_id in root_ids:
        visited = set()
        current_id = root_id
        while current_id in by_plan_id and current_id not in visited:
            visited.add(current_id)
            idx = by_plan_id[current_id]
            df.at[idx, "dop"] = 1

            operator_type = str(df.at[idx, "operator_type"]).upper()
            if "GATHER" in operator_type or "REDISTRIBUTE" in operator_type:
                break

            next_id = int(df.at[idx, "left_child"])
            if next_id < 0:
                break
            current_id = next_id
