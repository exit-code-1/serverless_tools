# -*- coding: utf-8 -*-
"""Convert a per-query slice of pipeline_optimization.csv into the query.txt
file consumed by openGauss. Mirrors the helper in the user's batch script."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


COLUMNS_TO_KEEP = ["plan_id", "operator_type", "width", "dop", "left_child", "parent_child"]


def generate_query_file(input_csv: str, query_id: int, output_path: str) -> str:
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
    out_df = out_df.sort_values("plan_id")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def preview_query_file(input_csv: str, query_id: int) -> str:
    """Return the would-be content as a string without writing to disk."""
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    filtered = df[df["query_id"].astype(int) == int(query_id)][COLUMNS_TO_KEEP]
    return filtered.sort_values("plan_id").to_csv(index=False)


def read_query_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()
