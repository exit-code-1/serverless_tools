# -*- coding: utf-8 -*-
"""SQLite-backed run history. Stores parameters, predicted summary and the
actual execution metrics so multiple runs can be compared in the UI."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from ..settings import get_settings


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    query_id INTEGER NOT NULL,
    algorithm TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL,
    finished_at REAL,
    params_json TEXT,
    predicted_summary_json TEXT,
    actual_summary_json TEXT,
    optimization_csv_path TEXT,
    query_txt_path TEXT,
    plan_info_path TEXT,
    query_info_path TEXT,
    predicted_total_time REAL,
    predicted_max_dop INTEGER,
    actual_execution_time REAL,
    actual_query_used_mem REAL,
    actual_operator_num INTEGER,
    note TEXT
);
"""


class HistoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def add_run(self, payload: Dict[str, Any]) -> int:
        predicted = payload.get("predicted") or {}
        actual = payload.get("actual") or {}
        params = payload.get("params") or {}
        started_at = payload.get("started_at") or time.time()
        finished_at = payload.get("finished_at") or time.time()

        predicted_total_time = predicted.get("total_cpu_time")
        predicted_max_dop = predicted.get("max_dop")
        actual_execution_time = actual.get("execution_time")
        actual_query_used_mem = actual.get("query_used_mem")
        actual_operator_num = actual.get("operator_num")

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (
                    dataset, query_id, algorithm, status,
                    started_at, finished_at,
                    params_json, predicted_summary_json, actual_summary_json,
                    optimization_csv_path, query_txt_path, plan_info_path, query_info_path,
                    predicted_total_time, predicted_max_dop,
                    actual_execution_time, actual_query_used_mem, actual_operator_num,
                    note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["dataset"],
                    int(payload["query_id"]),
                    payload.get("algorithm", "pipeline"),
                    payload.get("status", "completed"),
                    started_at,
                    finished_at,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(predicted, ensure_ascii=False) if predicted else None,
                    json.dumps(actual, ensure_ascii=False) if actual else None,
                    payload.get("optimization_csv_path"),
                    payload.get("query_txt_path"),
                    payload.get("plan_info_path"),
                    payload.get("query_info_path"),
                    predicted_total_time,
                    predicted_max_dop,
                    actual_execution_time,
                    actual_query_used_mem,
                    actual_operator_num,
                    payload.get("note"),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_runs(
        self,
        *,
        dataset: Optional[str] = None,
        query_id: Optional[int] = None,
        algorithm: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if dataset:
            clauses.append("dataset = ?")
            params.append(dataset)
        if query_id is not None:
            clauses.append("query_id = ?")
            params.append(int(query_id))
        if algorithm:
            clauses.append("algorithm = ?")
            params.append(algorithm)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY id DESC LIMIT ?",
                (*params, int(limit)),
            )
            return [self._row_to_summary(row) for row in cur.fetchall()]

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM runs WHERE id = ?", (int(run_id),))
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_detail(row)

    def delete_run(self, run_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
            conn.commit()
            return cur.rowcount > 0

    def _row_to_summary(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "dataset": row["dataset"],
            "query_id": row["query_id"],
            "algorithm": row["algorithm"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "predicted_total_time": row["predicted_total_time"],
            "predicted_max_dop": row["predicted_max_dop"],
            "actual_execution_time": row["actual_execution_time"],
            "actual_query_used_mem": row["actual_query_used_mem"],
            "actual_operator_num": row["actual_operator_num"],
            "note": row["note"],
        }

    def _row_to_detail(self, row: sqlite3.Row) -> Dict[str, Any]:
        base = self._row_to_summary(row)
        base.update({
            "params": _safe_json_load(row["params_json"]) or {},
            "predicted": _safe_json_load(row["predicted_summary_json"]),
            "actual": _safe_json_load(row["actual_summary_json"]),
            "optimization_csv_path": row["optimization_csv_path"],
            "query_txt_path": row["query_txt_path"],
            "plan_info_path": row["plan_info_path"],
            "query_info_path": row["query_info_path"],
        })
        return base


def _safe_json_load(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


_history_store_singleton: Optional[HistoryStore] = None


def get_history_store() -> HistoryStore:
    global _history_store_singleton
    if _history_store_singleton is None:
        settings = get_settings()
        _history_store_singleton = HistoryStore(settings.history_db_path)
    return _history_store_singleton
