# -*- coding: utf-8 -*-
"""Wraps the project's optimization entry points and slices their JSON output
to a single-query payload that the UI can render."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from ..settings import Settings, get_settings
from .task_manager import TaskManager


_OPTIMIZE_LOCK = threading.Lock()


def _ensure_predictor_on_path(settings: Settings) -> None:
    root = settings.predictor_root
    if root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def _redirect_stdio(task_manager: TaskManager, task_id: str):
    """Pipe stdout/stderr line-by-line into the task log stream."""

    class _Writer:
        def __init__(self, stream_name: str) -> None:
            self._buffer = ""
            self._stream_name = stream_name

        def write(self, text: str) -> int:
            if not text:
                return 0
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                task_manager.append_log(task_id, line)
            return len(text)

        def flush(self) -> None:
            if self._buffer:
                task_manager.append_log(task_id, self._buffer)
                self._buffer = ""

    saved_out, saved_err = sys.stdout, sys.stderr
    out_writer, err_writer = _Writer("stdout"), _Writer("stderr")
    sys.stdout, sys.stderr = out_writer, err_writer
    try:
        yield
    finally:
        try:
            out_writer.flush()
            err_writer.flush()
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err


def _resolve_algorithm_paths(settings: Settings, dataset: str, train_mode: str, algorithm: str) -> Tuple[str, str]:
    """Return (json_path, csv_path) where the optimization output should live."""
    _ensure_predictor_on_path(settings)
    from utils import get_output_paths

    if algorithm == "baseline":
        paths = get_output_paths(dataset, "dop_aware", train_mode)
        # baseline filename includes base_dop, but we resolve dynamically below.
        return paths["optimization_dir"], ""
    if algorithm == "pipeline":
        paths = get_output_paths(dataset, "dop_aware", train_mode)
        json_path = os.path.join(paths["optimization_dir"], "pipeline_optimization.json")
        return json_path, json_path.replace(".json", ".csv")
    if algorithm in {"query_level", "auto_dop", "ppm"}:
        paths = get_output_paths(dataset, "query_level", train_mode)
        json_path = os.path.join(paths["optimization_dir"], f"query_level_optimization_{algorithm}.json")
        return json_path, json_path.replace(".json", ".csv")
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_optimization(
    *,
    task_manager: TaskManager,
    task_id: str,
    dataset: str,
    query_id: int,
    algorithm: str,
    params: Dict[str, Any],
    force_rerun: bool,
) -> Dict[str, Any]:
    """Synchronous worker invoked from a background thread.

    Returns the per-query optimization payload (already saved to disk under
    `ui/runs/<task_id>/`)."""

    settings = get_settings()
    _ensure_predictor_on_path(settings)
    task_manager.mark_started(task_id)

    defaults = settings.optimize_defaults
    merged = {**defaults, **{k: v for k, v in params.items() if v is not None}}
    train_mode = merged.get("train_mode", "exact_train")

    task_manager.append_log(
        task_id,
        f"[optimize] dataset={dataset} query_id={query_id} algorithm={algorithm} train_mode={train_mode}",
    )

    json_path, csv_path = _resolve_algorithm_paths(settings, dataset, train_mode, algorithm)

    need_run = force_rerun or not os.path.exists(json_path)
    task_manager.append_log(
        task_id,
        f"[optimize] expected json={json_path} exists={os.path.exists(json_path)} force_rerun={force_rerun}",
    )

    if need_run:
        with _OPTIMIZE_LOCK:
            with _redirect_stdio(task_manager, task_id):
                _invoke_optimizer(algorithm, dataset, train_mode, merged)
    else:
        task_manager.append_log(task_id, "[optimize] reusing cached optimization output")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Optimization output missing after run: {json_path}")
    if not os.path.exists(csv_path):
        # The CSV is exported alongside JSON, but be defensive.
        from scripts.optimize import export_optimization_to_csv  # type: ignore

        export_optimization_to_csv(json_path, csv_path)

    payload = extract_query_payload(json_path, csv_path, query_id, algorithm, dataset)
    _persist_run(task_id, settings, payload)
    task_manager.append_log(task_id, "[optimize] done")
    return payload


def _invoke_optimizer(algorithm: str, dataset: str, train_mode: str, merged: Dict[str, Any]) -> None:
    from scripts.optimize import (  # type: ignore
        run_baseline_optimization,
        run_pipeline_optimization,
        run_query_level_optimization,
    )

    model_dataset = merged.get("model_dataset") or dataset
    use_estimates = bool(merged.get("use_estimates", False))
    train_ratio = merged.get("train_ratio")
    base_dop = int(merged.get("base_dop", 64))

    if algorithm == "baseline":
        ok = run_baseline_optimization(
            dataset,
            train_mode,
            base_dop=base_dop,
            use_estimates=use_estimates,
            model_dataset=model_dataset,
            train_ratio=train_ratio,
        )
        if not ok:
            raise RuntimeError("run_baseline_optimization returned failure")
        return
    if algorithm == "pipeline":
        ok = run_pipeline_optimization(
            dataset,
            train_mode,
            base_dop=base_dop,
            min_improvement_ratio=float(merged.get("min_improvement_ratio", 0.1)),
            min_reduction_threshold=int(merged.get("min_reduction_threshold", 200)),
            use_estimates=use_estimates,
            interval_tolerance=float(merged.get("interval_tolerance", 0.3)),
            use_moo=bool(merged.get("use_moo", True)),
            use_continuous_dop=bool(merged.get("use_continuous_dop", True)),
            moo_population_size=int(merged.get("moo_population_size", 30)),
            moo_generations=int(merged.get("moo_generations", 20)),
            use_pdg_moo=bool(merged.get("use_pdg_moo", True)),
            pdg_moo_WL=float(merged.get("pdg_moo_WL", 0.7)),
            pdg_moo_WC=float(merged.get("pdg_moo_WC", 0.3)),
            pdg_moo_B=int(merged.get("pdg_moo_B", 20)),
            pdg_moo_T=int(merged.get("pdg_moo_T", 20)),
            pdg_moo_K=int(merged.get("pdg_moo_K", 5)),
            pdg_moo_b=int(merged.get("pdg_moo_b", 4)),
            pdg_moo_p=int(merged.get("pdg_moo_p", 15)),
            pdg_moo_lambda_I=float(merged.get("pdg_moo_lambda_I", 0.1)),
            model_dataset=model_dataset,
            train_ratio=train_ratio,
        )
        if not ok:
            raise RuntimeError("run_pipeline_optimization returned failure")
        return
    if algorithm in {"query_level", "auto_dop", "ppm"}:
        ok = run_query_level_optimization(
            dataset,
            algorithm,
            train_mode,
            base_dop=base_dop,
            use_estimates=use_estimates,
            model_dataset=model_dataset,
            train_ratio=train_ratio,
        )
        if not ok:
            raise RuntimeError(f"run_query_level_optimization({algorithm}) returned failure")
        return
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def extract_query_payload(
    json_path: str,
    csv_path: str,
    query_id: int,
    algorithm: str,
    dataset: str,
) -> Dict[str, Any]:
    """Slice the optimization JSON to a single query and merge operator-level DOP
    from the CSV view."""

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    target: Optional[Dict[str, Any]] = None
    for query in data.get("queries", []):
        if int(query.get("query_id", -1)) == int(query_id):
            target = query
            break
    if target is None:
        raise KeyError(f"query_id {query_id} not found in optimization JSON: {json_path}")

    thread_blocks: List[Dict[str, Any]] = []
    flat_operators: List[Dict[str, Any]] = []
    settings = get_settings()
    dop_cap = settings.dop_cap
    for tb in target.get("thread_blocks", []):
        tb_id = int(tb.get("thread_block_id", 0))
        dop = int(tb.get("optimal_dop", 1))
        if dop_cap > 0:
            dop = min(dop, dop_cap)
        ops = []
        for op in tb.get("operators", []):
            entry = {
                "plan_id": int(op["plan_id"]),
                "operator_type": str(op["operator_type"]),
                "width": int(op.get("width", 0)),
                "dop": dop,
                "parent_child": int(op.get("parent_child", -1)),
                "left_child": int(op.get("left_child", -1)),
                "thread_block_id": tb_id,
            }
            ops.append(entry)
            flat_operators.append(entry)
        thread_blocks.append({
            "thread_block_id": tb_id,
            "optimal_dop": dop,
            "predicted_time": _safe_float(tb.get("predicted_time")),
            "operators": ops,
        })
    _force_root_pipeline_payload_dop_one(thread_blocks, flat_operators)
    flat_operators.sort(key=lambda x: x["plan_id"])
    max_dop = _safe_int(target.get("max_dop"))
    if dop_cap > 0 and max_dop is not None:
        max_dop = min(max_dop, dop_cap)

    return {
        "dataset": dataset,
        "query_id": int(query_id),
        "algorithm": algorithm,
        "total_cpu_time": _safe_float(target.get("total_cpu_time")),
        "query_total_threads": _safe_int(target.get("query_total_threads")),
        "max_dop": max_dop,
        "thread_blocks": thread_blocks,
        "operators": flat_operators,
        "optimization_csv_path": csv_path,
        "optimization_json_path": json_path,
    }


def _persist_run(task_id: str, settings: Settings, payload: Dict[str, Any]) -> None:
    run_dir = os.path.join(settings.runs_dir, task_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "optimize_payload.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def load_persisted_payload(task_id: str) -> Optional[Dict[str, Any]]:
    settings = get_settings()
    target = os.path.join(settings.runs_dir, task_id, "optimize_payload.json")
    if not os.path.exists(target):
        return None
    with open(target, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _force_root_pipeline_payload_dop_one(
    thread_blocks: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
) -> None:
    by_plan_id = {int(op["plan_id"]): op for op in operators}
    root_ids = [
        int(op["plan_id"])
        for op in operators
        if int(op.get("parent_child", -1)) < 0
    ]
    for root_id in root_ids:
        visited = set()
        current_id = root_id
        while current_id in by_plan_id and current_id not in visited:
            visited.add(current_id)
            op = by_plan_id[current_id]
            op["dop"] = 1

            operator_type = str(op.get("operator_type", "")).upper()
            if "GATHER" in operator_type or "REDISTRIBUTE" in operator_type:
                break

            next_id = int(op.get("left_child", -1))
            if next_id < 0:
                break
            current_id = next_id

    for tb in thread_blocks:
        ops = tb.get("operators", [])
        if ops:
            tb["optimal_dop"] = max(int(op.get("dop", 1)) for op in ops)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check
        return None
    return f


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
