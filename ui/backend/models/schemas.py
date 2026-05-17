# -*- coding: utf-8 -*-
"""API schemas shared by FastAPI routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    name: str
    display_name: str
    plan_info_path: str
    query_info_path: str
    query_count: int
    database: Optional[str] = None
    sql_dir: Optional[str] = None


class QuerySummary(BaseModel):
    query_id: int
    dop: Optional[int] = None
    execution_time: Optional[float] = None
    query_used_mem: Optional[float] = None
    operator_num: Optional[int] = None
    table_names: Optional[str] = None


class OperatorNode(BaseModel):
    plan_id: int
    operator_type: str
    width: int
    dop: int
    parent_child: int
    left_child: int
    thread_block_id: Optional[int] = None


class ThreadBlockPrediction(BaseModel):
    thread_block_id: int
    optimal_dop: int
    predicted_time: Optional[float] = None
    operators: List[OperatorNode]


class QueryDetail(BaseModel):
    dataset: str
    query_id: int
    summary: QuerySummary
    operators: List[OperatorNode]
    sql_text: Optional[str] = None
    sql_path: Optional[str] = None


class OptimizeRequest(BaseModel):
    dataset: str
    query_id: int
    algorithm: str = Field(default="pipeline")
    train_mode: Optional[str] = None
    base_dop: Optional[int] = None
    use_estimates: Optional[bool] = None
    model_dataset: Optional[str] = None
    train_ratio: Optional[float] = None
    use_moo: Optional[bool] = None
    use_continuous_dop: Optional[bool] = None
    moo_population_size: Optional[int] = None
    moo_generations: Optional[int] = None
    use_pdg_moo: Optional[bool] = None
    min_improvement_ratio: Optional[float] = None
    min_reduction_threshold: Optional[int] = None
    pdg_moo_WL: Optional[float] = None
    pdg_moo_WC: Optional[float] = None
    pdg_moo_B: Optional[int] = None
    pdg_moo_T: Optional[int] = None
    pdg_moo_K: Optional[int] = None
    pdg_moo_b: Optional[int] = None
    pdg_moo_p: Optional[int] = None
    pdg_moo_lambda_I: Optional[float] = None
    force_rerun: bool = False


class OptimizeResultPayload(BaseModel):
    dataset: str
    query_id: int
    algorithm: str
    total_cpu_time: Optional[float] = None
    query_total_threads: Optional[int] = None
    max_dop: Optional[int] = None
    thread_blocks: List[ThreadBlockPrediction]
    operators: List[OperatorNode]
    optimization_csv_path: str
    optimization_json_path: str


class ExecuteRequest(BaseModel):
    dataset: str
    query_id: int
    base_dop: Optional[int] = None
    restart_gauss: bool = True
    optimization_csv_path: Optional[str] = None


class TaskInfo(BaseModel):
    task_id: str
    kind: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str


class ActualOperator(BaseModel):
    plan_id: int
    operator_type: str
    dop: int
    query_dop: Optional[int] = None
    execution_time: Optional[float] = None
    total_time: Optional[float] = None
    actual_rows: Optional[float] = None
    peak_mem: Optional[float] = None
    width: Optional[int] = None
    parent_child: Optional[int] = None
    left_child: Optional[int] = None


class ActualResult(BaseModel):
    dataset: str
    query_id: int
    query_dop: Optional[int] = None
    execution_time: Optional[float] = None
    executor_start_time: Optional[float] = None
    query_used_mem: Optional[float] = None
    operator_mem: Optional[float] = None
    process_used_mem: Optional[float] = None
    cpu_time: Optional[float] = None
    io_time: Optional[float] = None
    total_costs: Optional[float] = None
    operator_num: Optional[int] = None
    table_names: Optional[str] = None
    operators: List[ActualOperator] = Field(default_factory=list)
    plan_info_path: str
    query_info_path: str
    sourced_at: float


class HistoryRunSummary(BaseModel):
    id: int
    dataset: str
    query_id: int
    algorithm: str
    status: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    predicted_total_time: Optional[float] = None
    predicted_max_dop: Optional[int] = None
    actual_execution_time: Optional[float] = None
    actual_query_used_mem: Optional[float] = None
    actual_operator_num: Optional[int] = None
    note: Optional[str] = None


class HistoryRunDetail(HistoryRunSummary):
    params: Dict[str, Any] = Field(default_factory=dict)
    predicted: Optional[Dict[str, Any]] = None
    actual: Optional[Dict[str, Any]] = None
    optimization_csv_path: Optional[str] = None
    query_txt_path: Optional[str] = None
    plan_info_path: Optional[str] = None
    query_info_path: Optional[str] = None


class HistoryCreateRequest(BaseModel):
    dataset: str
    query_id: int
    algorithm: str
    status: str = "completed"
    params: Dict[str, Any] = Field(default_factory=dict)
    predicted: Optional[Dict[str, Any]] = None
    actual: Optional[Dict[str, Any]] = None
    optimization_csv_path: Optional[str] = None
    query_txt_path: Optional[str] = None
    plan_info_path: Optional[str] = None
    query_info_path: Optional[str] = None
    note: Optional[str] = None
