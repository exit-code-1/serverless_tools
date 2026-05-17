export interface DatasetInfo {
  name: string;
  display_name: string;
  plan_info_path: string;
  query_info_path: string;
  query_count: number;
  database: string | null;
  sql_dir: string | null;
}

export interface QuerySummary {
  query_id: number;
  dop: number | null;
  execution_time: number | null;
  query_used_mem: number | null;
  operator_num: number | null;
  table_names: string | null;
}

export interface OperatorNode {
  plan_id: number;
  operator_type: string;
  width: number;
  dop: number;
  parent_child: number;
  left_child: number;
  thread_block_id?: number | null;
}

export interface QueryDetail {
  dataset: string;
  query_id: number;
  summary: QuerySummary;
  operators: OperatorNode[];
  sql_text: string | null;
  sql_path: string | null;
}

export interface ThreadBlockPrediction {
  thread_block_id: number;
  optimal_dop: number;
  predicted_time: number | null;
  operators: OperatorNode[];
}

export interface OptimizeResultPayload {
  dataset: string;
  query_id: number;
  algorithm: string;
  total_cpu_time: number | null;
  query_total_threads: number | null;
  max_dop: number | null;
  thread_blocks: ThreadBlockPrediction[];
  operators: OperatorNode[];
  optimization_csv_path: string;
  optimization_json_path: string;
}

export interface TaskInfo {
  task_id: string;
  kind: string;
  status: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  metadata: Record<string, unknown>;
}

export interface ActualOperator {
  plan_id: number;
  operator_type: string;
  dop: number;
  query_dop: number | null;
  execution_time: number | null;
  total_time: number | null;
  actual_rows: number | null;
  peak_mem: number | null;
  width: number | null;
  parent_child: number | null;
  left_child: number | null;
}

export interface ActualResult {
  dataset: string;
  query_id: number;
  query_dop: number | null;
  execution_time: number | null;
  executor_start_time: number | null;
  query_used_mem: number | null;
  operator_mem: number | null;
  process_used_mem: number | null;
  cpu_time: number | null;
  io_time: number | null;
  total_costs: number | null;
  operator_num: number | null;
  table_names: string | null;
  operators: ActualOperator[];
  plan_info_path: string;
  query_info_path: string;
  sourced_at: number;
}

export interface HistoryRunSummary {
  id: number;
  dataset: string;
  query_id: number;
  algorithm: string;
  status: string;
  started_at: number | null;
  finished_at: number | null;
  predicted_total_time: number | null;
  predicted_max_dop: number | null;
  actual_execution_time: number | null;
  actual_query_used_mem: number | null;
  actual_operator_num: number | null;
  note: string | null;
}

export interface HistoryRunDetail extends HistoryRunSummary {
  params: Record<string, unknown>;
  predicted: Record<string, unknown> | null;
  actual: Record<string, unknown> | null;
  optimization_csv_path: string | null;
  query_txt_path: string | null;
  plan_info_path: string | null;
  query_info_path: string | null;
}

export interface BackendConfig {
  predictor_root: string;
  query_txt_path: string;
  data_file_dir: string;
  gauss_data_dir: string;
  databases: Record<string, string>;
  sql_dirs: Record<string, string>;
  optimize_defaults: Record<string, unknown>;
}
