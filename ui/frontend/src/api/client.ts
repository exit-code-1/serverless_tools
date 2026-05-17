import type {
  ActualResult,
  BackendConfig,
  DatasetInfo,
  HistoryRunDetail,
  HistoryRunSummary,
  OptimizeResultPayload,
  QueryDetail,
  QuerySummary,
  TaskInfo,
} from '../types';

const BASE = '';

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => jsonRequest<{ status: string }>(`/api/health`),
  config: () => jsonRequest<BackendConfig>(`/api/config`),
  listDatasets: () => jsonRequest<DatasetInfo[]>(`/api/datasets`),
  listQueries: (dataset: string) =>
    jsonRequest<QuerySummary[]>(`/api/datasets/${dataset}/queries`),
  getQueryDetail: (dataset: string, queryId: number) =>
    jsonRequest<QueryDetail>(`/api/datasets/${dataset}/queries/${queryId}`),
  submitOptimize: (payload: Record<string, unknown>) =>
    jsonRequest<{ task_id: string; status: string }>(`/api/optimize`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getOptimizeStatus: (taskId: string) => jsonRequest<TaskInfo>(`/api/optimize/${taskId}`),
  getOptimizeResult: (taskId: string) =>
    jsonRequest<OptimizeResultPayload>(`/api/optimize/${taskId}/result`),
  submitExecute: (payload: Record<string, unknown>) =>
    jsonRequest<{ task_id: string; status: string }>(`/api/execute`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getExecuteStatus: (taskId: string) => jsonRequest<TaskInfo>(`/api/execute/${taskId}`),
  getActualResult: (dataset: string, queryId: number) =>
    jsonRequest<ActualResult>(`/api/results/${dataset}/${queryId}`),
  listHistory: (params: { dataset?: string; query_id?: number; algorithm?: string } = {}) => {
    const qs = new URLSearchParams();
    if (params.dataset) qs.set('dataset', params.dataset);
    if (params.query_id !== undefined) qs.set('query_id', String(params.query_id));
    if (params.algorithm) qs.set('algorithm', params.algorithm);
    const suffix = qs.toString();
    return jsonRequest<HistoryRunSummary[]>(`/api/history${suffix ? `?${suffix}` : ''}`);
  },
  getHistoryDetail: (id: number) => jsonRequest<HistoryRunDetail>(`/api/history/${id}`),
  saveHistory: (payload: Record<string, unknown>) =>
    jsonRequest<HistoryRunDetail>(`/api/history`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteHistory: (id: number) =>
    jsonRequest<{ deleted: number }>(`/api/history/${id}`, { method: 'DELETE' }),
  taskStreamUrl: (taskId: string) => `${BASE}/api/tasks/${taskId}/stream`,
  taskLogsUrl: (taskId: string) => `${BASE}/api/tasks/${taskId}/logs`,
};
