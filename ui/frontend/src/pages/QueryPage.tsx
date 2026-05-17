import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { Kpi, KpiGrid } from '../components/Kpi';
import PlanTree from '../components/PlanTree';
import StatusBanner from '../components/StatusBanner';
import { useAppState } from '../store';
import type { DatasetInfo, QueryDetail, QuerySummary } from '../types';
import { formatBytes, formatDuration } from '../utils/format';

export default function QueryPage() {
  const navigate = useNavigate();
  const { dataset, setDataset, queryId, setQueryId, queryDetail, setQueryDetail } = useAppState();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [queries, setQueries] = useState<QuerySummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listDatasets()
      .then((items) => {
        setDatasets(items);
        if (!dataset && items.length > 0) {
          const first = items.find((d) => d.query_count > 0) ?? items[0];
          setDataset(first.name);
        }
      })
      .catch((err) => setError(String(err)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!dataset) return;
    setLoading(true);
    setError(null);
    api
      .listQueries(dataset)
      .then(setQueries)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [dataset]);

  useEffect(() => {
    if (!dataset || queryId === null) return;
    api
      .getQueryDetail(dataset, queryId)
      .then((detail: QueryDetail) => setQueryDetail(detail))
      .catch((err) => setError(String(err)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, queryId]);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>选择查询</h2>
          <p>从训练/测试数据集中挑选 query_id，预览执行计划和 SQL。</p>
        </div>
      </div>

      {error && <StatusBanner kind="error">加载失败: {error}</StatusBanner>}

      <div className="toolbar">
        <label>
          数据集
          <select
            value={dataset ?? ''}
            onChange={(e) => setDataset(e.target.value || null)}
          >
            <option value="" disabled>
              选择数据集
            </option>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.display_name} ({d.query_count})
              </option>
            ))}
          </select>
        </label>
        <span className="muted">
          {dataset && datasets.find((d) => d.name === dataset)?.database
            ? `数据库: ${datasets.find((d) => d.name === dataset)?.database}`
            : null}
        </span>
        <div style={{ marginLeft: 'auto' }}>
          <button
            className="primary"
            disabled={queryId === null}
            onClick={() => navigate('/optimize')}
          >
            进入资源预测 →
          </button>
        </div>
      </div>

      <div className="split">
        <div className="card">
          <h3>查询列表 {loading ? <span className="muted">加载中...</span> : null}</h3>
          <div className="scroll-area" style={{ maxHeight: 520 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>query_id</th>
                  <th>dop</th>
                  <th>execution_time</th>
                  <th>used_mem</th>
                  <th>operator_num</th>
                  <th>tables</th>
                </tr>
              </thead>
              <tbody>
                {queries.map((q) => (
                  <tr
                    key={q.query_id}
                    className={queryId === q.query_id ? 'selected' : undefined}
                    onClick={() => setQueryId(q.query_id)}
                  >
                    <td>
                      <span className="mono">{q.query_id}</span>
                    </td>
                    <td>{q.dop ?? '—'}</td>
                    <td>{formatDuration(q.execution_time)}</td>
                    <td>{formatBytes(q.query_used_mem)}</td>
                    <td>{q.operator_num ?? '—'}</td>
                    <td className="muted" style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {q.table_names ?? '—'}
                    </td>
                  </tr>
                ))}
                {queries.length === 0 && (
                  <tr>
                    <td colSpan={6} className="muted" style={{ padding: 24, textAlign: 'center' }}>
                      暂无数据
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <h3>查询详情</h3>
          {!queryDetail && <div className="empty">从左侧选择 query_id 后查看 Plan Tree</div>}
          {queryDetail && (
            <div className="flex-col">
              <KpiGrid>
                <Kpi label="query_id" value={String(queryDetail.query_id)} />
                <Kpi label="query_dop" value={String(queryDetail.summary.dop ?? '—')} />
                <Kpi label="execution_time" value={formatDuration(queryDetail.summary.execution_time)} />
                <Kpi label="used_mem" value={formatBytes(queryDetail.summary.query_used_mem)} />
                <Kpi label="operators" value={String(queryDetail.summary.operator_num ?? '—')} />
              </KpiGrid>
              <PlanTree operators={queryDetail.operators} highlight="dop" />
              {queryDetail.sql_text && (
                <div>
                  <h3 style={{ marginBottom: 6 }}>SQL</h3>
                  {queryDetail.sql_path && (
                    <div className="muted mono" style={{ marginBottom: 6, fontSize: 11 }}>
                      {queryDetail.sql_path}
                    </div>
                  )}
                  <textarea
                    readOnly
                    value={queryDetail.sql_text}
                    style={{ width: '100%', minHeight: 160 }}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
