import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';
import StatusBanner from '../components/StatusBanner';
import type { HistoryRunDetail, HistoryRunSummary } from '../types';
import { formatBytes, formatDuration, formatTimestamp } from '../utils/format';

export default function HistoryPage() {
  const [rows, setRows] = useState<HistoryRunSummary[]>([]);
  const [filterDataset, setFilterDataset] = useState<string>('');
  const [filterQuery, setFilterQuery] = useState<string>('');
  const [filterAlgorithm, setFilterAlgorithm] = useState<string>('');
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [details, setDetails] = useState<Record<number, HistoryRunDetail>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listHistory({
        dataset: filterDataset || undefined,
        query_id: filterQuery ? Number(filterQuery) : undefined,
        algorithm: filterAlgorithm || undefined,
      });
      setRows(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    selectedIds.forEach(async (id) => {
      if (details[id]) return;
      try {
        const detail = await api.getHistoryDetail(id);
        setDetails((prev) => ({ ...prev, [id]: detail }));
      } catch (err) {
        setError(String(err));
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIds]);

  const selectedDetails = useMemo(
    () => selectedIds.map((id) => details[id]).filter(Boolean) as HistoryRunDetail[],
    [selectedIds, details],
  );

  function toggleSelected(id: number) {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  }

  async function handleDelete(id: number) {
    if (!confirm(`确认删除历史记录 #${id} ？`)) return;
    try {
      await api.deleteHistory(id);
      setSelectedIds((prev) => prev.filter((x) => x !== id));
      setDetails((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>历史记录</h2>
          <p>跨 query / 算法对比每次跑出来的预测和实际指标。</p>
        </div>
      </div>

      {error && <StatusBanner kind="error">{error}</StatusBanner>}

      <div className="toolbar">
        <label>
          dataset
          <input
            value={filterDataset}
            onChange={(e) => setFilterDataset(e.target.value)}
            placeholder="tpch / tpcds"
            style={{ width: 100 }}
          />
        </label>
        <label>
          query_id
          <input
            type="number"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="all"
            style={{ width: 80 }}
          />
        </label>
        <label>
          algorithm
          <input
            value={filterAlgorithm}
            onChange={(e) => setFilterAlgorithm(e.target.value)}
            placeholder="pipeline"
            style={{ width: 120 }}
          />
        </label>
        <button onClick={refresh} disabled={loading}>
          {loading ? '加载中...' : '刷新'}
        </button>
        <span className="muted" style={{ marginLeft: 'auto' }}>
          勾选两条以上做横向对比
        </span>
      </div>

      <div className="card">
        <h3>记录列表</h3>
        <div className="scroll-area" style={{ maxHeight: 360 }}>
          <table className="data">
            <thead>
              <tr>
                <th></th>
                <th>id</th>
                <th>dataset</th>
                <th>query_id</th>
                <th>algorithm</th>
                <th>status</th>
                <th>predicted_time</th>
                <th>actual_time</th>
                <th>actual_mem</th>
                <th>finished_at</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className={selectedIds.includes(row.id) ? 'selected' : undefined}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(row.id)}
                      onChange={() => toggleSelected(row.id)}
                    />
                  </td>
                  <td>{row.id}</td>
                  <td>{row.dataset}</td>
                  <td>{row.query_id}</td>
                  <td>{row.algorithm}</td>
                  <td>
                    <span className={`tag ${row.status === 'completed' ? 'success' : 'warn'}`}>
                      {row.status}
                    </span>
                  </td>
                  <td>{formatDuration(row.predicted_total_time)}</td>
                  <td>{formatDuration(row.actual_execution_time)}</td>
                  <td>{formatBytes(row.actual_query_used_mem)}</td>
                  <td>{formatTimestamp(row.finished_at)}</td>
                  <td>
                    <button onClick={() => handleDelete(row.id)}>删除</button>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={11} className="muted" style={{ textAlign: 'center', padding: 24 }}>
                    暂无历史记录
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {selectedDetails.length > 0 && (
        <div className="card">
          <h3>横向对比</h3>
          <div className="scroll-area" style={{ maxHeight: 380 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>指标</th>
                  {selectedDetails.map((d) => (
                    <th key={d.id}>
                      #{d.id} · {d.dataset}/q{d.query_id} · {d.algorithm}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <CompareRow
                  label="status"
                  values={selectedDetails.map((d) => d.status ?? '—')}
                />
                <CompareRow
                  label="predicted_total_time"
                  values={selectedDetails.map((d) => formatDuration(d.predicted_total_time))}
                />
                <CompareRow
                  label="predicted_max_dop"
                  values={selectedDetails.map((d) => String(d.predicted_max_dop ?? '—'))}
                />
                <CompareRow
                  label="actual_execution_time"
                  values={selectedDetails.map((d) => formatDuration(d.actual_execution_time))}
                />
                <CompareRow
                  label="actual_query_used_mem"
                  values={selectedDetails.map((d) => formatBytes(d.actual_query_used_mem))}
                />
                <CompareRow
                  label="actual_operator_num"
                  values={selectedDetails.map((d) => String(d.actual_operator_num ?? '—'))}
                />
                <CompareRow
                  label="finished_at"
                  values={selectedDetails.map((d) => formatTimestamp(d.finished_at))}
                />
                <CompareRow
                  label="opt csv"
                  values={selectedDetails.map((d) => d.optimization_csv_path ?? '—')}
                />
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function CompareRow({ label, values }: { label: string; values: string[] }) {
  return (
    <tr>
      <td className="muted">{label}</td>
      {values.map((v, idx) => (
        <td key={idx} className="mono" style={{ fontSize: 11 }}>
          {v}
        </td>
      ))}
    </tr>
  );
}
