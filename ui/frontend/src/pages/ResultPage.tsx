import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import EChart from '../components/EChart';
import { Kpi, KpiGrid } from '../components/Kpi';
import PlanTree from '../components/PlanTree';
import StatusBanner from '../components/StatusBanner';
import { useAppState } from '../store';
import { formatBytes, formatDuration } from '../utils/format';

export default function ResultPage() {
  const navigate = useNavigate();
  const {
    dataset,
    queryId,
    actualResult,
    setActualResult,
    optimizeResult,
  } = useAppState();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [savedId, setSavedId] = useState<number | null>(null);

  async function refresh() {
    if (!dataset || queryId === null) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.getActualResult(dataset, queryId);
      setActualResult(result);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!actualResult && dataset && queryId !== null) {
      refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function saveToHistory() {
    if (!dataset || queryId === null) return;
    setError(null);
    try {
      const detail = await api.saveHistory({
        dataset,
        query_id: queryId,
        algorithm: optimizeResult?.algorithm ?? 'pipeline',
        status: 'completed',
        params: {},
        predicted: optimizeResult
          ? {
              total_cpu_time: optimizeResult.total_cpu_time,
              max_dop: optimizeResult.max_dop,
              query_total_threads: optimizeResult.query_total_threads,
              thread_blocks: optimizeResult.thread_blocks.length,
            }
          : null,
        actual: actualResult,
        optimization_csv_path: optimizeResult?.optimization_csv_path ?? null,
        plan_info_path: actualResult?.plan_info_path ?? null,
        query_info_path: actualResult?.query_info_path ?? null,
      });
      setSavedId(detail.id);
    } catch (err) {
      setError(String(err));
    }
  }

  const opTimeChart = actualResult ? buildOperatorTimeChart(actualResult.operators) : null;
  const opDopChart = actualResult ? buildOperatorDopChart(actualResult.operators) : null;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>执行结果</h2>
          <p>读取 openGauss data_file 输出的 plan_info / query_info，展示实际指标。</p>
        </div>
        <div>
          <button onClick={refresh} disabled={loading}>
            {loading ? '加载中...' : '刷新结果'}
          </button>
          <button
            className="primary"
            style={{ marginLeft: 8 }}
            disabled={!actualResult}
            onClick={saveToHistory}
          >
            保存到历史
          </button>
        </div>
      </div>

      {(!dataset || queryId === null) && (
        <StatusBanner kind="warn">
          请先在
          <a onClick={() => navigate('/queries')}> 选择查询</a>
          页面挑好 dataset / query_id。
        </StatusBanner>
      )}
      {error && <StatusBanner kind="error">{error}</StatusBanner>}
      {savedId !== null && (
        <StatusBanner kind="info">已保存为历史记录 #{savedId}</StatusBanner>
      )}

      {!actualResult && <div className="empty">还没有读取到执行结果，点击「刷新结果」</div>}

      {actualResult && (
        <>
          <KpiGrid>
            <Kpi label="execution_time" value={formatDuration(actualResult.execution_time)} />
            <Kpi label="query_used_mem" value={formatBytes(actualResult.query_used_mem)} />
            <Kpi label="process_used_mem" value={formatBytes(actualResult.process_used_mem)} />
            <Kpi label="operator_num" value={String(actualResult.operator_num ?? '—')} />
            <Kpi label="query_dop" value={String(actualResult.query_dop ?? '—')} />
            <Kpi label="cpu_time" value={formatDuration(actualResult.cpu_time)} />
            <Kpi label="io_time" value={formatDuration(actualResult.io_time)} />
            <Kpi label="total_costs" value={String(actualResult.total_costs ?? '—')} />
          </KpiGrid>

          <div className="card">
            <h3>表参与</h3>
            <div className="muted">{actualResult.table_names ?? '—'}</div>
          </div>

          <div className="split">
            <div className="card">
              <h3>Plan Tree (按实际 DOP 着色)</h3>
              <PlanTree
                operators={actualResult.operators.map((op) => ({
                  plan_id: op.plan_id,
                  operator_type: op.operator_type,
                  width: op.width ?? 0,
                  dop: op.dop ?? 0,
                  parent_child: op.parent_child ?? -1,
                  left_child: op.left_child ?? -1,
                }))}
                highlight="dop"
              />
            </div>
            <div className="card">
              <h3>算子级 execution_time</h3>
              {opTimeChart && <EChart option={opTimeChart} />}
            </div>
          </div>

          <div className="card">
            <h3>算子级 DOP</h3>
            {opDopChart && <EChart option={opDopChart} />}
          </div>

          <div className="card">
            <h3>算子明细</h3>
            <div className="scroll-area" style={{ maxHeight: 360 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>plan_id</th>
                    <th>operator_type</th>
                    <th>dop</th>
                    <th>execution_time</th>
                    <th>total_time</th>
                    <th>actual_rows</th>
                    <th>peak_mem</th>
                  </tr>
                </thead>
                <tbody>
                  {actualResult.operators.map((op) => (
                    <tr key={op.plan_id}>
                      <td>{op.plan_id}</td>
                      <td>{op.operator_type}</td>
                      <td>{op.dop}</td>
                      <td>{formatDuration(op.execution_time)}</td>
                      <td>{formatDuration(op.total_time)}</td>
                      <td>{op.actual_rows ?? '—'}</td>
                      <td>{formatBytes(op.peak_mem)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h3>来源</h3>
            <div className="muted mono" style={{ fontSize: 11 }}>
              <div>plan_info: {actualResult.plan_info_path}</div>
              <div>query_info: {actualResult.query_info_path}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function buildOperatorTimeChart(operators: Array<{ plan_id: number; execution_time: number | null; operator_type: string }>) {
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 30, bottom: 60 },
    xAxis: {
      type: 'category',
      data: operators.map((op) => `${op.plan_id}`),
    },
    yAxis: { type: 'value', name: 'ms' },
    series: [
      {
        type: 'bar',
        name: 'execution_time',
        data: operators.map((op) => op.execution_time ?? 0),
        itemStyle: { color: '#4c8dff' },
      },
    ],
  } as Record<string, unknown>;
}

function buildOperatorDopChart(operators: Array<{ plan_id: number; dop: number }>) {
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 30, bottom: 60 },
    xAxis: {
      type: 'category',
      data: operators.map((op) => `${op.plan_id}`),
    },
    yAxis: { type: 'value', name: 'DOP' },
    series: [
      {
        type: 'bar',
        name: 'dop',
        data: operators.map((op) => op.dop ?? 0),
        itemStyle: {
          color: (params: { value: number }) => {
            const dop = params.value;
            if (dop <= 1) return '#475569';
            if (dop <= 8) return '#60a5fa';
            if (dop <= 32) return '#34d399';
            if (dop <= 64) return '#fbbf24';
            return '#f87171';
          },
        },
      },
    ],
  } as Record<string, unknown>;
}
