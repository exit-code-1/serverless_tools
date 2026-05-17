import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import EChart from '../components/EChart';
import { Kpi, KpiGrid } from '../components/Kpi';
import StatusBanner from '../components/StatusBanner';
import { useAppState } from '../store';
import { formatBytes, formatDuration } from '../utils/format';

export default function MetricsPage() {
  const navigate = useNavigate();
  const { actualResult, optimizeResult, queryDetail } = useAppState();

  const dopDistribution = useMemo(() => {
    if (!actualResult) return null;
    const counts = new Map<number, number>();
    actualResult.operators.forEach((op) => {
      counts.set(op.dop, (counts.get(op.dop) ?? 0) + 1);
    });
    const sorted = [...counts.entries()].sort((a, b) => a[0] - b[0]);
    return {
      tooltip: {},
      grid: { left: 40, right: 20, top: 30, bottom: 40 },
      xAxis: {
        type: 'category',
        data: sorted.map(([dop]) => `dop=${dop}`),
      },
      yAxis: { type: 'value' },
      series: [
        {
          type: 'bar',
          data: sorted.map(([, count]) => count),
          itemStyle: { color: '#34d399' },
        },
      ],
    };
  }, [actualResult]);

  const operatorTimePareto = useMemo(() => {
    if (!actualResult) return null;
    const sorted = [...actualResult.operators]
      .map((op) => ({ key: `${op.plan_id} ${op.operator_type}`, value: op.execution_time ?? 0 }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 15);
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 200, right: 30, top: 20, bottom: 30 },
      yAxis: {
        type: 'category',
        data: sorted.map((s) => s.key).reverse(),
      },
      xAxis: { type: 'value', name: 'ms' },
      series: [
        {
          type: 'bar',
          data: sorted.map((s) => s.value).reverse(),
          itemStyle: { color: '#4c8dff' },
        },
      ],
    };
  }, [actualResult]);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>指标面板</h2>
          <p>聚合实际执行的关键指标。Q-error 等预测对比的视图保留在未来的开关里。</p>
        </div>
      </div>

      {!actualResult && (
        <StatusBanner kind="warn">
          请先到
          <a onClick={() => navigate('/result')}> 执行结果</a>
          页面加载实际指标。
        </StatusBanner>
      )}

      {actualResult && (
        <>
          <KpiGrid>
            <Kpi label="execution_time" value={formatDuration(actualResult.execution_time)} />
            <Kpi label="query_used_mem" value={formatBytes(actualResult.query_used_mem)} />
            <Kpi label="operator_num" value={String(actualResult.operator_num ?? '—')} />
            <Kpi label="cpu_time" value={formatDuration(actualResult.cpu_time)} />
            <Kpi
              label="预测 max_dop"
              value={String(optimizeResult?.max_dop ?? '—')}
            />
            <Kpi
              label="预测 total_cpu_time"
              value={formatDuration(optimizeResult?.total_cpu_time ?? null)}
            />
            <Kpi
              label="thread_blocks"
              value={String(optimizeResult?.thread_blocks.length ?? '—')}
            />
            <Kpi
              label="操作员行数"
              value={String(actualResult.operators.length)}
            />
          </KpiGrid>

          <div className="split">
            {dopDistribution && (
              <div className="card">
                <h3>实际 DOP 分布</h3>
                <EChart option={dopDistribution} />
              </div>
            )}
            {operatorTimePareto && (
              <div className="card">
                <h3>耗时 Top 算子</h3>
                <EChart option={operatorTimePareto} height={420} />
              </div>
            )}
          </div>

          <div className="card">
            <h3>对比上下文</h3>
            <div className="muted mono" style={{ fontSize: 11 }}>
              <div>dataset: {actualResult.dataset}</div>
              <div>query_id: {actualResult.query_id}</div>
              {queryDetail?.summary?.table_names && (
                <div>tables: {queryDetail.summary.table_names}</div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
