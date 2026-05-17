import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import LogStream from '../components/LogStream';
import StatusBanner from '../components/StatusBanner';
import { useAppState } from '../store';

export default function ExecutePage() {
  const navigate = useNavigate();
  const {
    dataset,
    queryId,
    optimizeResult,
    executeTaskId,
    setExecuteTaskId,
    config,
  } = useAppState();
  const [baseDop, setBaseDop] = useState<number>(64);
  const [restartGauss, setRestartGauss] = useState<boolean>(true);
  const [confirm, setConfirm] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [doneStatus, setDoneStatus] = useState<string>('idle');

  useEffect(() => {
    if (!config) return;
    const dop = config.optimize_defaults['base_dop'];
    if (typeof dop === 'number') setBaseDop(dop);
  }, [config]);

  const ready = Boolean(dataset && queryId !== null && optimizeResult);

  async function submit() {
    if (!ready) return;
    setSubmitting(true);
    setError(null);
    try {
      const { task_id } = await api.submitExecute({
        dataset,
        query_id: queryId,
        base_dop: baseDop,
        restart_gauss: restartGauss,
        optimization_csv_path: optimizeResult?.optimization_csv_path,
      });
      setExecuteTaskId(task_id);
      setDoneStatus('running');
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>执行查询</h2>
          <p>把预测出的资源配置写入 query.txt，然后调度 openGauss 实际执行 SQL。</p>
        </div>
        <div>
          <button
            disabled={doneStatus !== 'completed'}
            className="primary"
            onClick={() => navigate('/result')}
          >
            查看执行结果 →
          </button>
        </div>
      </div>

      {!ready && (
        <StatusBanner kind="warn">
          请先在
          <a onClick={() => navigate('/optimize')}> 资源预测</a>
          页面跑出 pipeline_optimization.csv。
        </StatusBanner>
      )}
      {restartGauss && (
        <StatusBanner kind="warn">
          gs_ctl restart 将重启 openGauss 整库，请确认当前没有其他人在使用该实例。
        </StatusBanner>
      )}
      {error && <StatusBanner kind="error">{error}</StatusBanner>}

      <div className="card">
        <h3>执行参数</h3>
        <div className="toolbar">
          <label>
            dataset
            <span className="mono">{dataset ?? '—'}</span>
          </label>
          <label>
            query_id
            <span className="mono">{queryId ?? '—'}</span>
          </label>
          <label>
            base_dop
            <input
              type="number"
              min={1}
              value={baseDop}
              onChange={(e) => setBaseDop(parseInt(e.target.value || '64', 10))}
              style={{ width: 70 }}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={restartGauss}
              onChange={(e) => setRestartGauss(e.target.checked)}
            />
            重启 openGauss
          </label>
          <label>
            <input
              type="checkbox"
              checked={confirm}
              onChange={(e) => setConfirm(e.target.checked)}
            />
            我已确认可以执行
          </label>
          <div style={{ marginLeft: 'auto' }}>
            <button className="primary" disabled={!ready || !confirm || submitting} onClick={submit}>
              {submitting ? '提交中...' : executeTaskId ? '重新执行' : '开始执行'}
            </button>
          </div>
        </div>
        {config && (
          <div className="muted mono" style={{ fontSize: 11, marginTop: 6 }}>
            <div>query.txt → {config.query_txt_path}</div>
            <div>data_file → {config.data_file_dir}</div>
            <div>gauss data → {config.gauss_data_dir}</div>
          </div>
        )}
      </div>

      <LogStream
        taskId={executeTaskId}
        onStatusChange={setDoneStatus}
        onError={(err) => setError(err)}
        maxHeight={420}
      />
    </div>
  );
}
