import { useEffect } from 'react';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { api } from './api/client';
import { AppStateProvider, useAppState } from './store';
import QueryPage from './pages/QueryPage';
import OptimizePage from './pages/OptimizePage';
import ExecutePage from './pages/ExecutePage';
import ResultPage from './pages/ResultPage';
import MetricsPage from './pages/MetricsPage';
import HistoryPage from './pages/HistoryPage';

function Sidebar() {
  const { dataset, queryId, config } = useAppState();
  const navItem = ({ isActive }: { isActive: boolean }) => (isActive ? 'active' : undefined);

  return (
    <aside className="sidebar">
      <div className="brand">
        Serverless Predictor
        <small>资源预测可视化控制台</small>
      </div>
      <nav>
        <NavLink to="/queries" className={navItem}>
          1. 选择查询
        </NavLink>
        <NavLink to="/optimize" className={navItem}>
          2. 资源预测
        </NavLink>
        <NavLink to="/execute" className={navItem}>
          3. 执行查询
        </NavLink>
        <NavLink to="/result" className={navItem}>
          4. 执行结果
        </NavLink>
        <NavLink to="/metrics" className={navItem}>
          指标面板
        </NavLink>
        <NavLink to="/history" className={navItem}>
          历史记录
        </NavLink>
      </nav>
      <div className="footer">
        <div>
          dataset: <span className="mono">{dataset ?? '—'}</span>
        </div>
        <div>
          query_id: <span className="mono">{queryId ?? '—'}</span>
        </div>
        {config && (
          <div style={{ marginTop: 6 }}>
            query.txt:
            <div className="mono" style={{ wordBreak: 'break-all', fontSize: 10 }}>
              {config.query_txt_path}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}

function Bootstrapper() {
  const { setConfig } = useAppState();

  useEffect(() => {
    api
      .config()
      .then(setConfig)
      .catch(() => {
        /* ignore: shown by individual pages if needed */
      });
  }, [setConfig]);

  return null;
}

export default function App() {
  return (
    <AppStateProvider>
      <Bootstrapper />
      <div className="layout">
        <Sidebar />
        <main className="content">
          <Routes>
            <Route path="/" element={<Navigate to="/queries" replace />} />
            <Route path="/queries" element={<QueryPage />} />
            <Route path="/optimize" element={<OptimizePage />} />
            <Route path="/execute" element={<ExecutePage />} />
            <Route path="/result" element={<ResultPage />} />
            <Route path="/metrics" element={<MetricsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="*" element={<Navigate to="/queries" replace />} />
          </Routes>
        </main>
      </div>
    </AppStateProvider>
  );
}
