import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import type {
  ActualResult,
  BackendConfig,
  OptimizeResultPayload,
  QueryDetail,
} from './types';

interface SelectionState {
  dataset: string | null;
  queryId: number | null;
  setDataset: (dataset: string | null) => void;
  setQueryId: (queryId: number | null) => void;
  queryDetail: QueryDetail | null;
  setQueryDetail: (detail: QueryDetail | null) => void;
  optimizeTaskId: string | null;
  setOptimizeTaskId: (id: string | null) => void;
  optimizeResult: OptimizeResultPayload | null;
  setOptimizeResult: (payload: OptimizeResultPayload | null) => void;
  executeTaskId: string | null;
  setExecuteTaskId: (id: string | null) => void;
  actualResult: ActualResult | null;
  setActualResult: (payload: ActualResult | null) => void;
  config: BackendConfig | null;
  setConfig: (config: BackendConfig | null) => void;
}

const Ctx = createContext<SelectionState | null>(null);

const STORAGE_KEY = 'sp-ui-selection';

interface PersistedState {
  dataset: string | null;
  queryId: number | null;
}

function readPersisted(): PersistedState {
  if (typeof window === 'undefined') return { dataset: null, queryId: null };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { dataset: null, queryId: null };
    const parsed = JSON.parse(raw) as PersistedState;
    return {
      dataset: parsed.dataset ?? null,
      queryId: typeof parsed.queryId === 'number' ? parsed.queryId : null,
    };
  } catch {
    return { dataset: null, queryId: null };
  }
}

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const initial = useMemo(readPersisted, []);
  const [dataset, setDataset] = useState<string | null>(initial.dataset);
  const [queryId, setQueryId] = useState<number | null>(initial.queryId);
  const [queryDetail, setQueryDetail] = useState<QueryDetail | null>(null);
  const [optimizeTaskId, setOptimizeTaskId] = useState<string | null>(null);
  const [optimizeResult, setOptimizeResult] = useState<OptimizeResultPayload | null>(null);
  const [executeTaskId, setExecuteTaskId] = useState<string | null>(null);
  const [actualResult, setActualResult] = useState<ActualResult | null>(null);
  const [config, setConfig] = useState<BackendConfig | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ dataset, queryId } satisfies PersistedState),
      );
    } catch {
      // ignore
    }
  }, [dataset, queryId]);

  const value: SelectionState = {
    dataset,
    queryId,
    setDataset: (next) => {
      setDataset(next);
      setQueryDetail(null);
      setQueryId(null);
      setOptimizeResult(null);
      setActualResult(null);
    },
    setQueryId: (next) => {
      setQueryId(next);
      setQueryDetail(null);
      setOptimizeResult(null);
      setActualResult(null);
    },
    queryDetail,
    setQueryDetail,
    optimizeTaskId,
    setOptimizeTaskId,
    optimizeResult,
    setOptimizeResult,
    executeTaskId,
    setExecuteTaskId,
    actualResult,
    setActualResult,
    config,
    setConfig,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppState(): SelectionState {
  const ctx = useContext(Ctx);
  if (!ctx) {
    throw new Error('useAppState must be used within AppStateProvider');
  }
  return ctx;
}
