import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';

interface LogStreamProps {
  taskId: string | null;
  onStatusChange?: (status: string) => void;
  onComplete?: () => void;
  onError?: (error: string) => void;
  maxHeight?: number;
}

export default function LogStream({
  taskId,
  onStatusChange,
  onComplete,
  onError,
  maxHeight = 360,
}: LogStreamProps) {
  const [log, setLog] = useState<string>('');
  const [status, setStatus] = useState<string>('pending');
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!taskId) {
      setLog('');
      setStatus('idle');
      return;
    }
    setLog('');
    setStatus('pending');

    const url = api.taskStreamUrl(taskId);
    const source = new EventSource(url);

    source.addEventListener('log', (event) => {
      setLog((prev) => prev + (event as MessageEvent).data + '\n');
    });
    source.addEventListener('status', (event) => {
      const next = (event as MessageEvent).data;
      setStatus(next);
      onStatusChange?.(next);
    });
    source.addEventListener('done', () => {
      source.close();
      onComplete?.();
    });
    source.addEventListener('close', () => {
      source.close();
    });
    source.onerror = () => {
      onError?.('SSE connection error');
      source.close();
    };

    return () => {
      source.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [log]);

  if (!taskId) {
    return <div className="empty">还没有触发任何任务</div>;
  }

  return (
    <div className="card">
      <h3>
        日志流 <span className={`tag ${statusTag(status)}`}>{status}</span>
      </h3>
      <div className="log-stream" ref={containerRef} style={{ maxHeight }}>
        {log || '等待日志输出...'}
      </div>
    </div>
  );
}

function statusTag(status: string): string {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'running') return 'warn';
  return '';
}
