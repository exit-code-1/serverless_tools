# -*- coding: utf-8 -*-
"""In-memory task registry with async log streaming.

Each task carries a status, metadata dict and an asyncio queue that streams log
lines (or arbitrary string events) to SSE consumers. The synchronous worker can
push events via `submit` without needing a running event loop; we hop back onto
the loop via `loop.call_soon_threadsafe` when present, otherwise we buffer
events for late subscribers."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class TaskRecord:
    task_id: str
    kind: str
    status: str = "pending"  # pending | running | completed | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    loop: Optional[asyncio.AbstractEventLoop] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class TaskManager:
    """Thread-safe task registry. Workers run in background threads, SSE
    subscribers consume events on the FastAPI event loop."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, metadata: Optional[Dict[str, Any]] = None) -> TaskRecord:
        task_id = uuid.uuid4().hex[:12]
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        record = TaskRecord(
            task_id=task_id,
            kind=kind,
            metadata=dict(metadata or {}),
            loop=loop,
        )
        with self._lock:
            self._tasks[task_id] = record
        return record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_all(self) -> List[TaskRecord]:
        with self._lock:
            return list(self._tasks.values())

    def mark_started(self, task_id: str) -> None:
        record = self._require(task_id)
        with record._lock:
            record.status = "running"
            record.started_at = time.time()
        self._broadcast(record, "status", record.status)

    def mark_completed(self, task_id: str, **metadata: Any) -> None:
        record = self._require(task_id)
        with record._lock:
            record.status = "completed"
            record.finished_at = time.time()
            record.metadata.update(metadata)
        self._broadcast(record, "status", record.status)
        self._broadcast(record, "done", "1")

    def mark_failed(self, task_id: str, error: str) -> None:
        record = self._require(task_id)
        with record._lock:
            record.status = "failed"
            record.finished_at = time.time()
            record.error = error
        self._broadcast(record, "log", f"[ERROR] {error}")
        self._broadcast(record, "status", record.status)
        self._broadcast(record, "done", "1")

    def append_log(self, task_id: str, line: str) -> None:
        record = self._require(task_id)
        if not line:
            return
        if not line.endswith("\n"):
            line = line + "\n"
        with record._lock:
            record.logs.append(line)
            if len(record.logs) > 5000:
                # Trim buffer so we don't grow without bound.
                record.logs = record.logs[-5000:]
        self._broadcast(record, "log", line)

    def update_metadata(self, task_id: str, **metadata: Any) -> None:
        record = self._require(task_id)
        with record._lock:
            record.metadata.update(metadata)
        self._broadcast(record, "metadata", "1")

    def get_logs(self, task_id: str) -> str:
        record = self._require(task_id)
        with record._lock:
            return "".join(record.logs)

    async def subscribe(self, task_id: str) -> AsyncIterator[Dict[str, str]]:
        """Async generator yielding SSE events: replay buffered logs first,
        then live updates until the task reaches a terminal state."""
        record = self._require(task_id)
        queue: asyncio.Queue = asyncio.Queue()
        with record._lock:
            backlog = list(record.logs)
            current_status = record.status
            record.subscribers.append(queue)
            record.loop = asyncio.get_running_loop()

        try:
            for line in backlog:
                yield {"event": "log", "data": line}
            yield {"event": "status", "data": current_status}
            if current_status in {"completed", "failed", "cancelled"}:
                yield {"event": "done", "data": "1"}
                return

            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
                if event.get("event") == "done":
                    break
        finally:
            with record._lock:
                if queue in record.subscribers:
                    record.subscribers.remove(queue)

    def _require(self, task_id: str) -> TaskRecord:
        record = self.get(task_id)
        if record is None:
            raise KeyError(f"Task not found: {task_id}")
        return record

    def _broadcast(self, record: TaskRecord, event: str, data: str) -> None:
        payload = {"event": event, "data": data}
        with record._lock:
            subscribers = list(record.subscribers)
            loop = record.loop
        if not subscribers:
            return
        for q in subscribers:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(_safe_put, q, payload)
            else:
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass


def _safe_put(queue: asyncio.Queue, payload: Dict[str, str]) -> None:
    try:
        queue.put_nowait(payload)
    except Exception:
        pass


_task_manager_singleton: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    global _task_manager_singleton
    if _task_manager_singleton is None:
        _task_manager_singleton = TaskManager()
    return _task_manager_singleton
