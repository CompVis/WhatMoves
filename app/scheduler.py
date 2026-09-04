"""A bounded serial scheduler for stateful GPU inference."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field, replace
import itertools
import queue
import threading
import time
import traceback
from typing import Any, Callable
from uuid import uuid4


class Superseded(RuntimeError):
    """Raised when queued work is replaced or its owner is removed."""


class QueueFull(RuntimeError):
    """Raised when the bounded GPU queue has no available slot."""


@dataclass(order=True)
class _QueuedTask:
    priority: int
    order: int
    task_id: str = field(compare=False)
    function: Callable[[], Any] | None = field(compare=False)
    future: Future = field(compare=False)
    key: str | None = field(compare=False, default=None)
    owner: str | None = field(compare=False, default=None)
    retain_record: bool = field(compare=False, default=False)


@dataclass
class TaskRecord:
    id: str
    label: str
    status: str = "queued"
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    stage: str = "Queued"
    progress_current: int | None = None
    progress_total: int | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage": self.stage,
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
            "message": self.message,
        }


class GpuScheduler:
    """Run GPU work serially and release task payloads as soon as possible."""

    def __init__(self, max_pending: int = 32) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = int(max_pending)
        self._queue: queue.PriorityQueue[_QueuedTask] = queue.PriorityQueue()
        self._counter = itertools.count()
        self._lock = threading.RLock()
        self._latest_by_key: dict[str, str] = {}
        self._queued: dict[str, _QueuedTask] = {}
        self._records: dict[str, TaskRecord] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="whatmoves-gpu",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        function: Callable[[], Any],
        *,
        priority: int = 10,
        key: str | None = None,
        owner: str | None = None,
        label: str = "GPU task",
        retain_record: bool = False,
        task_id: str | None = None,
    ) -> tuple[str, Future]:
        future: Future = Future()
        task_id = task_id or uuid4().hex
        cancelled: tuple[Future, str] | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU scheduler is closed")
            if task_id in self._records:
                raise ValueError(f"Duplicate GPU task id: {task_id}")
            if key is not None:
                previous_id = self._latest_by_key.get(key)
                if previous_id is not None:
                    cancelled = self._cancel_locked(
                        previous_id,
                        f"Superseded by newer task: {label}",
                    )
            pending = sum(
                record.status in {"queued", "running"}
                for record in self._records.values()
            )
            if pending >= self.max_pending:
                raise QueueFull(f"GPU queue is full ({self.max_pending} pending tasks)")
            record = TaskRecord(task_id, label)
            task = _QueuedTask(
                int(priority),
                next(self._counter),
                task_id,
                function,
                future,
                key,
                owner,
                bool(retain_record),
            )
            self._records[task_id] = record
            self._queued[task_id] = task
            if key is not None:
                self._latest_by_key[key] = task_id
            self._queue.put(task)
        if cancelled is not None:
            cancelled_future, reason = cancelled
            if not cancelled_future.done():
                cancelled_future.set_exception(Superseded(reason))
        return task_id, future

    def update_progress(
        self,
        task_id: str,
        stage: str,
        current: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        """Publish honest stage progress from the worker that owns a task."""
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status not in {"queued", "running"}:
                return
            record.stage = str(stage)
            record.progress_current = None if current is None else int(current)
            record.progress_total = None if total is None else int(total)
            record.message = message

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            record = self._records.get(task_id)
            return replace(record) if record is not None else None

    def discard(self, task_id: str) -> None:
        """Forget a terminal retained record and its result."""
        with self._lock:
            record = self._records.get(task_id)
            if record is not None and record.status not in {"queued", "running"}:
                self._records.pop(task_id, None)

    def cancel_prefix(self, prefix: str) -> None:
        """Cancel queued tasks whose coalescing key starts with ``prefix``."""
        cancelled: list[tuple[Future, str]] = []
        with self._lock:
            for task_id, task in tuple(self._queued.items()):
                if task.key is not None and task.key.startswith(prefix):
                    item = self._cancel_locked(task_id, "Owning asset was removed")
                    if item is not None:
                        cancelled.append(item)
        for future, reason in cancelled:
            if not future.done():
                future.set_exception(Superseded(reason))

    def cancel_owner(self, owner: str) -> None:
        """Cancel all queued tasks belonging to one browser session."""
        cancelled: list[tuple[Future, str]] = []
        with self._lock:
            for task_id, task in tuple(self._queued.items()):
                if task.owner == owner:
                    item = self._cancel_locked(task_id, "Owning session expired")
                    if item is not None:
                        cancelled.append(item)
        for future, reason in cancelled:
            if not future.done():
                future.set_exception(Superseded(reason))

    def close(self) -> None:
        cancelled: list[tuple[Future, str]] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for task_id in tuple(self._queued):
                item = self._cancel_locked(task_id, "GPU scheduler is closing")
                if item is not None:
                    cancelled.append(item)
            sentinel = _QueuedTask(
                -(10**9),
                next(self._counter),
                "__close__",
                None,
                Future(),
            )
            self._queue.put(sentinel)
        for future, reason in cancelled:
            if not future.done():
                future.set_exception(Superseded(reason))
        self._thread.join()
        with self._lock:
            with self._queue.mutex:
                self._queue.queue.clear()
            self._queued.clear()
            self._latest_by_key.clear()
            self._records.clear()

    def _cancel_locked(
        self,
        task_id: str,
        reason: str,
    ) -> tuple[Future, str] | None:
        record = self._records.get(task_id)
        task = self._queued.get(task_id)
        if record is None or task is None or record.status != "queued":
            return None
        task.function = None
        record.status = "superseded"
        record.error = reason
        record.finished_at = time.time()
        if task.key is not None and self._latest_by_key.get(task.key) == task_id:
            self._latest_by_key.pop(task.key, None)
        return task.future, reason

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task.task_id == "__close__":
                del task
                return
            with self._lock:
                self._queued.pop(task.task_id, None)
                record = self._records.get(task.task_id)
                function = task.function
                task.function = None
                if record is None or function is None:
                    if not task.retain_record:
                        self._records.pop(task.task_id, None)
                    del task
                    continue
                record.status = "running"
                record.started_at = time.time()
                if record.stage == "Queued":
                    record.stage = "Starting"
            try:
                result = function()
            except Exception as error:  # worker errors must never kill the queue
                traceback.print_exc()
                with self._lock:
                    record.status = "failed"
                    record.error = str(error)
                    record.finished_at = time.time()
                    if (
                        task.key is not None
                        and self._latest_by_key.get(task.key) == task.task_id
                    ):
                        self._latest_by_key.pop(task.key, None)
                if not task.future.done():
                    task.future.set_exception(error)
            else:
                with self._lock:
                    record.status = "complete"
                    record.result = result
                    record.finished_at = time.time()
                    if (
                        task.key is not None
                        and self._latest_by_key.get(task.key) == task.task_id
                    ):
                        self._latest_by_key.pop(task.key, None)
                if not task.future.done():
                    task.future.set_result(result)
                del result
            finally:
                del function
                with self._lock:
                    if not task.retain_record:
                        self._records.pop(task.task_id, None)
                del task
