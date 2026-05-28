"""DB-backed implementation of ``SystemEventSink`` used by structlog.

Events are queued in-process; a background task drains the queue to the
``system_events`` table. The sink itself never blocks the calling code and
never raises into user code.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from typing import Any

from sqlalchemy import text

from affine_opeg.infrastructure.logging import SystemEventSink


class QueuedDbEventSink(SystemEventSink):
    """Bounded queue + background flusher. Drop-oldest on overflow."""

    def __init__(self, maxsize: int = 2048) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task[None] | None = None

    def emit(self, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(payload)

    async def run(self, session_factory: Any, batch_size: int = 64, flush_every: float = 1.0) -> None:
        while True:
            try:
                batch = await self._collect(batch_size=batch_size, flush_every=flush_every)
                if not batch:
                    continue
                async with session_factory() as session:
                    await session.execute(
                        text(
                            """
                            INSERT INTO system_events
                                (ts, service, level, event, message, context,
                                 trace_id, rollout_id, pair_id, run_id)
                            VALUES
                                (:ts, :service, :level, :event, :message, :context,
                                 :trace_id, :rollout_id, :pair_id, :run_id)
                            """
                        ),
                        batch,
                    )
                    await session.commit()
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 — sink must never crash the host process
                await asyncio.sleep(1.0)

    async def _collect(self, batch_size: int, flush_every: float) -> list[dict[str, Any]]:
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=flush_every)
        except asyncio.TimeoutError:
            return []
        batch: list[dict[str, Any]] = [_normalize(first)]
        while len(batch) < batch_size:
            try:
                batch.append(_normalize(self._queue.get_nowait()))
            except asyncio.QueueEmpty:
                break
        return batch


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    ts_raw = payload.get("ts")
    if isinstance(ts_raw, str):
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.utcnow()
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts = datetime.utcnow()
    return {
        "ts": ts,
        "service": payload.get("service", "unknown"),
        "level": payload.get("level", "info"),
        "event": payload.get("event", "")[:200],
        "message": payload.get("message"),
        "context": payload.get("context") or {},
        "trace_id": payload.get("trace_id"),
        "rollout_id": payload.get("rollout_id"),
        "pair_id": payload.get("pair_id"),
        "run_id": payload.get("run_id"),
    }
