"""In-process metric aggregator with periodic DB flush.

Replaces a prometheus push gateway. Each worker maintains an in-memory map of
``(metric, labels) -> counter / last value``, then once a minute upserts to
``metrics_minutely``. The obs API queries this table for charts.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from affine_opeg.infrastructure.logging import get_logger

log = get_logger("metrics")


def _labels_key(labels: dict[str, str]) -> str:
    return json.dumps(labels, sort_keys=True, separators=(",", ":"))


@dataclass
class MetricAggregator:
    """One per process. Thread-safe through the asyncio single-thread model."""

    counters: dict[tuple[str, str], float] = field(default_factory=dict)
    gauges: dict[tuple[str, str], float] = field(default_factory=dict)
    histograms: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    def incr(self, metric: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = (metric, _labels_key(labels or {}))
        self.counters[key] = self.counters.get(key, 0.0) + value

    def set(self, metric: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (metric, _labels_key(labels or {}))
        self.gauges[key] = value

    def observe(self, metric: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (metric, _labels_key(labels or {}))
        self.histograms.setdefault(key, []).append(value)

    def drain(self) -> list[tuple[str, dict[str, str], float]]:
        """Atomically swap and return all metric samples for the current minute."""
        out: list[tuple[str, dict[str, str], float]] = []
        for (metric, labels_json), v in self.counters.items():
            out.append((metric, json.loads(labels_json), v))
        for (metric, labels_json), v in self.gauges.items():
            out.append((metric, json.loads(labels_json), v))
        for (metric, labels_json), samples in self.histograms.items():
            if not samples:
                continue
            labels = json.loads(labels_json)
            out.append((f"{metric}.sum", labels, float(sum(samples))))
            out.append((f"{metric}.count", labels, float(len(samples))))
            out.append((f"{metric}.max", labels, float(max(samples))))
        # reset for the next bucket
        self.counters.clear()
        self.histograms.clear()
        # gauges are sticky — keep last value
        return out


_AGG = MetricAggregator()


def metrics() -> MetricAggregator:
    return _AGG


async def flush_metrics(session: AsyncSession, agg: MetricAggregator | None = None) -> int:
    """Flush one minute bucket to the DB. Returns number of rows written."""
    bucket = agg or _AGG
    samples = bucket.drain()
    if not samples:
        return 0
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    await session.execute(
        text(
            """
            INSERT INTO metrics_minutely (ts_minute, metric, labels, value)
            VALUES (:ts_minute, :metric, :labels, :value)
            ON CONFLICT (ts_minute, metric, labels) DO UPDATE SET value = EXCLUDED.value
            """
        ),
        [
            {"ts_minute": now, "metric": m, "labels": labels, "value": v}
            for m, labels, v in samples
        ],
    )
    return len(samples)


async def metrics_flush_loop(session_factory: Any, interval_seconds: int = 60) -> None:
    """Background task: call ``flush_metrics`` every interval. Cancellation-safe."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            async with session_factory() as session:
                n = await flush_metrics(session)
                await session.commit()
            if n:
                log.debug("metrics.flushed", rows=n)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("metrics.flush_failed", error=str(exc))


def timer(metric: str, labels: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    """Decorator/context: observe wall-clock latency in seconds."""

    class _T:
        def __enter__(self):
            self._t = time.monotonic()
            return self

        def __exit__(self, *_exc):
            metrics().observe(metric, time.monotonic() - self._t, labels=labels)

    return _T()
