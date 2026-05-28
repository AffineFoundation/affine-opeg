"""Structured logging.

stdout: JSON via structlog, captured by container log driver.
DB sink: events with level >= configured threshold are mirrored to the
``system_events`` table by a queue-backed processor so query/audit paths can
serve them via the obs API. The DB sink is fire-and-forget — log failures
must never break business code.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol
from uuid import uuid4

import structlog
from structlog.contextvars import bind_contextvars, merge_contextvars

from affine_opeg.infrastructure.config import AppConfig

_LEVEL_NUM = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_trace_id_ctx: ContextVar[str | None] = ContextVar("afr_trace_id", default=None)


class SystemEventSink(Protocol):
    """Asynchronous fan-out of structured log events to the ``system_events`` table."""

    def emit(self, payload: dict[str, Any]) -> None: ...


_db_sink: SystemEventSink | None = None
_db_sink_threshold: int = _LEVEL_NUM["WARNING"]


def _db_sink_processor(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor: forward selected events to the DB sink."""
    if _db_sink is None:
        return event_dict
    level = event_dict.get("level", "info").upper()
    if _LEVEL_NUM.get(level, 0) < _db_sink_threshold:
        return event_dict
    payload = {
        "ts": event_dict.get("timestamp"),
        "service": event_dict.get("service", "unknown"),
        "level": level.lower(),
        "event": event_dict.get("event", ""),
        "message": event_dict.get("message"),
        "context": {k: v for k, v in event_dict.items()
                    if k not in ("timestamp", "level", "event", "service", "message",
                                 "trace_id", "rollout_id", "pair_id", "run_id")},
        "trace_id": event_dict.get("trace_id"),
        "rollout_id": event_dict.get("rollout_id"),
        "pair_id": event_dict.get("pair_id"),
        "run_id": event_dict.get("run_id"),
    }
    try:
        _db_sink.emit(payload)
    except Exception:  # noqa: BLE001 — sink must not raise
        pass
    return event_dict


def configure_logging(cfg: AppConfig, *, service: str | None = None) -> None:
    """Configure structlog. Call once at process startup before any logger use."""
    global _db_sink_threshold  # noqa: PLW0603
    _db_sink_threshold = _LEVEL_NUM[cfg.logging.db_sink_min_level]

    processors = [
        merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _db_sink_processor,
    ]
    if cfg.logging.as_json:
        processors.append(structlog.processors.JSONRenderer(serializer=_orjson_dumps))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_LEVEL_NUM[cfg.logging.level]),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    logging.captureWarnings(True)
    bind_contextvars(service=service or cfg.service, env=cfg.env, version=cfg.version)


def _orjson_dumps(obj: Any, default: Any = None) -> str:
    import orjson

    return orjson.dumps(obj, default=default).decode()


def register_db_sink(sink: SystemEventSink) -> None:
    """Wire a DB sink into the logger. Safe to call after configure_logging."""
    global _db_sink  # noqa: PLW0603
    _db_sink = sink


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name or "affine_opeg")


def new_trace_id() -> str:
    return uuid4().hex


def current_trace_id() -> str | None:
    return _trace_id_ctx.get()


@contextmanager
def trace_context(trace_id: str | None = None, **extra: Any):
    """Bind a trace_id (and optional extra fields) for the duration of a block."""
    tid = trace_id or new_trace_id()
    token = _trace_id_ctx.set(tid)
    bind_contextvars(trace_id=tid, **extra)
    start = time.monotonic()
    try:
        yield tid
    finally:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        bind_contextvars(elapsed_ms=elapsed_ms)
        _trace_id_ctx.reset(token)
