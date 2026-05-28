"""Audit logging.

Every business write that crosses an API boundary or scheduler decision boundary
should call ``record_audit`` so the ``audit_log`` table preserves who/what/when.
This is independent of the system_events log sink — audit_log is intentional
historic record, system_events is operational visibility.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from affine_opeg.infrastructure.logging import current_trace_id


async def record_audit(
    session: AsyncSession,
    *,
    actor: str,
    action: str,
    entity_kind: str | None = None,
    entity_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Insert a row into ``audit_log``. Must run inside the same UoW as the
    write being audited so they commit atomically."""
    await session.execute(
        text(
            """
            INSERT INTO audit_log (actor, action, entity_kind, entity_id, payload, trace_id)
            VALUES (:actor, :action, :entity_kind, :entity_id, :payload, :trace_id)
            """
        ),
        {
            "actor": actor,
            "action": action,
            "entity_kind": entity_kind,
            "entity_id": entity_id,
            "payload": payload,
            "trace_id": current_trace_id(),
        },
    )
