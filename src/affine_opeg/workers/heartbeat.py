"""Worker heartbeat task.

Each worker spawns this as a background task. Upserts a row in
``service_heartbeats`` every ``interval`` seconds so the obs ``/workers``
endpoint and the dashboard can show liveness.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Any

from sqlalchemy import text

from affine_opeg.infrastructure.logging import get_logger

log = get_logger("heartbeat")


async def heartbeat_loop(
    session_factory: Any, *, worker_id: str, role: str, version: str,
    interval: float = 10.0,
) -> None:
    host = socket.gethostname()
    pid = os.getpid()
    while True:
        try:
            async with session_factory() as session:
                await session.execute(text(
                    """
                    INSERT INTO service_heartbeats
                        (worker_id, role, host, pid, version, status, last_seen)
                    VALUES
                        (:wid, :role, :host, :pid, :version, 'idle', now())
                    ON CONFLICT (worker_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        last_seen = now(),
                        version = EXCLUDED.version
                    """
                ), {"wid": worker_id, "role": role, "host": host, "pid": pid, "version": version})
                await session.commit()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("heartbeat.failed", error=str(exc))
        await asyncio.sleep(interval)
