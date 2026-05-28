"""Random cell scheduler — materialise new ``sampling_progress`` rows on demand.

Why
---

The old design (admin pre-INSERTs every cell into ``sampling_progress``)
makes the upcoming ``(env, task_id, teacher)`` triples *predictable*. A
miner that watches the published manifest can extrapolate the admin's
selection policy and pre-cache answers / pre-train on the next batch
before it's evaluated.

This module flips it: the admin only configures a **pool** (a set of
allowed envs / task_id ranges / teachers) inside
``sampling_lists.config.pool``. Cells are materialised one at a time by
:func:`maintain_active_pool` using :class:`secrets.SystemRandom`. The
random pick is unpredictable in advance — only the validator's
producer process knows which cell it just chose, and only *after*
``sampling_progress`` got the new row.

Concurrency: the unique primary key
``(list_name, env_name, task_id, teacher_name)`` on
``sampling_progress`` plus ``ON CONFLICT DO NOTHING`` makes
``maintain_active_pool`` safe to invoke from multiple producers
racing the same list.

Active vs frozen
----------------

An "active" cell is one that the producer still wants to (and can)
sample. Definition (mirrors ``claim_next_cell``'s eligibility):

    collected < target_samples
    AND attempts < 2 * target_samples
    AND published_at IS NULL

Once that's false the cell is frozen — counts toward the publisher's
queue, not the scheduler's pool. We keep refilling until the active
count reaches ``target_active_cells``.

Bounded retries
---------------

Random picks may collide with cells already in the table. We allow
``MAX_RETRIES_PER_INSERT`` consecutive empty picks before giving up —
this protects against the degenerate case where the pool space is
fully exhausted (rare for swe-rebench's 39k tasks × N teachers).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import and_, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg.orm import (
    sampling_lists,
    sampling_progress,
    tasks,
)
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("scheduler.cell")

MAX_RETRIES_PER_INSERT = 32


@dataclass(frozen=True)
class PoolConfig:
    """Decoded ``sampling_lists.config.pool``.

    ``task_id_pool[env_name]`` is the full list of task_ids the
    scheduler is allowed to draw from for that env. Storing the full
    enumeration (instead of just a [lo, hi] range) lets the admin
    pre-filter the dataset arbitrarily (e.g. python-only, post-2025,
    fail_to_pass≥3) without the scheduler needing to redo any of that
    work.
    """

    envs: tuple[str, ...]
    task_id_pool: dict[str, tuple[int, ...]]
    teachers: tuple[str, ...]
    target_active_cells: int
    target_samples: int


def parse_pool_config(raw: Any) -> PoolConfig:
    """Pull a ``PoolConfig`` out of the JSONB stored in ``sampling_lists``."""
    pool = raw.get("pool") if isinstance(raw, dict) else None
    if not isinstance(pool, dict):
        raise ValueError("sampling_lists.config.pool missing or not a dict")
    envs = tuple(pool.get("envs") or ())
    teachers = tuple(pool.get("teachers") or ())
    task_id_pool_raw = pool.get("task_id_pool") or {}
    task_id_pool = {
        str(k): tuple(int(x) for x in v)
        for k, v in task_id_pool_raw.items()
    }
    if not envs:
        raise ValueError("pool.envs is empty")
    if not teachers:
        raise ValueError("pool.teachers is empty")
    for env in envs:
        if not task_id_pool.get(env):
            raise ValueError(f"pool.task_id_pool[{env!r}] missing or empty")
    return PoolConfig(
        envs=envs,
        task_id_pool=task_id_pool,
        teachers=teachers,
        target_active_cells=int(raw.get("target_active_cells", 16)),
        target_samples=int(raw.get("target_samples", 8)),
    )


@dataclass
class ScheduleResult:
    inserted: int
    skipped_conflicts: int
    active_after: int
    pool_exhausted: bool


async def maintain_active_pool(
    sessionmaker, list_name: str, *,
    rng: secrets.SystemRandom | None = None,
) -> ScheduleResult:
    """Top up the list's active cell count to ``target_active_cells``.

    Returns a ``ScheduleResult`` describing the cycle. Safe to call
    concurrently with itself and with ``claim_next_cell`` — the unique
    constraint on ``sampling_progress`` provides the synchronisation.
    """
    rng = rng or secrets.SystemRandom()

    async with sessionmaker() as session:
        cfg_row = (await session.execute(
            select(sampling_lists.c.config).where(sampling_lists.c.list_name == list_name)
        )).first()
    if cfg_row is None:
        raise LookupError(f"sampling_lists row not found: {list_name}")
    pool = parse_pool_config(cfg_row[0])

    inserted = 0
    skipped = 0
    pool_exhausted = False

    while True:
        active = await _count_active(sessionmaker, list_name)
        if active >= pool.target_active_cells:
            break

        consecutive_conflicts = 0
        committed_this_round = False
        while consecutive_conflicts < MAX_RETRIES_PER_INSERT:
            env = rng.choice(pool.envs)
            task_id = rng.choice(pool.task_id_pool[env])
            teacher = rng.choice(pool.teachers)
            row_count = await _try_insert(
                sessionmaker, list_name=list_name,
                env_name=env, task_id=task_id, teacher_name=teacher,
                target_samples=pool.target_samples,
            )
            if row_count == 0:
                skipped += 1
                consecutive_conflicts += 1
                continue
            # Successful INSERT.
            inserted += 1
            committed_this_round = True
            log.info(
                "scheduler.cell.inserted",
                list_name=list_name, env_name=env,
                task_id=task_id, teacher_name=teacher,
            )
            break

        if not committed_this_round:
            # Hit MAX_RETRIES_PER_INSERT consecutive collisions — pool
            # is effectively saturated. Stop trying this cycle.
            pool_exhausted = True
            log.warning(
                "scheduler.pool_saturated",
                list_name=list_name,
                consecutive_conflicts=consecutive_conflicts,
            )
            break

    active_after = await _count_active(sessionmaker, list_name)
    return ScheduleResult(
        inserted=inserted, skipped_conflicts=skipped,
        active_after=active_after, pool_exhausted=pool_exhausted,
    )


async def _count_active(sessionmaker, list_name: str) -> int:
    async with sessionmaker() as session:
        stmt = (
            select(func.count())
            .select_from(sampling_progress)
            .where(and_(
                sampling_progress.c.list_name == list_name,
                sampling_progress.c.published_at.is_(None),
                sampling_progress.c.collected < sampling_progress.c.target_samples,
                sampling_progress.c.attempts < 2 * sampling_progress.c.target_samples,
            ))
        )
        return int((await session.execute(stmt)).scalar_one())


async def _try_insert(
    sessionmaker, *,
    list_name: str, env_name: str, task_id: int, teacher_name: str,
    target_samples: int,
) -> int:
    """INSERT one sampling_progress row; ON CONFLICT skip.

    Also guards against the FK violation when ``(env_name, task_id)``
    is not in the ``tasks`` pool — silently treats that as a conflict
    (the admin's PoolConfig is the source of truth; an
    out-of-date config that points at a removed task should not crash
    the scheduler, just retry).
    """
    async with sessionmaker() as session:
        # Cheap FK pre-check: if the task isn't seeded we skip. Avoids
        # poisoning the transaction with a constraint violation.
        exists = (await session.execute(
            select(func.count()).select_from(tasks).where(and_(
                tasks.c.env_name == env_name, tasks.c.task_id == task_id,
            ))
        )).scalar_one()
        if not exists:
            return 0

        stmt = pg_insert(sampling_progress).values(
            list_name=list_name, env_name=env_name, task_id=task_id,
            teacher_name=teacher_name, target_samples=target_samples,
            attempts=0, collected=0,
        ).on_conflict_do_nothing(
            index_elements=["list_name", "env_name", "task_id", "teacher_name"]
        )
        result = await session.execute(stmt)
        await session.commit()
        return int(result.rowcount or 0)
