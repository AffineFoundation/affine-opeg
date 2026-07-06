"""Weighted cell scheduler — materialise ``sampling_progress`` rows on demand.

Why
---

The old design (admin pre-INSERTs every cell into ``sampling_progress``)
makes the upcoming ``(env, task_id, teacher)`` triples *predictable*. A
miner that watches the published manifest can extrapolate the admin's
selection policy and pre-cache answers / pre-train on the next batch
before it's evaluated.

This module flips it: the admin only configures a **pool** (a set of
allowed envs / task_id ranges / teachers) inside
``sampling_lists.config.pool``. Cells are materialised by
:func:`maintain_active_pool` picking a random ``(task_id, teacher)`` with
:class:`secrets.SystemRandom`. The random pick is unpredictable in
advance — only the validator's producer process knows which cell it just
chose, and only *after* ``sampling_progress`` got the new row.

Per-env weighting
-----------------

Env selection is **not** uniform. Each env has a target *active* cell
count proportional to its ``pool.env_weights[env]`` share; the scheduler
tops each env up to its own target independently. This does two things a
global "N active cells, pick a uniform-random env" pool cannot:

  * **Weighting is exact.** A high-yield env (many cells convert to a
    publishable task) gets proportionally more in-flight cells.
  * **No slow-consumer starvation.** A slow / error-prone env (e.g.
    swe-rebench, whose cells take minutes and often never reach
    ``target_samples``) can only fill *its own* budget. Under the old
    global count its un-completable cells accumulated until they crowded
    out the whole active pool and every other env drained to zero.

When ``env_weights`` is absent the weight defaults to each env's
multiplicity in ``pool.envs`` (back-compat: the legacy hand-tuned pool
listed hot envs multiple times).

Zombie cells (TTL)
------------------

A cell is a **zombie** when it is still eligible (``collected <
target_samples`` and within the attempt budget) but its ``last_updated``
is older than ``cell_ttl_s`` — nothing has touched it for a long time
because the claim order (``collected ASC``) deprioritises it and its task
is effectively un-completable (broken build, always-erroring teacher).
Left alone it lingers forever, occupying an active slot. The scheduler
**excludes zombies from the active count** so the env keeps flowing fresh
cells; the publisher settles them out separately (see
``publisher._list_pending_cells`` TTL branch).

Active vs frozen
----------------

An "active" cell is one the producer still wants to (and can) sample:

    collected < target_samples
    AND attempts < 2 * target_samples
    AND published_at IS NULL
    AND last_updated >= now() - cell_ttl_s   (not a zombie)

Bounded retries
---------------

Random picks may collide with cells already in the table. We allow
``MAX_RETRIES_PER_INSERT`` consecutive empty picks per env before giving
up on that env this cycle (its (task, teacher) space is saturated).
"""

from __future__ import annotations

import secrets
from collections import Counter
from dataclasses import dataclass
from typing import Any

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
DEFAULT_CELL_TTL_S = 21600  # 6h; normal cells complete in minutes
DEFAULT_MIN_ACTIVE_PER_ENV = 1  # cold-start floor: every listed env keeps a probe


@dataclass(frozen=True)
class PoolConfig:
    """Decoded ``sampling_lists.config`` (pool block + top-level knobs).

    ``task_id_pool[env_name]`` is the full list of task_ids the scheduler
    is allowed to draw from for that env. Storing the full enumeration
    (instead of just a [lo, hi] range) lets the admin pre-filter the
    dataset arbitrarily (e.g. python-only, post-2025, fail_to_pass>=3).

    ``env_weights[env_name]`` is a non-negative float; the per-env active
    target is ``weight / sum(weights) * target_active_cells`` (floored at
    ``min_active_per_env``). Envs with weight 0 get only the floor — the
    controller "cuts" a dud env by driving its weight toward 0, but the
    floor keeps a recovery probe so an env zeroed by early bad luck can
    climb back (cold-start protection).
    """

    envs: tuple[str, ...]
    task_id_pool: dict[str, tuple[int, ...]]
    teachers: tuple[str, ...]
    env_weights: dict[str, float]
    target_active_cells: int
    target_samples: int
    cell_ttl_s: int
    min_active_per_env: int


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
    # Distinct envs preserve first-seen order; multiplicity (legacy weight)
    # is folded into env_weights below.
    distinct_envs = tuple(dict.fromkeys(envs))
    for env in distinct_envs:
        if not task_id_pool.get(env):
            raise ValueError(f"pool.task_id_pool[{env!r}] missing or empty")

    # Weights: explicit pool.env_weights wins; else fall back to the
    # env's multiplicity in pool.envs (the legacy hand-tuned encoding).
    raw_weights = pool.get("env_weights")
    if isinstance(raw_weights, dict) and raw_weights:
        env_weights = {e: max(0.0, float(raw_weights.get(e, 0.0))) for e in distinct_envs}
    else:
        counts = Counter(envs)
        env_weights = {e: float(counts[e]) for e in distinct_envs}

    return PoolConfig(
        envs=distinct_envs,
        task_id_pool=task_id_pool,
        teachers=teachers,
        env_weights=env_weights,
        target_active_cells=int(raw.get("target_active_cells", 16)),
        target_samples=int(raw.get("target_samples", 8)),
        cell_ttl_s=int(raw.get("cell_ttl_s", pool.get("cell_ttl_s", DEFAULT_CELL_TTL_S))),
        min_active_per_env=int(
            raw.get("min_active_per_env", pool.get("min_active_per_env", DEFAULT_MIN_ACTIVE_PER_ENV))
        ),
    )


def _env_targets(pool: PoolConfig) -> dict[str, int]:
    """Per-env active-cell target from weights (floored at min_active_per_env)."""
    total = sum(pool.env_weights.get(e, 0.0) for e in pool.envs)
    targets: dict[str, int] = {}
    for e in pool.envs:
        if total > 0:
            share = pool.env_weights.get(e, 0.0) / total
            tgt = round(share * pool.target_active_cells)
        else:
            tgt = 0
        targets[e] = max(pool.min_active_per_env, tgt)
    return targets


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
    """Top up each env's active cell count to its per-env target.

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

    targets = _env_targets(pool)
    active_by_env = await _count_active_by_env(sessionmaker, list_name, pool.cell_ttl_s)

    inserted = 0
    skipped = 0
    saturated_envs = 0

    # Fill the env with the largest deficit first so a saturated env can't
    # stall the whole cycle before productive envs get their budget.
    order = sorted(pool.envs, key=lambda e: targets[e] - active_by_env.get(e, 0), reverse=True)
    for env in order:
        need = targets[env] - active_by_env.get(env, 0)
        if need <= 0:
            continue
        task_ids = pool.task_id_pool[env]
        consecutive_conflicts = 0
        while need > 0 and consecutive_conflicts < MAX_RETRIES_PER_INSERT:
            task_id = rng.choice(task_ids)
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
            inserted += 1
            need -= 1
            consecutive_conflicts = 0
            log.info(
                "scheduler.cell.inserted",
                list_name=list_name, env_name=env,
                task_id=task_id, teacher_name=teacher,
            )
        if need > 0 and consecutive_conflicts >= MAX_RETRIES_PER_INSERT:
            saturated_envs += 1
            log.warning(
                "scheduler.env_saturated",
                list_name=list_name, env_name=env,
                target=targets[env], active=active_by_env.get(env, 0),
            )

    active_after = await _count_active(sessionmaker, list_name, pool.cell_ttl_s)
    return ScheduleResult(
        inserted=inserted, skipped_conflicts=skipped,
        active_after=active_after, pool_exhausted=saturated_envs > 0,
    )


def _active_predicate(ttl_s: int):
    """SQLAlchemy predicate for a non-zombie active cell.

    ``last_updated >= now() - ttl`` excludes zombies (eligible but stale).
    """
    return and_(
        sampling_progress.c.published_at.is_(None),
        sampling_progress.c.collected < sampling_progress.c.target_samples,
        sampling_progress.c.attempts < 2 * sampling_progress.c.target_samples,
        sampling_progress.c.last_updated >= func.now() - text(f"interval '{int(ttl_s)} seconds'"),
    )


async def _count_active(sessionmaker, list_name: str, ttl_s: int) -> int:
    async with sessionmaker() as session:
        stmt = (
            select(func.count())
            .select_from(sampling_progress)
            .where(and_(
                sampling_progress.c.list_name == list_name,
                _active_predicate(ttl_s),
            ))
        )
        return int((await session.execute(stmt)).scalar_one())


async def _count_active_by_env(sessionmaker, list_name: str, ttl_s: int) -> dict[str, int]:
    async with sessionmaker() as session:
        stmt = (
            select(sampling_progress.c.env_name, func.count())
            .where(and_(
                sampling_progress.c.list_name == list_name,
                _active_predicate(ttl_s),
            ))
            .group_by(sampling_progress.c.env_name)
        )
        rows = (await session.execute(stmt)).all()
    return {str(env): int(n) for env, n in rows}


async def _try_insert(
    sessionmaker, *,
    list_name: str, env_name: str, task_id: int, teacher_name: str,
    target_samples: int,
) -> int:
    """INSERT one sampling_progress row; ON CONFLICT skip.

    Also guards against the FK violation when ``(env_name, task_id)``
    is not in the ``tasks`` pool — silently treats that as a conflict
    (the admin's PoolConfig is the source of truth; an out-of-date
    config that points at a removed task should not crash the scheduler,
    just retry).
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
