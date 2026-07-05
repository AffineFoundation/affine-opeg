"""Publish rollouts to R2 as immutable per-cell shards.

Reproducibility model
---------------------

A **distill-v2 task** is a finished sampling cell:

    cell = (list_name, env_name, task_id, teacher_name)

The cell is "frozen" — and thus eligible for publish — when

    collected >= target_samples            (enough successes)
    OR attempts >= 2 * target_samples      (attempt budget spent)

In both cases ``claim_next_cell`` will no longer hand out new
sample_idx values for the cell, so the rollouts set will not grow.

The publisher uses ``published_at`` as its idempotency marker —
incremental queries always look at ``published_at IS NULL``, so the
per-cycle work scales with *pending* cells, not *all* cells ever
produced. Crash safety: the R2 manifest is the authoritative
``task_idx → cell`` mapping; ``published_at`` is best-effort and gets
re-marked from the manifest's ``known`` set if a previous cycle died
between manifest PUT and PG UPDATE.

Layout on R2 (bucket name doubles as the dataset namespace, e.g.
``distill-v2``):

    s3://{bucket}/manifest.jsonl                    # one cell per line
    s3://{bucket}/tasks/{task_idx:08d}.parquet      # immutable per-cell shard

If a non-empty ``BlobConfig.prefix`` is set, both keys gain that
prefix (e.g. ``staging/manifest.jsonl``); empty prefix puts keys at
the root.

Per-cell shard schema mirrors the ``rollouts`` PG columns, with
``extra_compressed`` (zstd-compressed ``NormalizedTrajectory`` JSON)
carried inline. All rows in a shard share the same teacher, so
within-cell advantage z-score eliminates style leakage across teachers.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import and_, func, or_, select, update

from affine_opeg.adapters.blob_stores.s3 import S3BlobStore
from affine_opeg.adapters.metadata_stores.sqlalchemy_pg.orm import (
    rollouts as rollouts_t,
    sampling_progress as sp_t,
)
from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import get_sessionmaker
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("publisher.r2")


@dataclass(frozen=True)
class PublishParams:
    """Per-cycle knobs.

    ``min_reward_std`` — drop cells whose within-cell reward std is
    below this. Default 0.05 — rejects cells where the N rollouts have
    no meaningful reward spread (z-score advantage would be all zeros,
    evaluator would get a useless degenerate cell). Lower to 1e-6 if
    you only want to skip exact std=0 cells; raise to 0.1 if your
    reward scale is coarse and you want only clear winners/losers.

    ``max_new_per_cycle`` — bound on shards uploaded per cycle. Keeps
    cycle time predictable.

    ``maturation_window_s`` — gap between ``committed_at`` (publisher
    upload time) and ``mature_at`` (the earliest moment downstream
    consumers — i.e. miners that fetch via a public manifest endpoint —
    should be allowed to see the entry). Default 24h, matching the
    swe-infinite convention. Validators that read the manifest with
    ``include_immature=True`` see the row immediately.

    ``list_names`` / ``env_names`` / ``teacher_names`` — optional
    filters on which cells are considered eligible.
    """

    list_names: tuple[str, ...] = ()
    env_names: tuple[str, ...] = ()
    teacher_names: tuple[str, ...] = ()
    min_reward_std: float = 0.05
    max_new_per_cycle: int = 200
    # 0 means no maturation delay; promoter controls release via
    # ``AFR_PROMOTE_MAX_PER_DAY`` rate cap instead.
    maturation_window_s: int = 0


@dataclass(frozen=True)
class PublishResult:
    n_committed: int                 # new cells uploaded this cycle
    n_recovered: int                 # cells already in manifest, just PG-marked
    manifest_uri: str
    total_tasks_after: int           # manifest length after this cycle
    skipped_degenerate: int          # frozen but reward_std below threshold
    skipped_empty: int               # frozen but zero ok rollouts


_CELL_PARQUET_SCHEMA = pa.schema([
    ("rollout_id", pa.string()),
    ("env_name", pa.string()),
    ("task_id", pa.int64()),
    ("teacher_name", pa.string()),
    ("sample_idx", pa.int32()),
    ("status", pa.string()),
    ("reward", pa.float64()),
    ("temperature", pa.float32()),
    ("top_p", pa.float32()),
    ("seed", pa.int64()),
    ("steps", pa.int32()),
    ("tokens_in", pa.int64()),
    ("tokens_out", pa.int64()),
    ("schema_version", pa.string()),
    ("extra_compressed", pa.binary()),
    ("extra_sha256", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")),
])

CellKey = tuple[str, str, int, str]  # (list_name, env_name, task_id, teacher_name)


def _manifest_key(prefix: str) -> str:
    p = prefix.rstrip("/")
    return f"{p}/manifest.jsonl" if p else "manifest.jsonl"


def _metadata_key(prefix: str) -> str:
    """A small pointer file consumed by external consumers (miners,
    validators) so they can enumerate task_idx without paying the
    cost of pulling the full manifest.

    Schema (matches the SWE-Infinite convention):
        {"version": 1,
         "last_updated": "<ISO8601>",
         "tasks": {"total": <int>, ...optional counters}}
    """
    p = prefix.rstrip("/")
    return f"{p}/metadata.json" if p else "metadata.json"


def _task_object_key(prefix: str, task_idx: int) -> str:
    p = prefix.rstrip("/")
    leaf = f"tasks/{task_idx:08d}.parquet"
    return f"{p}/{leaf}" if p else leaf


async def publish_rollouts(params: PublishParams) -> PublishResult:
    """One publish cycle. See module docstring for the immutability rules."""
    cfg = load_config()
    if not cfg.blob.bucket or not cfg.blob.endpoint:
        raise RuntimeError(
            "publisher requires AFR_BLOB__BUCKET + AFR_BLOB__ENDPOINT "
            "(R2 / S3-compatible store)"
        )

    sm = get_sessionmaker(cfg)
    blob = S3BlobStore(cfg.blob)
    prefix = cfg.blob.prefix.rstrip("/")

    # 1) Load existing manifest. ``known`` lets us recover from a prior
    # crash where we PUT manifest but didn't UPDATE published_at: those
    # cells reappear in the query below and we'll just mark them.
    manifest_lines, known, next_idx = await _load_manifest(blob, prefix)
    log.info("publisher.manifest.loaded", existing=len(known), next_idx=next_idx)

    # 2) Pending = frozen + not yet marked published.
    cells = await _list_pending_cells(sm, params)
    log.info("publisher.cells.pending", n=len(cells))

    new_lines: list[str] = []
    cells_to_mark: list[CellKey] = []
    n_committed = 0
    n_recovered = 0
    skipped_degenerate = 0
    skipped_empty = 0

    for cell in cells:
        key: CellKey = (cell.list_name, cell.env_name, cell.task_id, cell.teacher_name)

        # Recovery path: manifest already has this cell — last cycle
        # died before marking PG. Just mark and move on.
        if key in known:
            cells_to_mark.append(key)
            n_recovered += 1
            continue

        rows = await _fetch_cell_rows(sm, cell)
        if not rows:
            skipped_empty += 1
            cells_to_mark.append(key)  # also mark to avoid re-query
            continue

        rewards = [float(r.reward) for r in rows if r.reward is not None]
        reward_std = statistics.pstdev(rewards) if len(rewards) >= 2 else 0.0
        reward_mean = statistics.fmean(rewards) if rewards else 0.0
        if reward_std < params.min_reward_std:
            skipped_degenerate += 1
            cells_to_mark.append(key)  # decided final: skip; don't re-query
            continue

        table = _rows_to_table(rows)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")

        task_idx = next_idx
        object_key = _task_object_key(prefix, task_idx)
        object_uri = await blob.put(
            object_key, buf.getvalue(),
            content_type="application/octet-stream",
        )

        now = datetime.now(timezone.utc)
        mature_at = now + timedelta(seconds=params.maturation_window_s)
        entry = {
            "task_idx": task_idx,
            "list_name": cell.list_name,
            "env_name": cell.env_name,
            "task_id": cell.task_id,
            "teacher_name": cell.teacher_name,
            "n_rollouts": len(rows),
            "attempts": cell.attempts,
            "reward_mean": round(reward_mean, 6),
            "reward_std": round(reward_std, 6),
            "object_uri": object_uri,
            "object_key": object_key,
            "committed_at": now.isoformat(),
            "mature_at": mature_at.isoformat(),
        }
        new_lines.append(json.dumps(entry, sort_keys=True))
        known.add(key)
        cells_to_mark.append(key)
        next_idx += 1
        n_committed += 1

        log.info(
            "publisher.cell.committed",
            task_idx=task_idx,
            list_name=cell.list_name, env_name=cell.env_name,
            task_id=cell.task_id, teacher_name=cell.teacher_name,
            n=len(rows), reward_std=reward_std,
        )
        if n_committed >= params.max_new_per_cycle:
            log.info("publisher.cycle.cap", capped_at=params.max_new_per_cycle)
            break

    # 3) PUT manifest first (atomic single-object overwrite). If we
    # crash before step 4 the manifest is correct and PG will catch up
    # next cycle via the ``known`` recovery branch above.
    if new_lines:
        new_manifest = "\n".join([*manifest_lines, *new_lines]).encode("utf-8")
        manifest_uri = await blob.put(
            _manifest_key(prefix), new_manifest,
            content_type="application/jsonl",
        )
    else:
        manifest_uri = f"s3://{cfg.blob.bucket}/{_manifest_key(prefix)}"

    # 3b) PUT the thin metadata.json pointer so external consumers can
    # discover the published task range without downloading the full
    # manifest. Rewritten every cycle even when nothing was added so
    # the ``last_updated`` timestamp stays meaningful.
    metadata_body = json.dumps({
        "version": 1,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tasks": {"total": next_idx},
    }, sort_keys=True).encode("utf-8")
    await blob.put(
        _metadata_key(prefix), metadata_body,
        content_type="application/json",
    )

    # 4) Mark PG. One UPDATE per cell — ok at expected throughput
    # (~hundreds/cycle). Could be batched with VALUES if it becomes a
    # bottleneck.
    if cells_to_mark:
        async with sm() as session:
            for k in cells_to_mark:
                await session.execute(
                    update(sp_t)
                    .where(and_(
                        sp_t.c.list_name == k[0],
                        sp_t.c.env_name == k[1],
                        sp_t.c.task_id == k[2],
                        sp_t.c.teacher_name == k[3],
                    ))
                    .values(published_at=func.now())
                )
            await session.commit()

    return PublishResult(
        n_committed=n_committed,
        n_recovered=n_recovered,
        manifest_uri=manifest_uri,
        total_tasks_after=next_idx,
        skipped_degenerate=skipped_degenerate,
        skipped_empty=skipped_empty,
    )


# --------------------------------------------------------------------------- #
# Manifest IO
# --------------------------------------------------------------------------- #


async def _load_manifest(blob, prefix: str):  # type: ignore[no-untyped-def]
    uri = f"s3://{blob._cfg.bucket}/{_manifest_key(prefix)}"  # noqa: SLF001
    if not await blob.exists(uri):
        return [], set(), 0
    data = await blob.get(uri)
    text_body = data.decode("utf-8").strip()
    lines = text_body.splitlines() if text_body else []
    known: set[CellKey] = set()
    for line in lines:
        try:
            obj = json.loads(line)
            known.add((
                str(obj["list_name"]),
                str(obj["env_name"]),
                int(obj["task_id"]),
                str(obj["teacher_name"]),
            ))
        except (json.JSONDecodeError, KeyError):
            log.warning("publisher.manifest.malformed_line", line=line[:200])
    return lines, known, len(lines)


# --------------------------------------------------------------------------- #
# PG queries
# --------------------------------------------------------------------------- #


@dataclass
class _Cell:
    list_name: str
    env_name: str
    task_id: int
    teacher_name: str
    target_samples: int
    attempts: int
    collected: int


async def _list_pending_cells(sm, params: PublishParams) -> list[_Cell]:
    """Frozen + unpublished cells.

    Frozen condition matches ``claim_next_cell``'s "no new attempts
    possible" predicate:

        collected >= target_samples            (already enough successes)
        OR attempts >= 2 * target_samples      (attempt budget spent)

    ``attempts`` increments at *claim* time but ``collected`` only at
    *completion* time, so for high-concurrency (high sampling-weight) envs a
    fresh cell's ``attempts`` can race past ``2 * target_samples`` while its
    rollouts are still in flight. Publishing then reads ~0 rows, marks the
    cell ``skipped_empty`` + ``published_at``, and the variance-bearing samples
    that land moments later are lost forever. Gate the attempts branch on a
    settle window (no ``last_updated`` activity) so we only freeze-by-attempts
    once the in-flight rollouts have actually resolved.
    """
    conds = [
        sp_t.c.published_at.is_(None),
        or_(
            sp_t.c.collected >= sp_t.c.target_samples,
            and_(
                sp_t.c.attempts >= 2 * sp_t.c.target_samples,
                sp_t.c.last_updated < func.now() - timedelta(minutes=5),
            ),
        ),
    ]
    if params.list_names:
        conds.append(sp_t.c.list_name.in_(params.list_names))
    if params.env_names:
        conds.append(sp_t.c.env_name.in_(params.env_names))
    if params.teacher_names:
        conds.append(sp_t.c.teacher_name.in_(params.teacher_names))

    stmt = (
        select(
            sp_t.c.list_name, sp_t.c.env_name, sp_t.c.task_id,
            sp_t.c.teacher_name, sp_t.c.target_samples,
            sp_t.c.attempts, sp_t.c.collected,
        )
        .where(and_(*conds))
        .order_by(sp_t.c.last_updated, sp_t.c.list_name, sp_t.c.env_name,
                  sp_t.c.task_id, sp_t.c.teacher_name)
    )

    out: list[_Cell] = []
    async with sm() as session:
        result = await session.execute(stmt)
        for row in result:
            out.append(_Cell(
                list_name=row.list_name, env_name=row.env_name,
                task_id=int(row.task_id), teacher_name=row.teacher_name,
                target_samples=int(row.target_samples),
                attempts=int(row.attempts),
                collected=int(row.collected),
            ))
    return out


async def _fetch_cell_rows(sm, cell: _Cell):  # type: ignore[no-untyped-def]
    """Pull the ok rollouts that compose this cell.

    ``sample_idx < attempts`` bounds the read to the slots actually
    handed out for this cell — defensive against any stale rows.
    """
    stmt = (
        select(
            rollouts_t.c.rollout_id, rollouts_t.c.env_name, rollouts_t.c.task_id,
            rollouts_t.c.teacher_name, rollouts_t.c.sample_idx, rollouts_t.c.status,
            rollouts_t.c.reward, rollouts_t.c.temperature, rollouts_t.c.top_p,
            rollouts_t.c.seed, rollouts_t.c.steps, rollouts_t.c.tokens_in,
            rollouts_t.c.tokens_out, rollouts_t.c.schema_version,
            rollouts_t.c.extra_compressed, rollouts_t.c.extra_sha256,
            rollouts_t.c.created_at,
        )
        .where(and_(
            rollouts_t.c.status == "ok",
            rollouts_t.c.env_name == cell.env_name,
            rollouts_t.c.task_id == cell.task_id,
            rollouts_t.c.teacher_name == cell.teacher_name,
            rollouts_t.c.sample_idx < cell.attempts,
        ))
        .order_by(rollouts_t.c.sample_idx)
    )
    async with sm() as session:
        result = await session.execute(stmt)
        return list(result.fetchall())


def _rows_to_table(rows: Sequence) -> pa.Table:  # type: ignore[type-arg]
    def _col(name: str):  # type: ignore[no-untyped-def]
        return [getattr(r, name) for r in rows]

    def _str_col(name: str) -> list[str | None]:
        return [None if getattr(r, name) is None else str(getattr(r, name)) for r in rows]

    return pa.table(
        {
            "rollout_id": _str_col("rollout_id"),
            "env_name": _str_col("env_name"),
            "task_id": _col("task_id"),
            "teacher_name": _str_col("teacher_name"),
            "sample_idx": _col("sample_idx"),
            "status": _str_col("status"),
            "reward": _col("reward"),
            "temperature": _col("temperature"),
            "top_p": _col("top_p"),
            "seed": _col("seed"),
            "steps": _col("steps"),
            "tokens_in": _col("tokens_in"),
            "tokens_out": _col("tokens_out"),
            "schema_version": _str_col("schema_version"),
            "extra_compressed": _col("extra_compressed"),
            "extra_sha256": _col("extra_sha256"),
            "created_at": _col("created_at"),
        },
        schema=_CELL_PARQUET_SCHEMA,
    )


# --------------------------------------------------------------------------- #
# Long-running loop
# --------------------------------------------------------------------------- #


async def publisher_loop(*, interval_s: float = 300.0) -> None:
    """One process, two stages per cycle.

    Stage 1 (publish): drain frozen cells from PG into the **private**
    bucket. Always fresh, always immutable.

    Stage 2 (promote): any cell that's been in the private bucket past
    its maturation window gets server-side copied (R2 CopyObject) to
    the **public** bucket, and its public manifest line is appended.
    """
    from affine_opeg.publishing.promoter import (
        params_from_env as promote_params_from_env,
        promote_mature,
    )

    log.info("publisher.loop.start", interval_s=interval_s)
    while True:
        try:
            pub = await publish_rollouts(_params_from_env())
            log.info(
                "publisher.cycle.publish_done",
                n_committed=pub.n_committed,
                n_recovered=pub.n_recovered,
                total_tasks=pub.total_tasks_after,
                skipped_degenerate=pub.skipped_degenerate,
                skipped_empty=pub.skipped_empty,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("publisher.publish_failed", error=str(exc)[:400])

        try:
            prm = await promote_mature(promote_params_from_env())
            log.info(
                "publisher.cycle.promote_done",
                n_promoted=prm.n_promoted,
                public_manifest_uri=prm.public_manifest_uri,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("publisher.promote_failed", error=str(exc)[:400])

        await asyncio.sleep(interval_s)


def _params_from_env() -> PublishParams:
    """Build PublishParams from AFR_PUBLISH_* env vars.

    Recognised:
        AFR_PUBLISH_LIST_NAMES        CSV; restrict to these sampling lists
        AFR_PUBLISH_ENV_NAMES         CSV; restrict to these env names
        AFR_PUBLISH_TEACHER_NAMES     CSV; restrict to these teachers
        AFR_PUBLISH_MIN_REWARD_STD    float; cells below this are dropped
                                       (default 0.0; raise to skip degenerate cells)
        AFR_PUBLISH_MAX_NEW_PER_CYCLE upper bound on shards per cycle (default 200)
    """
    return PublishParams(
        list_names=_csv("AFR_PUBLISH_LIST_NAMES"),
        env_names=_csv("AFR_PUBLISH_ENV_NAMES"),
        teacher_names=_csv("AFR_PUBLISH_TEACHER_NAMES"),
        min_reward_std=_float("AFR_PUBLISH_MIN_REWARD_STD", default=0.05),
        max_new_per_cycle=_int("AFR_PUBLISH_MAX_NEW_PER_CYCLE", default=200),
        maturation_window_s=_int("AFR_PUBLISH_MATURATION_S", default=0),
    )


def _csv(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _int(name: str, *, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, *, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
