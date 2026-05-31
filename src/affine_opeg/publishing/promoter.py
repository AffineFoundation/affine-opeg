"""Promote mature shards from the private bucket to the public bucket.

Lifecycle on R2:

    cell published_at <= now AND mature_at > now    (fresh)
        → only in ``cfg.blob.bucket`` (private)
        → ``sampling_progress.promoted_at IS NULL``

    cell mature_at <= now                            (mature)
        → mirrored to ``cfg.blob.public_bucket``
        → ``sampling_progress.promoted_at IS NOT NULL``

The mirror is **append-only on the public side**: the parquet shard
is copied via R2 CopyObject (server-side, no roundtrip through this
host), and the public manifest gets one extra line per cell.

Crash safety:
    1. CopyObject (idempotent, parquet is immutable)
    2. PUT public manifest (atomic single-object overwrite)
    3. UPDATE sampling_progress.promoted_at = now()

Any crash leaves the system recoverable on the next cycle (see
publisher.py for the same pattern on the publish path).

This module is tightly coupled to ``publisher`` — both share the
``CellKey`` / manifest schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, select, update

from affine_opeg.adapters.blob_stores.s3 import S3BlobStore
from affine_opeg.adapters.metadata_stores.sqlalchemy_pg.orm import (
    sampling_progress as sp_t,
)
from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import get_sessionmaker
from affine_opeg.infrastructure.logging import get_logger
from affine_opeg.publishing.publisher import (
    _manifest_key,
    _metadata_key,
    _task_object_key,
)

log = get_logger("publisher.promoter")


@dataclass(frozen=True)
class PromoteParams:
    max_per_cycle: int = 200
    # Rate-based release: cap how many cells are promoted per rolling
    # 24h window. When the cap is hit the promoter just sits on the
    # backlog until tomorrow — no per-cell maturation delay needed.
    # Set to 0 to disable the cap (release as fast as the publisher
    # feeds the backlog).
    max_per_day: int = 100


@dataclass(frozen=True)
class PromoteResult:
    n_promoted: int
    public_manifest_uri: str | None
    skipped_immature: int


async def promote_mature(params: PromoteParams) -> PromoteResult:
    """One promote cycle: copy mature cells private → public."""
    cfg = load_config()
    if not cfg.blob.bucket or not cfg.blob.public_bucket or not cfg.blob.endpoint:
        raise RuntimeError(
            "promoter requires AFR_BLOB__BUCKET + AFR_BLOB__PUBLIC_BUCKET + "
            "AFR_BLOB__ENDPOINT"
        )
    if cfg.blob.bucket == cfg.blob.public_bucket:
        raise RuntimeError(
            f"private and public buckets must differ; both are "
            f"{cfg.blob.bucket!r}. Set AFR_BLOB__PUBLIC_BUCKET to a "
            f"different value."
        )

    sm = get_sessionmaker(cfg)
    blob = S3BlobStore(cfg.blob)
    prefix = cfg.blob.prefix.rstrip("/")
    private_bucket = cfg.blob.bucket
    public_bucket = cfg.blob.public_bucket

    # 1) Pull the existing public manifest (start of cycle).
    public_manifest_lines, public_known = await _load_public_manifest(
        blob, prefix, public_bucket,
    )

    # 2) Find manifest entries that aren't in the public manifest yet.
    candidates = await _list_ready_to_promote(
        sm, params.max_per_cycle, params.max_per_day,
        already_promoted=public_known,
    )
    log.info("promoter.candidates", n=len(candidates))

    appended_lines: list[str] = []
    keys_to_mark: list[tuple] = []
    n_promoted = 0

    for cand in candidates:
        cell_key = (cand.list_name, cand.env_name, cand.task_id, cand.teacher_name)
        if cell_key in public_known:
            # Crash-recovery: object already promoted on a prior cycle
            # but PG mark didn't land. Just refresh the PG mark.
            keys_to_mark.append(cell_key)
            continue

        object_key = _task_object_key(prefix, cand.task_idx)
        # 3) Server-side cross-bucket copy (R2 supports S3 CopyObject).
        await _copy_object(
            blob, src_bucket=private_bucket, src_key=object_key,
            dst_bucket=public_bucket, dst_key=object_key,
        )

        entry = {
            "task_idx": cand.task_idx,
            "list_name": cand.list_name,
            "env_name": cand.env_name,
            "task_id": cand.task_id,
            "teacher_name": cand.teacher_name,
            "n_rollouts": cand.n_rollouts,
            "attempts": cand.attempts,
            "reward_mean": cand.reward_mean,
            "reward_std": cand.reward_std,
            "object_uri": f"s3://{public_bucket}/{object_key}",
            "object_key": object_key,
            "committed_at": cand.committed_at.isoformat(),
            "mature_at": cand.mature_at.isoformat(),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        appended_lines.append(json.dumps(entry, sort_keys=True))
        public_known.add(cell_key)
        keys_to_mark.append(cell_key)
        n_promoted += 1
        log.info(
            "promoter.cell.copied",
            task_idx=cand.task_idx, env_name=cand.env_name,
            task_id=cand.task_id, teacher_name=cand.teacher_name,
        )

    # 4) Rewrite public manifest atomically.
    public_uri: str | None = None
    if appended_lines:
        new_body = "\n".join([*public_manifest_lines, *appended_lines]).encode("utf-8")
        public_uri = await _put_object(
            blob, bucket=public_bucket,
            key=_manifest_key(prefix), body=new_body,
            content_type="application/jsonl",
        )

    # 4b) Refresh the thin public metadata.json. ``completed_up_to`` is
    # the count of mature cells (= entries in the public manifest);
    # ``staged_up_to`` is what publisher has put into the private
    # bucket so consumers can see the in-flight pipeline depth.
    # Rewritten every cycle so ``last_updated`` stays meaningful even
    # when no new promotions occurred.
    completed_up_to = len(public_manifest_lines) + len(appended_lines)
    staged_up_to = await _private_manifest_count(blob, prefix, private_bucket)
    metadata_body = json.dumps({
        "version": 1,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tasks": {
            "staged_up_to": staged_up_to,
            "completed_up_to": completed_up_to,
        },
    }, sort_keys=True).encode("utf-8")
    await _put_object(
        blob, bucket=public_bucket,
        key=_metadata_key(prefix), body=metadata_body,
        content_type="application/json",
    )

    # 5) Mark PG promoted_at for every cell we processed (including
    # recovery-only ones from the known-set hit above).
    if keys_to_mark:
        async with sm() as session:
            for k in keys_to_mark:
                await session.execute(
                    update(sp_t).where(and_(
                        sp_t.c.list_name == k[0],
                        sp_t.c.env_name == k[1],
                        sp_t.c.task_id == k[2],
                        sp_t.c.teacher_name == k[3],
                    )).values(promoted_at=func.now())
                )
            await session.commit()

    return PromoteResult(
        n_promoted=n_promoted,
        public_manifest_uri=public_uri,
        skipped_immature=0,
    )


# --------------------------------------------------------------------------- #
# Public manifest IO
# --------------------------------------------------------------------------- #


async def _private_manifest_count(blob, prefix: str, private_bucket: str) -> int:
    """Cheap count of entries in the private manifest, used as the
    ``staged_up_to`` value advertised through the public metadata.json.
    Returns 0 when the manifest does not exist yet (cold start)."""
    body = await _get_object_or_none(blob, private_bucket, _manifest_key(prefix))
    if not body:
        return 0
    text_body = body.decode("utf-8").strip()
    if not text_body:
        return 0
    return sum(1 for line in text_body.splitlines() if line.strip())


async def _load_public_manifest(blob, prefix: str, public_bucket: str):
    """Read the existing public manifest from ``public_bucket``.

    Returns ``(raw_lines, known_cell_keys)`` so the promoter can append
    without re-emitting existing entries and skip cells that crash-survived
    a previous cycle.
    """
    key = _manifest_key(prefix)
    body = await _get_object_or_none(blob, public_bucket, key)
    if body is None:
        return [], set()
    text_body = body.decode("utf-8").strip()
    if not text_body:
        return [], set()
    lines = text_body.splitlines()
    known: set[tuple] = set()
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
            log.warning("promoter.public_manifest.malformed_line", line=line[:200])
    return lines, known


# --------------------------------------------------------------------------- #
# PG candidate query
# --------------------------------------------------------------------------- #


@dataclass
class _Candidate:
    task_idx: int
    list_name: str
    env_name: str
    task_id: int
    teacher_name: str
    attempts: int
    n_rollouts: int           # = collected at publish time (read from manifest entry)
    reward_mean: float
    reward_std: float
    committed_at: datetime
    mature_at: datetime


async def _list_ready_to_promote(
    sm, max_per_cycle: int, max_per_day: int,
    *,
    already_promoted: set,
) -> list[_Candidate]:
    """Pending-promote cells: present in the private manifest + not yet
    in the public manifest.

    The source of truth for "promotable" is the private manifest;
    "already-promoted" is the public manifest. Both come from R2, so
    the DB ``sampling_progress`` table can diverge (generator-side
    hooks like ``freeze_degenerate_cell`` set ``published_at`` without
    a manifest entry, and historical row deletions can leave manifest
    keys with no DB peer) without breaking the promoter — we just use
    R2 as the authority for both sides of the decision.

    The DB is still read for the rate-cap window (``promoted_at`` in
    the last 24h), because it has per-cell timestamps the manifest
    doesn't carry historically.
    """
    cfg = load_config()
    blob = S3BlobStore(cfg.blob)
    prefix = cfg.blob.prefix.rstrip("/")
    body = await _get_object_or_none(blob, cfg.blob.bucket, _manifest_key(prefix))
    if body is None:
        return []

    manifest_entries: list[dict] = []
    for line in body.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            manifest_entries.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("promoter.private_manifest.malformed_line", line=line[:200])
            continue
    if not manifest_entries:
        return []

    now = datetime.now(timezone.utc)
    # Rate-based release: count how many cells have been promoted in
    # the last 24h, deduct from the daily cap to get this cycle's
    # remaining budget. ``max_per_day=0`` disables the cap entirely.
    today_count: int = 0
    if max_per_day > 0:
        async with sm() as session:
            today_count = int(await session.scalar(
                select(func.count(sp_t.c.task_id)).where(
                    sp_t.c.promoted_at.is_not(None)
                    & (sp_t.c.promoted_at > now - timedelta(hours=24))
                )
            ) or 0)
        budget = max(0, max_per_day - today_count)
        cap = min(max_per_cycle, budget)
    else:
        cap = max_per_cycle
    if cap <= 0:
        log.info("promoter.rate_capped", max_per_day=max_per_day,
                 promoted_last_24h=today_count)
        return []

    # Walk manifest in publish order (it's already append-ordered by
    # task_idx, which mirrors publisher commit order). Take the first
    # ``cap`` entries that aren't yet in the public manifest.
    out: list[_Candidate] = []
    for entry in manifest_entries:
        try:
            key = (
                str(entry["list_name"]), str(entry["env_name"]),
                int(entry["task_id"]), str(entry["teacher_name"]),
            )
        except (KeyError, TypeError, ValueError):
            log.warning("promoter.private_manifest.bad_entry", entry=entry)
            continue
        if key in already_promoted:
            continue
        committed_at = _parse_iso(entry["committed_at"])
        try:
            mature_at = _parse_iso(entry.get("mature_at") or entry["committed_at"])
        except Exception:
            mature_at = committed_at
        out.append(_Candidate(
            task_idx=int(entry["task_idx"]),
            list_name=key[0], env_name=key[1],
            task_id=key[2], teacher_name=key[3],
            attempts=int(entry.get("attempts", 0)),
            n_rollouts=int(entry["n_rollouts"]),
            reward_mean=float(entry["reward_mean"]),
            reward_std=float(entry["reward_std"]),
            committed_at=committed_at, mature_at=mature_at,
        ))
        if len(out) >= cap:
            break
    return out


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# --------------------------------------------------------------------------- #
# Cross-bucket copy and IO helpers (S3BlobStore doesn't have these natively;
# we go through the underlying session.)
# --------------------------------------------------------------------------- #


async def _copy_object(blob, *, src_bucket: str, src_key: str,
                       dst_bucket: str, dst_key: str) -> None:
    async with blob._session.client("s3", **blob._client_kwargs()) as s3:  # noqa: SLF001
        await s3.copy_object(
            Bucket=dst_bucket, Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
        )


async def _put_object(blob, *, bucket: str, key: str, body: bytes,
                      content_type: str) -> str:
    async with blob._session.client("s3", **blob._client_kwargs()) as s3:  # noqa: SLF001
        await s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    return f"s3://{bucket}/{key}"


async def _get_object_or_none(blob, bucket: str, key: str) -> bytes | None:
    async with blob._session.client("s3", **blob._client_kwargs()) as s3:  # noqa: SLF001
        try:
            obj = await s3.get_object(Bucket=bucket, Key=key)
        except s3.exceptions.NoSuchKey:
            return None
        async with obj["Body"] as stream:
            return await stream.read()


# --------------------------------------------------------------------------- #
# Env helper for publisher_loop integration
# --------------------------------------------------------------------------- #


def params_from_env() -> PromoteParams:
    raw = os.environ.get("AFR_PROMOTE_MAX_PER_CYCLE", "").strip()
    try:
        max_per_cycle = int(raw) if raw else 200
    except ValueError:
        max_per_cycle = 200
    raw_day = os.environ.get("AFR_PROMOTE_MAX_PER_DAY", "").strip()
    try:
        max_per_day = int(raw_day) if raw_day else 100
    except ValueError:
        max_per_day = 100
    return PromoteParams(max_per_cycle=max_per_cycle, max_per_day=max_per_day)
