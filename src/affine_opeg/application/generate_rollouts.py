"""Use case: produce one or many rollouts.

A rollout is produced for a single ``(env_name, task_id, teacher_name, sample_idx)``
key. The orchestration here is:

    1. resolve the Task and Teacher (DB lookups)
    2. acquire a sandbox scoped to the task (workspace ready, network sealed)
    3. run the agent loop — handed off to a provider-agnostic
       ``AgentLoopFn`` (default: affent CLI inside the sandbox container)
    4. normalize the resulting trajectory
    5. compress, persist a Rollout row, and (async) archive the blob

The agent loop itself is in ``adapters/sandboxes/affent_loop.py`` —
affent is OpenAI-chat-compat and speaks to every teacher endpoint
uniformly, so this module never imports a provider SDK.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable
from uuid import uuid4

from sqlalchemy import text

from affine_opeg.domain.errors import NormalizationError, TeacherError
from affine_opeg.domain.ids import EnvName, SamplingListName, TaskId, TeacherName
from affine_opeg.domain.models import (
    RawTrajectory,
    Rollout,
    RolloutStatus,
    Task,
    Teacher,
)
from affine_opeg.domain.ports import (
    MetadataStore,
    Sandbox,
    SandboxFactory,
    TrajectoryNormalizer,
)
from affine_opeg.infrastructure.compression import compress_json
from affine_opeg.infrastructure.logging import current_trace_id, get_logger, trace_context
from affine_opeg.infrastructure.metrics import metrics

log = get_logger("rollout")

# Provider-agnostic agent loop. Receives the Teacher metadata (endpoint,
# api_key_env, model name) and the prepared Sandbox; returns the RawTrajectory.
AgentLoopFn = Callable[[Teacher, Sandbox, "RolloutParams"], Awaitable[RawTrajectory]]


@dataclass(frozen=True)
class RolloutParams:
    env_name: EnvName
    task_id: TaskId
    teacher_name: TeacherName
    sample_idx: int
    temperature: float
    top_p: float | None = None
    seed: int | None = None
    max_steps: int = 40
    # Owning sampling_list — required so we can bump
    # ``sampling_progress.collected`` in the same transaction as the
    # rollout insert when status='ok'. When None (e.g. ad-hoc CLI
    # invocations not bound to a list) the collected bump is skipped.
    list_name: SamplingListName | None = None


@dataclass(frozen=True)
class GenerateRolloutDeps:
    """Bundle of ports for the use case. Wired up by the worker bootstrap."""

    metadata: MetadataStore
    sandbox: SandboxFactory
    normalizer: TrajectoryNormalizer
    agent_loop: AgentLoopFn


async def generate_rollout(
    deps: GenerateRolloutDeps,
    params: RolloutParams,
    *,
    producer_id: str | None = None,
) -> Rollout:
    """Produce one rollout end-to-end. Persists exactly one ``rollouts`` row.

    Raises only if persistence itself fails — agent loop / normalize failures
    are turned into status=='parse_failed' / 'teacher_error' rows so they're
    still observable.
    """
    producer_id = producer_id or f"producer-{socket.gethostname()}-{uuid4().hex[:6]}"

    with trace_context(rollout_key=f"{params.env_name}:{params.task_id}:{params.teacher_name}:{params.sample_idx}"):
        task = await _load_task(deps.metadata, params.env_name, params.task_id)
        teacher = await _load_teacher(deps.metadata, params.teacher_name)

        status, raw, err = await _run_loop(deps, params, task, teacher)
        normalized, normalize_err = _normalize(deps.normalizer, raw) if raw else (None, None)
        if normalize_err is not None:
            status = RolloutStatus.parse_failed

        # Zero comparable tokens -> mark non-ok so the sample is excluded from
        # ``collected`` and from the published cell (see RolloutStatus docs).
        if status == RolloutStatus.ok and not _has_comparable_content(normalized):
            status = RolloutStatus.empty_completion
            log.warning(
                "rollout.empty_completion",
                env_name=str(params.env_name), task_id=int(params.task_id),
                teacher_name=str(params.teacher_name),
            )

        # We always have *some* payload to persist — fall back to a stub
        # describing the failure so the row is still parseable.
        if normalized is None:
            from affine_opeg.domain.models import NormalizedTrajectory

            normalized = NormalizedTrajectory(
                schema_version=deps.normalizer.schema_version,
                messages=[],
                reward_breakdown={},
                teacher_meta={"failure": str(err or normalize_err)},
            )

        blob, sha = compress_json(normalized.model_dump(mode="json"))

        rollout_id = uuid4()
        rollout = Rollout(
            rollout_id=rollout_id,  # type: ignore[arg-type]
            env_name=params.env_name,
            task_id=params.task_id,
            teacher_name=params.teacher_name,
            sample_idx=params.sample_idx,
            temperature=params.temperature,
            top_p=params.top_p,
            seed=params.seed,
            status=status,
            reward=_reward(normalized.reward_breakdown),
            reward_breakdown=normalized.reward_breakdown or None,
            steps=_step_count(raw),
            latency_ms=normalized.teacher_meta.get("latency_ms"),
            tokens_in=normalized.teacher_meta.get("tokens_in"),
            tokens_out=normalized.teacher_meta.get("tokens_out"),
            cost_usd=normalized.teacher_meta.get("cost_usd"),
            schema_version=normalized.schema_version,
            extra_compressed=blob,
            extra_sha256=sha,
            blob_uri=None,  # filled in by archiver
            group_label=None,
            producer_id=producer_id,
            trace_id=current_trace_id(),
        )

        async with deps.metadata.unit_of_work() as uow:
            await uow.rollouts.insert(rollout)
            # Bump the cell's success counter in the same transaction
            # so a crash between insert and bump is impossible. Only
            # status='ok' rows count toward ``collected``; failed
            # attempts have already consumed their sample_idx via
            # claim_next_cell's ``attempts`` bump.
            froze = False
            if status == RolloutStatus.ok and params.list_name is not None:
                await uow.sampling_lists.increment_collected(
                    params.list_name, params.env_name,
                    params.task_id, params.teacher_name, delta=1,
                )
                # Early-freeze degenerate cells so we don't keep burning
                # samples on a (task, teacher) tuple that is producing
                # the same reward every time. The publisher's variance
                # filter would drop these anyway — save the compute now.
                # Both the rollout insert and increment above used
                # session.execute(), so the SELECT inside this UPDATE
                # already sees the new row.
                froze = await uow.sampling_lists.freeze_degenerate_cell(
                    params.list_name, params.env_name,
                    params.task_id, params.teacher_name,
                    min_samples=4,
                )
            await uow.commit()
            if froze:
                log.info(
                    "rollout.cell_frozen_degenerate",
                    env_name=params.env_name,
                    task_id=params.task_id,
                    teacher_name=params.teacher_name,
                )

        log.info(
            "rollout.persisted",
            rollout_id=str(rollout_id),
            status=status.value,
            env_name=params.env_name,
            task_id=params.task_id,
            teacher_name=params.teacher_name,
            sample_idx=params.sample_idx,
        )
        metrics().incr(
            "rollouts.count",
            labels={"teacher": str(params.teacher_name), "status": status.value, "env": str(params.env_name)},
        )

        # NOTE: no R2 archive write on the generator side. The
        # zstd-compressed trajectory is already inline in
        # ``rollouts.extra_compressed`` (PG), which the publisher reads
        # when it builds per-cell parquet shards. Adding a second copy
        # at ``rollouts/{rollout_id}.zst`` was the original reborn
        # backup path and is intentionally retired.

        return rollout


# --------------------------------------------------------------------------- #


async def _load_task(metadata: MetadataStore, env_name: EnvName, task_id: TaskId) -> Task:
    async with metadata.unit_of_work() as uow:
        task = await uow.tasks.get_task(env_name, task_id)
        if task is None:
            raise ValueError(f"unknown task: {env_name}:{task_id}")
        return task


async def _load_teacher(metadata: MetadataStore, teacher_name: TeacherName) -> Teacher:
    async with metadata.unit_of_work() as uow:
        teacher = await uow.teachers.get(teacher_name)
        if teacher is None:
            raise ValueError(f"unknown teacher: {teacher_name}")
        return teacher


async def _run_loop(
    deps: GenerateRolloutDeps, params: RolloutParams, task: Task, teacher: Teacher,
) -> tuple[RolloutStatus, RawTrajectory | None, str | None]:
    try:
        async with deps.sandbox.acquire(task) as sb:
            raw = await deps.agent_loop(teacher, sb, params)
            return RolloutStatus.ok, raw, None
    except TeacherError as e:
        log.warning("rollout.teacher_error", error=str(e))
        return RolloutStatus.teacher_error, None, str(e)
    except Exception as e:  # noqa: BLE001
        log.warning("rollout.env_error", error=str(e))
        return RolloutStatus.env_error, None, str(e)


def _normalize(
    normalizer: TrajectoryNormalizer, raw: RawTrajectory,
):
    try:
        return normalizer.normalize(raw), None
    except NormalizationError as e:
        log.warning("rollout.normalize_failed", error=str(e))
        return None, str(e)


def _has_comparable_content(normalized) -> bool:  # type: ignore[no-untyped-def]
    """True if any assistant message carries a loss-bearing token span.

    Mirrors the eval renderer's loss mask (``assistant_loss`` = non-empty
    ``content`` + ``tool_calls``; ``reasoning`` is masked). A trajectory with
    no such span renders to an all-zero loss mask -> nothing to compare.
    """
    if normalized is None:
        return False
    for msg in normalized.messages:
        if getattr(msg, "role", None) != "assistant":
            continue
        if (msg.content or "").strip():
            return True
        if getattr(msg, "tool_calls", None):
            return True
    return False


def _reward(breakdown: dict) -> float | None:
    if not breakdown:
        return None
    if "score" in breakdown:
        return float(breakdown["score"])
    if "tests_passed" in breakdown and "tests_total" in breakdown:
        total = breakdown["tests_total"]
        return float(breakdown["tests_passed"]) / max(total, 1)
    return None


def _step_count(raw: RawTrajectory | None) -> int | None:
    return len(raw.steps) if raw else None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
