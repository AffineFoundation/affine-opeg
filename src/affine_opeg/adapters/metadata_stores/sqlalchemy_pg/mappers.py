"""Row <-> domain model mapping.

Kept separate from repositories so each repository stays focused on queries.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from affine_opeg.domain.ids import (
    EnvName,
    PairId,
    PairSetName,
    Revision,
    RolloutId,
    RunId,
    SamplingListName,
    StudentName,
    TaskId,
    TeacherName,
)
from affine_opeg.domain.models import (
    AntiCopyResult,
    DeploymentStatus,
    Environment,
    GroupLabel,
    Pair,
    PairSet,
    PairSetStatus,
    Rollout,
    RolloutStatus,
    RunStatus,
    SamplingList,
    SamplingProgress,
    StudentDeployment,
    StudentScore,
    StudentSubmission,
    Task,
    Teacher,
)


def _g(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Safe getter for both dict rows and SQLAlchemy Row objects."""
    if hasattr(row, "_mapping"):
        return row._mapping.get(key, default)  # type: ignore[no-any-return]
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def _gj(row: Any, key: str, default: Any = None) -> Any:
    """JSONB-aware getter.

    asyncpg/SQLAlchemy returns either a Python ``dict``/``list`` (when the
    per-connection JSONB codec is wired) or the raw JSON string (when it
    isn't — depends on driver internals). Normalising at the mapper boundary
    means everything downstream sees the same shape.
    """
    val = _g(row, key, default)
    if isinstance(val, (bytes, bytearray)):
        try:
            val = val.decode()
        except UnicodeDecodeError:
            return default
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return default
    return val


def environment_from_row(row: Any) -> Environment:
    rng = _g(row, "task_id_range")
    lo, hi = _parse_range(rng)
    return Environment(
        env_name=EnvName(_g(row, "env_name")),
        dataset=_g(row, "dataset"),
        dataset_version=_g(row, "dataset_version"),
        task_id_min=lo,
        task_id_max=hi,
        meta=_gj(row, "meta") or {},
        created_at=_g(row, "created_at"),
    )


def _parse_range(rng: Any) -> tuple[int, int]:
    """asyncpg returns Range objects; SA may return strings. Normalize both."""
    if rng is None:
        return (0, 0)
    if hasattr(rng, "lower") and hasattr(rng, "upper"):
        return int(rng.lower or 0), int(rng.upper or 0)
    s = str(rng).strip("[](){}")
    parts = s.split(",")
    return int(parts[0]) if parts[0] else 0, int(parts[1]) if len(parts) > 1 and parts[1] else 0


def task_from_row(row: Any) -> Task:
    return Task(
        env_name=EnvName(_g(row, "env_name")),
        task_id=TaskId(_g(row, "task_id")),
        repo=_g(row, "repo"),
        base_commit=_g(row, "base_commit"),
        problem=_g(row, "problem"),
        hidden_tests=_gj(row, "hidden_tests") or {},
        difficulty=_g(row, "difficulty"),
        meta=_gj(row, "meta") or {},
    )


def teacher_from_row(row: Any) -> Teacher:
    return Teacher(
        teacher_name=TeacherName(_g(row, "teacher_name")),
        model_family=_g(row, "model_family"),
        provider=_g(row, "provider"),
        endpoint=_g(row, "endpoint"),
        api_key_env=_g(row, "api_key_env"),
        tool_format=_g(row, "tool_format"),
        reasoning_format=_g(row, "reasoning_format"),
        context_window=_g(row, "context_window"),
        price_per_mtoken_in=_g(row, "price_per_mtoken_in"),
        price_per_mtoken_out=_g(row, "price_per_mtoken_out"),
        active=_g(row, "active"),
        meta=_gj(row, "meta") or {},
    )


def rollout_from_row(row: Any) -> Rollout:
    gl = _g(row, "group_label")
    return Rollout(
        rollout_id=RolloutId(_g(row, "rollout_id")),
        env_name=EnvName(_g(row, "env_name")),
        task_id=TaskId(_g(row, "task_id")),
        teacher_name=TeacherName(_g(row, "teacher_name")),
        sample_idx=_g(row, "sample_idx"),
        temperature=_g(row, "temperature"),
        top_p=_g(row, "top_p"),
        seed=_g(row, "seed"),
        status=RolloutStatus(_g(row, "status")),
        reward=_g(row, "reward"),
        reward_breakdown=_gj(row, "reward_breakdown"),
        steps=_g(row, "steps"),
        latency_ms=_g(row, "latency_ms"),
        tokens_in=_g(row, "tokens_in"),
        tokens_out=_g(row, "tokens_out"),
        cost_usd=_g(row, "cost_usd"),
        schema_version=_g(row, "schema_version"),
        extra_compressed=bytes(_g(row, "extra_compressed")),
        extra_sha256=_g(row, "extra_sha256"),
        blob_uri=_g(row, "blob_uri"),
        group_label=GroupLabel(gl) if gl else None,
        producer_id=_g(row, "producer_id"),
        trace_id=_g(row, "trace_id"),
        created_at=_g(row, "created_at"),
    )


def pair_from_row(row: Any) -> Pair:
    return Pair(
        pair_id=PairId(_g(row, "pair_id")),
        pair_set_name=PairSetName(_g(row, "pair_set_name")),
        env_name=EnvName(_g(row, "env_name")),
        task_id=TaskId(_g(row, "task_id")),
        teacher_name=TeacherName(_g(row, "teacher_name")),
        win_rollout=RolloutId(_g(row, "win_rollout")),
        lose_rollout=RolloutId(_g(row, "lose_rollout")),
        reward_win=_g(row, "reward_win"),
        reward_lose=_g(row, "reward_lose"),
        reward_gap=_g(row, "reward_gap"),
        selection_meta=_gj(row, "selection_meta") or {},
    )


def pair_set_from_row(row: Any) -> PairSet:
    return PairSet(
        pair_set_name=PairSetName(_g(row, "pair_set_name")),
        source_filter=_gj(row, "source_filter") or {},
        strategy=_gj(row, "strategy") or {},
        rollouts_scanned=_g(row, "rollouts_scanned"),
        pairs_created=_g(row, "pairs_created"),
        groups_summary=_gj(row, "groups_summary"),
        status=PairSetStatus(_g(row, "status")),
        triggered_by=_g(row, "triggered_by"),
        started_at=_g(row, "started_at"),
        finished_at=_g(row, "finished_at"),
        trace_id=_g(row, "trace_id"),
    )


def sampling_list_from_row(row: Any) -> SamplingList:
    return SamplingList(
        list_name=SamplingListName(_g(row, "list_name")),
        config=_gj(row, "config") or {},
        description=_g(row, "description"),
        created_by=_g(row, "created_by"),
        created_at=_g(row, "created_at"),
    )


def sampling_progress_from_row(row: Any) -> SamplingProgress:
    return SamplingProgress(
        list_name=SamplingListName(_g(row, "list_name")),
        env_name=EnvName(_g(row, "env_name")),
        task_id=TaskId(_g(row, "task_id")),
        teacher_name=TeacherName(_g(row, "teacher_name")),
        target_samples=_g(row, "target_samples"),
        attempts=_g(row, "attempts") if _g(row, "attempts") is not None else 0,
        collected=_g(row, "collected"),
    )


def student_from_row(row: Any) -> StudentSubmission:
    return StudentSubmission(
        student_name=StudentName(_g(row, "student_name")),
        revision=Revision(_g(row, "revision")),
        hf_repo=_g(row, "hf_repo"),
        arch=_g(row, "arch"),
        param_b=_g(row, "param_b"),
        model_hash=_g(row, "model_hash"),
        template_check=_g(row, "template_check"),
        is_valid=_g(row, "is_valid"),
        invalid_reason=_g(row, "invalid_reason"),
        submitted_by=_g(row, "submitted_by"),
        submitted_at=_g(row, "submitted_at"),
        meta=_gj(row, "meta") or {},
    )


def anti_copy_from_row(row: Any) -> AntiCopyResult:
    return AntiCopyResult(
        student_name=StudentName(_g(row, "student_name")),
        revision=Revision(_g(row, "revision")),
        round_ts=_g(row, "round_ts"),
        is_copy=_g(row, "is_copy"),
        copied_from=_gj(row, "copied_from"),
        detection_meta=_gj(row, "detection_meta"),
    )


def student_score_from_row(row: Any) -> StudentScore:
    return StudentScore(
        run_id=RunId(_g(row, "run_id")),
        student_name=StudentName(_g(row, "student_name")),
        revision=Revision(_g(row, "revision")),
        pair_set_name=PairSetName(_g(row, "pair_set_name")),
        mean_score=_g(row, "mean_score"),
        win_rate=_g(row, "win_rate"),
        scores_by_env=_gj(row, "scores_by_env") or {},
        scores_by_teacher=_gj(row, "scores_by_teacher") or {},
        scores_by_difficulty=_gj(row, "scores_by_difficulty"),
        pairs_evaluated=_g(row, "pairs_evaluated"),
        config=_gj(row, "config") or {},
        forward_engine=_g(row, "forward_engine"),
        total_forward_tokens=_g(row, "total_forward_tokens"),
        total_compute_seconds=_g(row, "total_compute_seconds"),
        status=RunStatus(_g(row, "status")),
        latest_marker=_g(row, "latest_marker"),
        triggered_by=_g(row, "triggered_by"),
        started_at=_g(row, "started_at"),
        finished_at=_g(row, "finished_at"),
        trace_id=_g(row, "trace_id"),
    )


def deployment_from_row(row: Any) -> StudentDeployment:
    return StudentDeployment(
        deployment_id=_g(row, "deployment_id"),
        student_name=StudentName(_g(row, "student_name")),
        revision=Revision(_g(row, "revision")),
        engine=_g(row, "engine"),
        host=_g(row, "host"),
        gpu_indices=list(_g(row, "gpu_indices") or []),
        base_url=_g(row, "base_url"),
        instance_count=_g(row, "instance_count"),
        status=DeploymentStatus(_g(row, "status")),
        consecutive_failures=_g(row, "consecutive_failures"),
        next_retry_at=_g(row, "next_retry_at"),
        last_health_check_at=_g(row, "last_health_check_at"),
        meta=_gj(row, "meta"),
        created_at=_g(row, "created_at"),
        updated_at=_g(row, "updated_at"),
    )
