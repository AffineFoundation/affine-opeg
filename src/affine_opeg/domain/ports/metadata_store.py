"""Metadata store port.

Repository-pattern façade over Postgres. Application code never sees
SQLAlchemy. Repositories return domain models, accept domain models. Reads
are pageable via cursor-style ``after_id`` parameters.

The MetadataStore aggregate exposes a ``unit_of_work()`` context manager that
returns a UnitOfWork — a bundle of repositories sharing one transaction. This
is the only place application code commits / rolls back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

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
    Environment,
    Pair,
    PairCandidate,
    PairSet,
    Rollout,
    SamplingList,
    SamplingProgress,
    StudentDeployment,
    StudentScore,
    StudentSubmission,
    Task,
    Teacher,
)


@runtime_checkable
class TaskRepository(Protocol):
    async def upsert_environment(self, env: Environment) -> None: ...
    async def get_environment(self, env_name: EnvName) -> Environment | None: ...
    async def list_environments(self) -> list[Environment]: ...

    async def upsert_task(self, task: Task) -> None: ...
    async def upsert_tasks_bulk(self, tasks: Sequence[Task]) -> int: ...
    async def get_task(self, env_name: EnvName, task_id: TaskId) -> Task | None: ...
    async def iter_tasks(
        self, env_name: EnvName, *, difficulty: str | None = None
    ) -> AsyncIterator[Task]: ...


@runtime_checkable
class TeacherRepository(Protocol):
    async def upsert(self, teacher: Teacher) -> None: ...
    async def get(self, teacher_name: TeacherName) -> Teacher | None: ...
    async def list_active(self) -> list[Teacher]: ...


@runtime_checkable
class RolloutRepository(Protocol):
    async def insert(self, rollout: Rollout) -> None:
        """Conditional insert: fails on (env, task, teacher, sample_idx) duplicate."""

    async def get(self, rollout_id: RolloutId) -> Rollout | None: ...

    async def list_by_business_key(
        self,
        env_name: EnvName,
        task_id: TaskId,
        teacher_name: TeacherName,
    ) -> list[Rollout]: ...

    async def iter_groups_for_mining(
        self,
        *,
        env_names: Sequence[EnvName] | None = None,
        teacher_names: Sequence[TeacherName] | None = None,
        since: datetime | None = None,
    ) -> AsyncIterator[list[Rollout]]:
        """Yield one list per ``(env, task, teacher)`` group. Status='ok' only."""

    async def update_group_label(self, rollout_id: RolloutId, label: str) -> None: ...

    async def coverage_matrix(
        self, env_name: EnvName
    ) -> dict[tuple[TeacherName, TaskId], int]: ...


@runtime_checkable
class PairSetRepository(Protocol):
    async def create(self, pair_set: PairSet) -> None: ...
    async def get(self, name: PairSetName) -> PairSet | None: ...
    async def list(self, *, status: str | None = None) -> list[PairSet]: ...
    async def update_status(self, name: PairSetName, status: str, **fields: Any) -> None: ...


@runtime_checkable
class PairRepository(Protocol):
    async def bulk_insert(self, pair_set: PairSetName, candidates: Sequence[PairCandidate]) -> int: ...
    async def get(self, pair_id: PairId) -> Pair | None: ...
    async def list_in_set(
        self,
        pair_set: PairSetName,
        *,
        env_name: EnvName | None = None,
        teacher_name: TeacherName | None = None,
        min_reward_gap: float | None = None,
        limit: int = 100,
        after_id: PairId | None = None,
    ) -> list[Pair]: ...
    async def count_in_set(self, pair_set: PairSetName) -> int: ...


@runtime_checkable
class SamplingListRepository(Protocol):
    async def create(self, sampling_list: SamplingList) -> None: ...
    async def get(self, name: SamplingListName) -> SamplingList | None: ...
    async def list(self) -> list[SamplingList]: ...

    async def init_progress(self, items: Sequence[SamplingProgress]) -> None: ...
    async def list_progress(self, list_name: SamplingListName) -> list[SamplingProgress]: ...
    async def list_open_progress(self, list_name: SamplingListName) -> list[SamplingProgress]: ...
    async def increment_collected(
        self, list_name: SamplingListName, env_name: EnvName,
        task_id: TaskId, teacher_name: TeacherName, delta: int = 1,
    ) -> None: ...

    async def freeze_degenerate_cell(
        self, list_name: SamplingListName, env_name: EnvName,
        task_id: TaskId, teacher_name: TeacherName,
        min_samples: int = 4,
    ) -> bool:
        """Mark cell published_at if its latest ``min_samples`` ok
        rollouts all have identical reward (stddev = 0). Returns True
        when newly frozen. No-op otherwise (idempotent)."""
        ...

    async def claim_next_cell(
        self,
        list_name: SamplingListName,
        *,
        env_names: Sequence[EnvName] | None = None,
        teacher_names: Sequence[TeacherName] | None = None,
        batch_size: int = 16,
    ) -> list[SamplingProgress]:
        """Reserve up to ``batch_size`` open cells under FOR UPDATE SKIP LOCKED."""


@runtime_checkable
class StudentRepository(Protocol):
    async def upsert(self, submission: StudentSubmission) -> None: ...
    async def get(self, name: StudentName, revision: Revision) -> StudentSubmission | None: ...
    async def list(self, *, only_valid: bool = False) -> list[StudentSubmission]: ...
    async def list_revisions(self, name: StudentName) -> list[StudentSubmission]: ...
    async def find_by_model_hash(self, model_hash: str) -> list[StudentSubmission]: ...

    async def insert_anti_copy_result(self, result: AntiCopyResult) -> None: ...
    async def latest_anti_copy(
        self, name: StudentName, revision: Revision
    ) -> AntiCopyResult | None: ...


@runtime_checkable
class StudentScoreRepository(Protocol):
    async def upsert_snapshot(self, score: StudentScore) -> None: ...
    async def mark_latest(self, score: StudentScore) -> None:
        """Atomically clear any prior LATEST for (student, revision, pair_set) and mark this row."""

    async def get_latest(
        self,
        student_name: StudentName,
        revision: Revision,
        pair_set: PairSetName,
    ) -> StudentScore | None: ...

    async def get_run(self, run_id: RunId) -> list[StudentScore]: ...

    async def list_leaderboard(
        self,
        pair_set: PairSetName,
        *,
        limit: int = 50,
    ) -> list[StudentScore]: ...

    async def insert_pair_score(
        self,
        run_id: RunId,
        student_name: StudentName,
        revision: Revision,
        pair_id: PairId,
        *,
        ce_win: float, ce_lose: float,
        tokens_win: int, tokens_lose: int,
        score: float,
        ce_per_message_win: list[float] | None = None,
        ce_per_message_lose: list[float] | None = None,
    ) -> None: ...

    async def claim_next_run(self) -> StudentScore | None:
        """Atomically claim the oldest queued run, transition queued→running."""

    async def mark_failed(
        self, run_id: RunId, student_name: StudentName, revision: Revision,
        *, error: str,
    ) -> None:
        """Mark a run as failed and persist the error message in ``config.error``."""


@runtime_checkable
class DeploymentRepository(Protocol):
    async def upsert(self, deployment: StudentDeployment) -> None: ...
    async def get(self, deployment_id: str) -> StudentDeployment | None: ...
    async def list_by_status(self, status: str) -> list[StudentDeployment]: ...


class UnitOfWork(Protocol):
    """Group of repositories sharing one transaction."""

    tasks: TaskRepository
    teachers: TeacherRepository
    rollouts: RolloutRepository
    pair_sets: PairSetRepository
    pairs: PairRepository
    sampling_lists: SamplingListRepository
    students: StudentRepository
    student_scores: StudentScoreRepository
    deployments: DeploymentRepository

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


@runtime_checkable
class MetadataStore(Protocol):
    """Aggregate root for all repositories."""

    def unit_of_work(self) -> AbstractAsyncContextManager[UnitOfWork]:
        """One scope == one transaction. Auto-commits on success, rolls back on raise."""
