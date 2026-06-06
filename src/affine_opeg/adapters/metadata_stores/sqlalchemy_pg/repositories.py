"""SQLAlchemy implementations of the repository ports.

These translate domain objects to SQL via Core (not ORM declarative) for speed
and explicitness. Each repository holds a reference to one ``AsyncSession``;
construction happens in the SqlAlchemyUnitOfWork.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import and_, delete, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg import mappers
from affine_opeg.adapters.metadata_stores.sqlalchemy_pg.orm import (
    anti_copy_results,
    environments,
    pair_sets,
    pairs,
    rollouts,
    sampling_lists,
    sampling_progress,
    student_deployments,
    student_pair_scores,
    student_scores,
    student_submissions,
    tasks,
    teachers,
)
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


class SaTaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert_environment(self, env: Environment) -> None:
        # asyncpg accepts dicts for INT4RANGE via raw SQL casting
        await self._s.execute(
            text(
                """
                INSERT INTO environments (env_name, dataset, dataset_version, task_id_range, meta)
                VALUES (:env_name, :dataset, :dataset_version, int4range(:lo, :hi), :meta)
                ON CONFLICT (env_name) DO UPDATE SET
                    dataset = EXCLUDED.dataset,
                    dataset_version = EXCLUDED.dataset_version,
                    task_id_range = EXCLUDED.task_id_range,
                    meta = EXCLUDED.meta
                """
            ),
            {
                "env_name": env.env_name,
                "dataset": env.dataset,
                "dataset_version": env.dataset_version,
                "lo": env.task_id_min,
                "hi": env.task_id_max,
                "meta": env.meta,
            },
        )

    async def get_environment(self, env_name: EnvName) -> Environment | None:
        row = (await self._s.execute(
            select(environments).where(environments.c.env_name == env_name)
        )).one_or_none()
        return mappers.environment_from_row(row) if row else None

    async def list_environments(self) -> list[Environment]:
        rows = (await self._s.execute(select(environments).order_by(environments.c.env_name))).all()
        return [mappers.environment_from_row(r) for r in rows]

    async def upsert_task(self, task: Task) -> None:
        stmt = pg_insert(tasks).values(
            env_name=task.env_name,
            task_id=task.task_id,
            repo=task.repo,
            base_commit=task.base_commit,
            problem=task.problem,
            hidden_tests=task.hidden_tests,
            difficulty=task.difficulty,
            meta=task.meta,
        )
        await self._s.execute(stmt.on_conflict_do_update(
            index_elements=["env_name", "task_id"],
            set_={
                "repo": stmt.excluded.repo,
                "base_commit": stmt.excluded.base_commit,
                "problem": stmt.excluded.problem,
                "hidden_tests": stmt.excluded.hidden_tests,
                "difficulty": stmt.excluded.difficulty,
                "meta": stmt.excluded.meta,
            },
        ))

    async def upsert_tasks_bulk(self, tasks_: Sequence[Task]) -> int:
        if not tasks_:
            return 0
        rows = [
            {
                "env_name": t.env_name, "task_id": t.task_id, "repo": t.repo,
                "base_commit": t.base_commit, "problem": t.problem,
                "hidden_tests": t.hidden_tests, "difficulty": t.difficulty, "meta": t.meta,
            }
            for t in tasks_
        ]
        stmt = pg_insert(tasks).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["env_name", "task_id"])
        result = await self._s.execute(stmt)
        return result.rowcount or 0

    async def get_task(self, env_name: EnvName, task_id: TaskId) -> Task | None:
        row = (await self._s.execute(
            select(tasks).where(and_(tasks.c.env_name == env_name, tasks.c.task_id == task_id))
        )).one_or_none()
        return mappers.task_from_row(row) if row else None

    async def iter_tasks(
        self, env_name: EnvName, *, difficulty: str | None = None
    ) -> AsyncIterator[Task]:
        q = select(tasks).where(tasks.c.env_name == env_name).order_by(tasks.c.task_id)
        if difficulty:
            q = q.where(tasks.c.difficulty == difficulty)
        result = await self._s.stream(q)
        async for row in result:
            yield mappers.task_from_row(row)


class SaTeacherRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert(self, teacher: Teacher) -> None:
        stmt = pg_insert(teachers).values(
            teacher_name=teacher.teacher_name,
            model_family=teacher.model_family,
            provider=teacher.provider,
            endpoint=teacher.endpoint,
            api_key_env=teacher.api_key_env,
            tool_format=teacher.tool_format,
            reasoning_format=teacher.reasoning_format,
            context_window=teacher.context_window,
            price_per_mtoken_in=teacher.price_per_mtoken_in,
            price_per_mtoken_out=teacher.price_per_mtoken_out,
            active=teacher.active,
            meta=teacher.meta,
        )
        await self._s.execute(stmt.on_conflict_do_update(
            index_elements=["teacher_name"],
            set_={
                "model_family": stmt.excluded.model_family,
                "provider": stmt.excluded.provider,
                "endpoint": stmt.excluded.endpoint,
                "api_key_env": stmt.excluded.api_key_env,
                "tool_format": stmt.excluded.tool_format,
                "reasoning_format": stmt.excluded.reasoning_format,
                "context_window": stmt.excluded.context_window,
                "price_per_mtoken_in": stmt.excluded.price_per_mtoken_in,
                "price_per_mtoken_out": stmt.excluded.price_per_mtoken_out,
                "active": stmt.excluded.active,
                "meta": stmt.excluded.meta,
            },
        ))

    async def get(self, teacher_name: TeacherName) -> Teacher | None:
        row = (await self._s.execute(
            select(teachers).where(teachers.c.teacher_name == teacher_name)
        )).one_or_none()
        return mappers.teacher_from_row(row) if row else None

    async def list_active(self) -> list[Teacher]:
        rows = (await self._s.execute(
            select(teachers).where(teachers.c.active.is_(True)).order_by(teachers.c.teacher_name)
        )).all()
        return [mappers.teacher_from_row(r) for r in rows]


class SaRolloutRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def insert(self, rollout: Rollout) -> None:
        # Upsert on the business key. A retry that previously crashed mid-way
        # (status='env_error' etc.) gets overwritten by the new attempt —
        # otherwise the unique constraint blocks the producer from making
        # forward progress on the same cell.
        stmt = pg_insert(rollouts).values(
            rollout_id=rollout.rollout_id,
            env_name=rollout.env_name,
            task_id=rollout.task_id,
            teacher_name=rollout.teacher_name,
            sample_idx=rollout.sample_idx,
            temperature=rollout.temperature,
            top_p=rollout.top_p,
            seed=rollout.seed,
            status=rollout.status.value,
            reward=rollout.reward,
            reward_breakdown=rollout.reward_breakdown,
            steps=rollout.steps,
            latency_ms=rollout.latency_ms,
            tokens_in=rollout.tokens_in,
            tokens_out=rollout.tokens_out,
            cost_usd=rollout.cost_usd,
            schema_version=rollout.schema_version,
            extra_compressed=rollout.extra_compressed,
            extra_sha256=rollout.extra_sha256,
            blob_uri=rollout.blob_uri,
            group_label=rollout.group_label.value if rollout.group_label else None,
            producer_id=rollout.producer_id,
            trace_id=rollout.trace_id,
        )
        await self._s.execute(stmt.on_conflict_do_update(
            index_elements=["env_name", "task_id", "teacher_name", "sample_idx"],
            set_={
                "rollout_id": stmt.excluded.rollout_id,
                "temperature": stmt.excluded.temperature,
                "top_p": stmt.excluded.top_p,
                "seed": stmt.excluded.seed,
                "status": stmt.excluded.status,
                "reward": stmt.excluded.reward,
                "reward_breakdown": stmt.excluded.reward_breakdown,
                "steps": stmt.excluded.steps,
                "latency_ms": stmt.excluded.latency_ms,
                "tokens_in": stmt.excluded.tokens_in,
                "tokens_out": stmt.excluded.tokens_out,
                "cost_usd": stmt.excluded.cost_usd,
                "schema_version": stmt.excluded.schema_version,
                "extra_compressed": stmt.excluded.extra_compressed,
                "extra_sha256": stmt.excluded.extra_sha256,
                "blob_uri": stmt.excluded.blob_uri,
                "producer_id": stmt.excluded.producer_id,
                "trace_id": stmt.excluded.trace_id,
                "created_at": text("now()"),
            },
        ))

    async def get(self, rollout_id: RolloutId) -> Rollout | None:
        row = (await self._s.execute(
            select(rollouts).where(rollouts.c.rollout_id == rollout_id)
        )).one_or_none()
        return mappers.rollout_from_row(row) if row else None

    async def list_by_business_key(
        self, env_name: EnvName, task_id: TaskId, teacher_name: TeacherName,
    ) -> list[Rollout]:
        rows = (await self._s.execute(
            select(rollouts).where(and_(
                rollouts.c.env_name == env_name,
                rollouts.c.task_id == task_id,
                rollouts.c.teacher_name == teacher_name,
            )).order_by(rollouts.c.sample_idx)
        )).all()
        return [mappers.rollout_from_row(r) for r in rows]

    async def iter_groups_for_mining(
        self, *,
        env_names: Sequence[EnvName] | None = None,
        teacher_names: Sequence[TeacherName] | None = None,
        since: datetime | None = None,
    ) -> AsyncIterator[list[Rollout]]:
        conds = [rollouts.c.status == "ok"]
        if env_names:
            conds.append(rollouts.c.env_name.in_(env_names))
        if teacher_names:
            conds.append(rollouts.c.teacher_name.in_(teacher_names))
        if since:
            conds.append(rollouts.c.created_at >= since)

        q = (
            select(rollouts)
            .where(and_(*conds))
            .order_by(rollouts.c.env_name, rollouts.c.task_id, rollouts.c.teacher_name, rollouts.c.sample_idx)
        )
        result = await self._s.stream(q)
        current_key: tuple[Any, Any, Any] | None = None
        buf: list[Rollout] = []
        async for row in result:
            key = (row.env_name, row.task_id, row.teacher_name)
            if current_key is None:
                current_key = key
            if key != current_key:
                if buf:
                    yield buf
                buf = []
                current_key = key
            buf.append(mappers.rollout_from_row(row))
        if buf:
            yield buf

    async def update_group_label(self, rollout_id: RolloutId, label: str) -> None:
        await self._s.execute(
            update(rollouts).where(rollouts.c.rollout_id == rollout_id).values(group_label=label)
        )

    async def coverage_matrix(self, env_name: EnvName) -> dict[tuple[TeacherName, TaskId], int]:
        rows = (await self._s.execute(text(
            """
            SELECT teacher_name, task_id, COUNT(*) AS n
            FROM rollouts
            WHERE env_name = :env AND status = 'ok'
            GROUP BY teacher_name, task_id
            """
        ), {"env": env_name})).all()
        return {(TeacherName(r.teacher_name), TaskId(r.task_id)): r.n for r in rows}


class SaPairSetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, pair_set: PairSet) -> None:
        await self._s.execute(pair_sets.insert().values(
            pair_set_name=pair_set.pair_set_name,
            source_filter=pair_set.source_filter,
            strategy=pair_set.strategy,
            rollouts_scanned=pair_set.rollouts_scanned,
            pairs_created=pair_set.pairs_created,
            groups_summary=pair_set.groups_summary,
            status=pair_set.status.value,
            triggered_by=pair_set.triggered_by,
            started_at=pair_set.started_at,
            finished_at=pair_set.finished_at,
            trace_id=pair_set.trace_id,
        ))

    async def get(self, name: PairSetName) -> PairSet | None:
        row = (await self._s.execute(
            select(pair_sets).where(pair_sets.c.pair_set_name == name)
        )).one_or_none()
        return mappers.pair_set_from_row(row) if row else None

    async def list(self, *, status: str | None = None) -> list[PairSet]:
        q = select(pair_sets).order_by(pair_sets.c.started_at.desc())
        if status:
            q = q.where(pair_sets.c.status == status)
        rows = (await self._s.execute(q)).all()
        return [mappers.pair_set_from_row(r) for r in rows]

    async def update_status(self, name: PairSetName, status: str, **fields: Any) -> None:
        await self._s.execute(
            update(pair_sets).where(pair_sets.c.pair_set_name == name).values(status=status, **fields)
        )


class SaPairRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def bulk_insert(self, pair_set: PairSetName, candidates: Sequence[PairCandidate]) -> int:
        if not candidates:
            return 0
        rows = [
            {
                "pair_set_name": pair_set,
                "env_name": c.env_name,
                "task_id": c.task_id,
                "teacher_name": c.teacher_name,
                "win_rollout": c.win_rollout,
                "lose_rollout": c.lose_rollout,
                "reward_win": c.reward_win,
                "reward_lose": c.reward_lose,
                "reward_gap": c.reward_gap,
                "selection_meta": c.selection_meta,
            }
            for c in candidates
        ]
        result = await self._s.execute(pairs.insert(), rows)
        return result.rowcount or 0

    async def get(self, pair_id: PairId) -> Pair | None:
        row = (await self._s.execute(select(pairs).where(pairs.c.pair_id == pair_id))).one_or_none()
        return mappers.pair_from_row(row) if row else None

    async def list_in_set(
        self,
        pair_set: PairSetName,
        *,
        env_name: EnvName | None = None,
        teacher_name: TeacherName | None = None,
        min_reward_gap: float | None = None,
        limit: int = 100,
        after_id: PairId | None = None,
    ) -> list[Pair]:
        conds = [pairs.c.pair_set_name == pair_set]
        if env_name:
            conds.append(pairs.c.env_name == env_name)
        if teacher_name:
            conds.append(pairs.c.teacher_name == teacher_name)
        if min_reward_gap is not None:
            conds.append(pairs.c.reward_gap >= min_reward_gap)
        if after_id:
            conds.append(pairs.c.pair_id > after_id)
        q = select(pairs).where(and_(*conds)).order_by(pairs.c.pair_id).limit(limit)
        rows = (await self._s.execute(q)).all()
        return [mappers.pair_from_row(r) for r in rows]

    async def count_in_set(self, pair_set: PairSetName) -> int:
        row = (await self._s.execute(text(
            "SELECT COUNT(*) AS n FROM pairs WHERE pair_set_name = :p"
        ), {"p": pair_set})).one()
        return int(row.n)


class SaSamplingListRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, sampling_list: SamplingList) -> None:
        await self._s.execute(sampling_lists.insert().values(
            list_name=sampling_list.list_name,
            config=sampling_list.config,
            description=sampling_list.description,
            created_by=sampling_list.created_by,
        ))

    async def get(self, name: SamplingListName) -> SamplingList | None:
        row = (await self._s.execute(
            select(sampling_lists).where(sampling_lists.c.list_name == name)
        )).one_or_none()
        return mappers.sampling_list_from_row(row) if row else None

    async def list(self) -> list[SamplingList]:
        rows = (await self._s.execute(
            select(sampling_lists).order_by(sampling_lists.c.created_at.desc())
        )).all()
        return [mappers.sampling_list_from_row(r) for r in rows]

    async def init_progress(self, items: Sequence[SamplingProgress]) -> None:
        if not items:
            return
        rows = [
            {
                "list_name": p.list_name, "env_name": p.env_name, "task_id": p.task_id,
                "teacher_name": p.teacher_name, "target_samples": p.target_samples,
                # ``attempts`` is NOT NULL with no server_default once migration
                # 0002's backfill default is dropped — set it explicitly (the
                # model defaults it to 0) so seeding doesn't trip the constraint.
                "attempts": p.attempts, "collected": p.collected,
            } for p in items
        ]
        stmt = pg_insert(sampling_progress).values(rows)
        await self._s.execute(stmt.on_conflict_do_nothing())

    async def list_progress(self, list_name: SamplingListName) -> list[SamplingProgress]:
        rows = (await self._s.execute(
            select(sampling_progress).where(sampling_progress.c.list_name == list_name)
        )).all()
        return [mappers.sampling_progress_from_row(r) for r in rows]

    async def list_open_progress(self, list_name: SamplingListName) -> list[SamplingProgress]:
        rows = (await self._s.execute(
            select(sampling_progress).where(and_(
                sampling_progress.c.list_name == list_name,
                sampling_progress.c.collected < sampling_progress.c.target_samples,
            ))
        )).all()
        return [mappers.sampling_progress_from_row(r) for r in rows]

    async def increment_collected(
        self, list_name: SamplingListName, env_name: EnvName,
        task_id: TaskId, teacher_name: TeacherName, delta: int = 1,
    ) -> None:
        await self._s.execute(
            update(sampling_progress)
            .where(and_(
                sampling_progress.c.list_name == list_name,
                sampling_progress.c.env_name == env_name,
                sampling_progress.c.task_id == task_id,
                sampling_progress.c.teacher_name == teacher_name,
            ))
            .values(
                collected=sampling_progress.c.collected + delta,
                last_updated=text("now()"),
            )
        )

    async def freeze_degenerate_cell(
        self, list_name: SamplingListName, env_name: EnvName,
        task_id: TaskId, teacher_name: TeacherName,
        min_samples: int = 4,
    ) -> bool:
        """If the latest ``min_samples`` ok rollouts of this cell all have
        the same reward, mark the cell ``published_at = now()`` so the
        scheduler stops claiming new attempts for it.

        This frees up the remaining attempt budget for cells that might
        actually produce variance. Run after each ok-rollout commit; the
        upstream publisher's variance filter (min_reward_std=0.05) would
        skip these cells anyway — so we save the wasted samples up front.

        Returns True if the cell was newly frozen.
        """
        # stddev over the latest N ok rollouts of this cell; zero stddev
        # means N consecutive identical rewards => degenerate. We restrict
        # to the latest N (not all-history) so a recovered cell can
        # re-emerge if upstream behavior shifts (e.g. after a teacher
        # config change). published_at IS NULL guard makes this idempotent.
        result = await self._s.execute(
            text(
                """
                WITH latest AS (
                    SELECT reward
                    FROM rollouts
                    WHERE env_name = :env_name
                      AND task_id = :task_id
                      AND teacher_name = :teacher_name
                      AND status = 'ok'
                    ORDER BY created_at DESC
                    LIMIT :n
                ),
                agg AS (
                    SELECT count(*) AS n, stddev_samp(reward) AS s
                    FROM latest
                )
                UPDATE sampling_progress sp
                SET published_at = now(),
                    last_updated = now()
                FROM agg
                WHERE sp.list_name = :list_name
                  AND sp.env_name = :env_name
                  AND sp.task_id = :task_id
                  AND sp.teacher_name = :teacher_name
                  AND sp.published_at IS NULL
                  AND agg.n >= :n
                  AND COALESCE(agg.s, 0) = 0
                RETURNING sp.task_id
                """
            ),
            {
                "list_name": str(list_name),
                "env_name": str(env_name),
                "task_id": int(task_id),
                "teacher_name": str(teacher_name),
                "n": int(min_samples),
            },
        )
        return result.first() is not None

    async def claim_next_cell(
        self,
        list_name: SamplingListName,
        *,
        env_names: Sequence[EnvName] | None = None,
        teacher_names: Sequence[TeacherName] | None = None,
        batch_size: int = 16,
        max_attempts_per_cell: int | None = None,
    ) -> list[SamplingProgress]:
        """Atomically reserve up to ``batch_size`` open cells.

        Eligibility: the cell still needs success rows (``collected <
        target_samples``) AND the per-cell attempt budget hasn't been
        spent (``attempts < max_attempts_per_cell``). When
        ``max_attempts_per_cell`` is None the budget defaults to
        ``2 * target_samples`` (50% failure tolerance).

        Inside the transaction we bump ``attempts`` by 1 — this hands
        the caller a unique ``sample_idx`` (= attempts - 1) even when
        many producers race on the same cell. ``collected`` is **not**
        touched here; it's bumped by ``increment_collected`` only when
        a status='ok' rollout has been persisted.

        Returned ``SamplingProgress.attempts`` is the *post-increment*
        value; ``sample_idx = attempts - 1`` is the caller's slot.
        """
        env_filter = ""
        teacher_filter = ""
        params: dict[str, Any] = {"list_name": list_name, "batch": batch_size}
        if env_names:
            env_filter = "AND env_name = ANY(:envs)"
            params["envs"] = list(env_names)
        if teacher_names:
            teacher_filter = "AND teacher_name = ANY(:teachers)"
            params["teachers"] = list(teacher_names)
        # Default attempt budget = 2 * target_samples; expressed inline
        # so cells with different targets keep proportional headroom.
        if max_attempts_per_cell is not None:
            attempt_cap = "AND attempts < :max_attempts"
            params["max_attempts"] = max_attempts_per_cell
        else:
            attempt_cap = "AND attempts < (2 * target_samples)"
        sql = f"""
            WITH cte AS (
                SELECT list_name, env_name, task_id, teacher_name
                FROM sampling_progress
                WHERE list_name = :list_name
                  AND collected < target_samples
                  {attempt_cap}
                  {env_filter}
                  {teacher_filter}
                ORDER BY collected ASC, attempts ASC, task_id ASC
                LIMIT :batch
                FOR UPDATE SKIP LOCKED
            )
            UPDATE sampling_progress sp
            SET attempts = sp.attempts + 1,
                last_updated = now()
            FROM cte
            WHERE sp.list_name = cte.list_name
              AND sp.env_name = cte.env_name
              AND sp.task_id = cte.task_id
              AND sp.teacher_name = cte.teacher_name
            RETURNING sp.list_name, sp.env_name, sp.task_id, sp.teacher_name,
                      sp.target_samples, sp.attempts, sp.collected
        """
        rows = (await self._s.execute(text(sql), params)).all()
        return [mappers.sampling_progress_from_row(r) for r in rows]


class SaStudentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert(self, submission: StudentSubmission) -> None:
        stmt = pg_insert(student_submissions).values(
            student_name=submission.student_name,
            revision=submission.revision,
            hf_repo=submission.hf_repo,
            arch=submission.arch,
            param_b=submission.param_b,
            model_hash=submission.model_hash,
            template_check=submission.template_check,
            is_valid=submission.is_valid,
            invalid_reason=submission.invalid_reason,
            submitted_by=submission.submitted_by,
            meta=submission.meta,
        )
        await self._s.execute(stmt.on_conflict_do_update(
            index_elements=["student_name", "revision"],
            set_={
                "hf_repo": stmt.excluded.hf_repo,
                "arch": stmt.excluded.arch,
                "param_b": stmt.excluded.param_b,
                "model_hash": stmt.excluded.model_hash,
                "template_check": stmt.excluded.template_check,
                "is_valid": stmt.excluded.is_valid,
                "invalid_reason": stmt.excluded.invalid_reason,
                "meta": stmt.excluded.meta,
            },
        ))

    async def get(self, name: StudentName, revision: Revision) -> StudentSubmission | None:
        row = (await self._s.execute(
            select(student_submissions).where(and_(
                student_submissions.c.student_name == name,
                student_submissions.c.revision == revision,
            ))
        )).one_or_none()
        return mappers.student_from_row(row) if row else None

    async def list(self, *, only_valid: bool = False) -> list[StudentSubmission]:
        q = select(student_submissions).order_by(student_submissions.c.submitted_at.desc())
        if only_valid:
            q = q.where(student_submissions.c.is_valid.is_(True))
        rows = (await self._s.execute(q)).all()
        return [mappers.student_from_row(r) for r in rows]

    async def list_revisions(self, name: StudentName) -> list[StudentSubmission]:
        rows = (await self._s.execute(
            select(student_submissions)
            .where(student_submissions.c.student_name == name)
            .order_by(student_submissions.c.submitted_at.desc())
        )).all()
        return [mappers.student_from_row(r) for r in rows]

    async def find_by_model_hash(self, model_hash: str) -> list[StudentSubmission]:
        rows = (await self._s.execute(
            select(student_submissions).where(student_submissions.c.model_hash == model_hash)
        )).all()
        return [mappers.student_from_row(r) for r in rows]

    async def insert_anti_copy_result(self, result: AntiCopyResult) -> None:
        await self._s.execute(anti_copy_results.insert().values(
            student_name=result.student_name,
            revision=result.revision,
            round_ts=result.round_ts,
            is_copy=result.is_copy,
            copied_from=result.copied_from,
            detection_meta=result.detection_meta,
        ))

    async def latest_anti_copy(
        self, name: StudentName, revision: Revision
    ) -> AntiCopyResult | None:
        row = (await self._s.execute(
            select(anti_copy_results)
            .where(and_(
                anti_copy_results.c.student_name == name,
                anti_copy_results.c.revision == revision,
            ))
            .order_by(anti_copy_results.c.round_ts.desc())
            .limit(1)
        )).one_or_none()
        return mappers.anti_copy_from_row(row) if row else None


class SaStudentScoreRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert_snapshot(self, score: StudentScore) -> None:
        stmt = pg_insert(student_scores).values(
            run_id=score.run_id,
            student_name=score.student_name,
            revision=score.revision,
            pair_set_name=score.pair_set_name,
            mean_score=score.mean_score,
            win_rate=score.win_rate,
            scores_by_env=score.scores_by_env,
            scores_by_teacher=score.scores_by_teacher,
            scores_by_difficulty=score.scores_by_difficulty,
            pairs_evaluated=score.pairs_evaluated,
            config=score.config,
            forward_engine=score.forward_engine,
            total_forward_tokens=score.total_forward_tokens,
            total_compute_seconds=score.total_compute_seconds,
            status=score.status.value,
            latest_marker=score.latest_marker,
            triggered_by=score.triggered_by,
            started_at=score.started_at,
            finished_at=score.finished_at,
            trace_id=score.trace_id,
        )
        await self._s.execute(stmt.on_conflict_do_update(
            index_elements=["run_id", "student_name", "revision"],
            set_={
                "mean_score": stmt.excluded.mean_score,
                "win_rate": stmt.excluded.win_rate,
                "scores_by_env": stmt.excluded.scores_by_env,
                "scores_by_teacher": stmt.excluded.scores_by_teacher,
                "scores_by_difficulty": stmt.excluded.scores_by_difficulty,
                "pairs_evaluated": stmt.excluded.pairs_evaluated,
                "config": stmt.excluded.config,
                "forward_engine": stmt.excluded.forward_engine,
                "total_forward_tokens": stmt.excluded.total_forward_tokens,
                "total_compute_seconds": stmt.excluded.total_compute_seconds,
                "status": stmt.excluded.status,
                "latest_marker": stmt.excluded.latest_marker,
                "finished_at": stmt.excluded.finished_at,
            },
        ))

    async def mark_latest(self, score: StudentScore) -> None:
        """Clear prior LATEST for (student, revision, pair_set) then mark this row.

        Atomic under a single transaction. The partial-unique index
        ``uq_student_latest`` prevents two LATEST rows from coexisting.
        """
        await self._s.execute(
            update(student_scores)
            .where(and_(
                student_scores.c.student_name == score.student_name,
                student_scores.c.revision == score.revision,
                student_scores.c.pair_set_name == score.pair_set_name,
                student_scores.c.latest_marker == "LATEST",
            ))
            .values(latest_marker=None)
        )
        await self._s.execute(
            update(student_scores)
            .where(and_(
                student_scores.c.run_id == score.run_id,
                student_scores.c.student_name == score.student_name,
                student_scores.c.revision == score.revision,
            ))
            .values(latest_marker="LATEST")
        )

    async def get_latest(
        self,
        student_name: StudentName,
        revision: Revision,
        pair_set: PairSetName,
    ) -> StudentScore | None:
        row = (await self._s.execute(
            select(student_scores).where(and_(
                student_scores.c.student_name == student_name,
                student_scores.c.revision == revision,
                student_scores.c.pair_set_name == pair_set,
                student_scores.c.latest_marker == "LATEST",
            )).limit(1)
        )).one_or_none()
        return mappers.student_score_from_row(row) if row else None

    async def get_run(self, run_id: RunId) -> list[StudentScore]:
        rows = (await self._s.execute(
            select(student_scores).where(student_scores.c.run_id == run_id)
        )).all()
        return [mappers.student_score_from_row(r) for r in rows]

    async def list_leaderboard(
        self,
        pair_set: PairSetName,
        *,
        limit: int = 50,
    ) -> list[StudentScore]:
        rows = (await self._s.execute(
            select(student_scores)
            .where(and_(
                student_scores.c.pair_set_name == pair_set,
                student_scores.c.latest_marker == "LATEST",
            ))
            .order_by(student_scores.c.mean_score.desc())
            .limit(limit)
        )).all()
        return [mappers.student_score_from_row(r) for r in rows]

    async def insert_pair_score(
        self, run_id: RunId, student_name: StudentName, revision: Revision, pair_id: PairId,
        *, ce_win: float, ce_lose: float, tokens_win: int, tokens_lose: int, score: float,
        ce_per_message_win: list[float] | None = None,
        ce_per_message_lose: list[float] | None = None,
    ) -> None:
        await self._s.execute(student_pair_scores.insert().values(
            run_id=run_id, student_name=student_name, revision=revision, pair_id=pair_id,
            ce_win=ce_win, ce_lose=ce_lose, tokens_win=tokens_win, tokens_lose=tokens_lose,
            score=score,
            ce_per_message_win=ce_per_message_win,
            ce_per_message_lose=ce_per_message_lose,
        ))

    async def claim_next_run(self) -> StudentScore | None:
        """Atomically pick the oldest queued eval run.

        Mark it ``running`` in the same transaction so concurrent consumers
        don't double-pick. Returns the claimed snapshot or None if the queue
        is empty.
        """
        row = (await self._s.execute(text(
            """
            WITH cte AS (
                SELECT run_id, student_name, revision
                FROM student_scores
                WHERE status = 'queued'
                ORDER BY started_at NULLS FIRST, run_id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE student_scores ss
            SET status = 'running', started_at = now()
            FROM cte
            WHERE ss.run_id = cte.run_id
              AND ss.student_name = cte.student_name
              AND ss.revision = cte.revision
            RETURNING ss.*
            """
        ))).one_or_none()
        return mappers.student_score_from_row(row) if row else None

    async def mark_failed(
        self, run_id: RunId, student_name: StudentName, revision: Revision,
        *, error: str,
    ) -> None:
        # ``:err::text`` doesn't parse — SQLAlchemy's text() bind-param scanner
        # treats the second colon as the start of another bind. Wrap the bind
        # in parens so PostgreSQL sees `(:err)::text` and SQLAlchemy still
        # sees ``:err`` as a single named param.
        await self._s.execute(text(
            """
            UPDATE student_scores
            SET status = 'failed',
                finished_at = now(),
                config = jsonb_set(coalesce(config, '{}'::jsonb),
                                   '{error}',
                                   to_jsonb((:err)::text))
            WHERE run_id = :rid AND student_name = :name AND revision = :rev
            """
        ), {"rid": run_id, "name": student_name, "rev": revision, "err": error[:2000]})


class SaDeploymentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert(self, deployment: StudentDeployment) -> None:
        stmt = pg_insert(student_deployments).values(
            deployment_id=deployment.deployment_id,
            student_name=deployment.student_name,
            revision=deployment.revision,
            engine=deployment.engine,
            host=deployment.host,
            gpu_indices=deployment.gpu_indices,
            base_url=deployment.base_url,
            instance_count=deployment.instance_count,
            status=deployment.status.value,
            consecutive_failures=deployment.consecutive_failures,
            next_retry_at=deployment.next_retry_at,
            last_health_check_at=deployment.last_health_check_at,
            meta=deployment.meta,
        )
        await self._s.execute(stmt.on_conflict_do_update(
            index_elements=["deployment_id"],
            set_={
                "engine": stmt.excluded.engine,
                "host": stmt.excluded.host,
                "gpu_indices": stmt.excluded.gpu_indices,
                "base_url": stmt.excluded.base_url,
                "instance_count": stmt.excluded.instance_count,
                "status": stmt.excluded.status,
                "consecutive_failures": stmt.excluded.consecutive_failures,
                "next_retry_at": stmt.excluded.next_retry_at,
                "last_health_check_at": stmt.excluded.last_health_check_at,
                "meta": stmt.excluded.meta,
                "updated_at": text("now()"),
            },
        ))

    async def get(self, deployment_id: str) -> StudentDeployment | None:
        row = (await self._s.execute(
            select(student_deployments).where(student_deployments.c.deployment_id == deployment_id)
        )).one_or_none()
        return mappers.deployment_from_row(row) if row else None

    async def list_by_status(self, status: str) -> list[StudentDeployment]:
        rows = (await self._s.execute(
            select(student_deployments).where(student_deployments.c.status == status)
        )).all()
        return [mappers.deployment_from_row(r) for r in rows]


__all__ = [
    "SaDeploymentRepository",
    "SaPairRepository",
    "SaPairSetRepository",
    "SaRolloutRepository",
    "SaSamplingListRepository",
    "SaStudentRepository",
    "SaStudentScoreRepository",
    "SaTaskRepository",
    "SaTeacherRepository",
]
