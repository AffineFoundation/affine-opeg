"""Bootstrap use cases.

Loading the static reference data (teachers, environments, tasks) and
materialising a ``sampling_list`` into ``sampling_progress`` rows. These run
once per dataset / refresh, not on the hot path.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from affine_opeg.domain.errors import TaskSourceError
from affine_opeg.domain.ids import EnvName, SamplingListName, TaskId, TeacherName
from affine_opeg.domain.models import (
    Environment,
    SamplingList,
    SamplingProgress,
    Teacher,
)
from affine_opeg.domain.ports import MetadataStore
from affine_opeg.domain.ports.task_source import TaskSource
from affine_opeg.infrastructure.audit import record_audit
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("bootstrap")


# --- teachers ---------------------------------------------------------------


async def load_teachers_from_yaml(
    metadata: MetadataStore, path: Path | str, *, actor: str,
) -> int:
    """Load teacher rows from a yaml file under ``conf/teachers/*.yaml``."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of teacher entries at top of {path}")
    teachers = [_teacher_from_dict(d) for d in raw]
    async with metadata.unit_of_work() as uow:
        for t in teachers:
            await uow.teachers.upsert(t)
        await record_audit(
            uow._session,  # type: ignore[attr-defined]
            actor=actor, action="teachers.load",
            entity_kind="teachers", entity_id=str(path),
            payload={"count": len(teachers), "names": [t.teacher_name for t in teachers]},
        )
    log.info("teachers.loaded", count=len(teachers), source=str(path))
    return len(teachers)


def _teacher_from_dict(d: dict[str, Any]) -> Teacher:
    return Teacher(
        teacher_name=TeacherName(d["teacher_name"]),
        model_family=d["model_family"],
        provider=d["provider"],
        endpoint=d["endpoint"],
        api_key_env=d["api_key_env"],
        tool_format=d["tool_format"],
        reasoning_format=d["reasoning_format"],
        context_window=int(d["context_window"]),
        price_per_mtoken_in=d.get("price_per_mtoken_in"),
        price_per_mtoken_out=d.get("price_per_mtoken_out"),
        active=bool(d.get("active", True)),
        meta=d.get("meta") or {},
    )


# --- environments + tasks ---------------------------------------------------


async def load_env_and_tasks(
    metadata: MetadataStore,
    *,
    env_name: EnvName,
    dataset: str,
    dataset_version: str,
    task_source: TaskSource,
    actor: str,
    batch_size: int = 256,
) -> tuple[int, int]:
    """Register the environment + bulk-insert tasks from a TaskSource.

    Returns ``(env_rows_upserted, tasks_inserted)``. Idempotent: re-running
    on the same dataset is a no-op except for the env row.
    """
    if task_source.env_name != env_name:
        raise TaskSourceError(
            f"env mismatch: task_source.env_name={task_source.env_name} vs env_name={env_name}"
        )
    total = await task_source.task_count()
    if total == 0:
        log.warning("bootstrap.empty_dataset", env_name=env_name)
    lo, hi = 0, total

    async with metadata.unit_of_work() as uow:
        await uow.tasks.upsert_environment(Environment(
            env_name=env_name,
            dataset=dataset,
            dataset_version=dataset_version,
            task_id_min=lo,
            task_id_max=hi,
        ))

    inserted = 0
    batch = []
    async for task in task_source.iter_tasks():
        batch.append(task)
        if len(batch) >= batch_size:
            inserted += await _flush_tasks(metadata, batch)
            batch = []
    if batch:
        inserted += await _flush_tasks(metadata, batch)

    async with metadata.unit_of_work() as uow:
        await record_audit(
            uow._session,  # type: ignore[attr-defined]
            actor=actor, action="env.load",
            entity_kind="environment", entity_id=str(env_name),
            payload={
                "dataset": dataset, "dataset_version": dataset_version,
                "tasks_inserted": inserted, "tasks_total": total,
            },
        )
    log.info("env.loaded",
             env_name=env_name, dataset=dataset,
             dataset_version=dataset_version, inserted=inserted, total=total)
    return 1, inserted


async def _flush_tasks(metadata: MetadataStore, batch: list) -> int:
    async with metadata.unit_of_work() as uow:
        return await uow.tasks.upsert_tasks_bulk(batch)


# --- sampling list ----------------------------------------------------------


async def init_sampling_list(
    metadata: MetadataStore,
    *,
    list_name: SamplingListName,
    env_names: Sequence[EnvName],
    teacher_names: Sequence[TeacherName],
    target_samples_per_cell: int,
    task_id_filter: tuple[int, int] | None = None,
    description: str | None = None,
    actor: str,
) -> int:
    """Persist a SamplingList + every (env, task, teacher) progress cell.

    Returns the total number of cells created.
    """
    config = {
        "env_names": list(env_names),
        "teacher_names": list(teacher_names),
        "target_samples_per_cell": target_samples_per_cell,
        "task_id_filter": list(task_id_filter) if task_id_filter else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    async with metadata.unit_of_work() as uow:
        await uow.sampling_lists.create(SamplingList(
            list_name=list_name,
            config=config,
            description=description,
            created_by=actor,
        ))
        cells: list[SamplingProgress] = []
        for env in env_names:
            env_row = await uow.tasks.get_environment(env)
            if env_row is None:
                raise ValueError(f"environment not registered: {env}")
            lo, hi = env_row.task_id_min, env_row.task_id_max
            if task_id_filter is not None:
                lo = max(lo, task_id_filter[0])
                hi = min(hi, task_id_filter[1])
            for tid, t in product(range(lo, hi), teacher_names):
                cells.append(SamplingProgress(
                    list_name=list_name, env_name=env, task_id=TaskId(tid),
                    teacher_name=t, target_samples=target_samples_per_cell,
                    collected=0,
                ))
        # bulk insert in chunks to avoid statement size limits
        chunk = 5_000
        for i in range(0, len(cells), chunk):
            await uow.sampling_lists.init_progress(cells[i:i + chunk])
        await record_audit(
            uow._session,  # type: ignore[attr-defined]
            actor=actor, action="sampling_list.init_progress",
            entity_kind="sampling_list", entity_id=str(list_name),
            payload={"cells": len(cells), "config": config},
        )
    log.info("sampling_list.initialized",
             list_name=list_name, cells=len(cells))
    return len(cells)
