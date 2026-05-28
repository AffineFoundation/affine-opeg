"""Task source port (SWE-rebench / SWE-Gym / ...)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from affine_opeg.domain.ids import EnvName, TaskId
from affine_opeg.domain.models import Task


@runtime_checkable
class TaskSource(Protocol):
    """One task source (one ``env_name`` namespace)."""

    env_name: EnvName

    async def load_task(self, task_id: TaskId) -> Task:
        """Materialize a single task by id."""

    def iter_tasks(self, *, start: TaskId | None = None, end: TaskId | None = None) -> AsyncIterator[Task]:
        """Stream tasks in id order. Used by ingestion."""

    async def task_count(self) -> int: ...
