"""Null sandbox — a no-op ``Sandbox`` for container-free rollouts.

Some agent loops (notably the verifiers / PI loop) do all their work
in-process against an HTTP endpoint: there is no repo to check out, no
container to exec into, and scoring is the env's own rubric rather than a
hidden test suite. ``generate_rollout`` still acquires a sandbox
unconditionally, so we hand it this placeholder. It carries only the
``task`` the loop needs and reports ``container_id = None``.

Concurrency is still bounded here so the producer's global episode cap means
the same thing in both modes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from affine_opeg.domain.models import Task


class NullSandbox:
    """A sandbox that owns nothing. ``run_hidden_tests`` / ``extract_patch``
    are inert — verifiers envs score inside the agent loop instead."""

    def __init__(self, task: Task) -> None:
        self.task = task
        self.workspace_path = ""
        self.container_id: str | None = None

    async def run_hidden_tests(self) -> dict[str, Any]:
        return {}

    async def extract_patch(self) -> str:
        return ""


class NullSandboxFactory:
    """``SandboxFactory`` that yields :class:`NullSandbox` instances."""

    def __init__(self, *, max_concurrent: int = 32) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)

    @asynccontextmanager
    async def acquire(self, task: Task) -> AsyncIterator[NullSandbox]:
        async with self._sem:
            yield NullSandbox(task)
