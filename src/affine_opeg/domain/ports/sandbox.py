"""Sandbox port.

A Sandbox is a one-shot, per-rollout scratch environment:

    1. ``setup`` brings up a container with the task's repo at base_commit
       and produces a ``workspace_path`` the agent will operate inside.
    2. The agent (affent) is given that path and runs autonomously — it owns
       its own tools (shell / read_file / edit_file / write_file / list_files)
       and exits when the loop terminates.
    3. ``run_hidden_tests`` evaluates the resulting workspace against the
       task's hidden test suite and returns a ``reward_breakdown``-shaped dict.
    4. ``teardown`` (always called) removes the container.

The Sandbox does NOT expose view / edit / bash to the producer because affent
takes ownership of tool execution inside the container. The producer's job is
to acquire a workspace, hand it to affent, then score.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from affine_opeg.domain.models import Task


@runtime_checkable
class Sandbox(Protocol):
    """Per-rollout scratch environment."""

    task: Task
    workspace_path: str            # absolute path the agent will see (e.g. /app)
    container_id: str | None       # for diagnostics; None for non-container sandboxes

    async def run_hidden_tests(self) -> dict[str, Any]:
        """Run the task's hidden tests against the current workspace.

        Returns a dict shaped like ``{score, tests_passed, tests_total, ...}``
        that becomes ``Rollout.reward_breakdown`` verbatim.
        """

    async def extract_patch(self) -> str:
        """Return the git diff of agent-made changes (empty string if none)."""


@runtime_checkable
class SandboxFactory(Protocol):
    """Creates / disposes sandboxes. Implementations are concurrency-limited."""

    def acquire(self, task: Task) -> AbstractAsyncContextManager[Sandbox]:
        """Async context manager. Releases the underlying container on exit."""
