"""Evaluation environment port.

An ``Env`` bundles everything that is *specific* to one kind of evaluation
benchmark — task source, container image, reward grading, patch extraction,
agent system prompt overlay. Adding a new benchmark (SWE-rebench V2, a math
suite, a JS-only set, …) means writing one new ``Env`` and registering it,
nothing else.

The sandbox / producer / consumer layers only see ``Env`` and the per-task
``Evaluator`` / ``PatchExtractor`` it produces — they never special-case a
benchmark.

Why two layers (Env + Evaluator):
    ``Env`` is shared across all tasks of one benchmark — cheap to construct,
    stateless. ``Evaluator`` / ``PatchExtractor`` may carry per-task config
    (which test command, which language toolchain, which file globs for
    ``git diff``) so we instantiate them per-task on sandbox acquisition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from affine_opeg.domain.ids import EnvName
from affine_opeg.domain.models import Task
from affine_opeg.domain.ports.task_source import TaskSource


@runtime_checkable
class SandboxExec(Protocol):
    """The slice of sandbox capabilities an Evaluator / PatchExtractor needs.

    Kept narrow on purpose: evaluators must not depend on container internals
    (container_id, docker CLI) — that way the same evaluator runs unchanged
    against future non-docker sandboxes (firecracker, modal, etc.).
    """

    workspace_path: str
    task: Task

    async def exec(self, script: str, *, timeout: int = 60) -> tuple[int, str, str]:
        """Run ``bash -c script`` inside the sandbox. Returns (rc, stdout, stderr)."""

    async def write_file(self, container_path: str, content: str) -> None:
        """Materialise ``content`` at ``container_path`` inside the sandbox."""

    async def apply_patch(self, container_path: str, patch_text: str) -> None:
        """Write ``patch_text`` to ``container_path`` and apply it (``patch -p1`` or ``git apply``)."""


@runtime_checkable
class Evaluator(Protocol):
    """Turn a finished sandbox state into a reward breakdown.

    Returns ``{score: float, ...}``. ``score`` becomes the rollout reward;
    additional keys (tests_passed, raw_output, exit_code, …) are preserved
    verbatim in ``Rollout.reward_breakdown``.
    """

    async def evaluate(self, sandbox: SandboxExec) -> dict[str, Any]: ...


@runtime_checkable
class PatchExtractor(Protocol):
    """Extract the agent-authored diff from the sandbox workspace."""

    async def extract(self, sandbox: SandboxExec) -> str: ...


@runtime_checkable
class Env(Protocol):
    """An evaluation environment (one benchmark family)."""

    name: str  # canonical prefix used for env_name lookup (e.g. "swe-rebench")

    def matches(self, env_name: EnvName) -> bool:
        """Whether this Env handles tasks under ``env_name`` (e.g. prefix match)."""

    def task_source(self, env_name: EnvName, path: Path) -> TaskSource:
        """Build a TaskSource for the given env_name + dataset path."""

    def image(self, task: Task) -> str:
        """Container image to start for this task."""

    def workspace_path(self, task: Task) -> str:
        """Absolute path inside the container the agent operates on."""

    def evaluator(self, task: Task) -> Evaluator:
        """Per-task reward grader."""

    def patch_extractor(self, task: Task) -> PatchExtractor:
        """Per-task diff extractor."""

    def system_prompt(self, task: Task) -> str | None:
        """Optional per-env system prompt for the agent loop.

        ``None`` means "fall back to the agent loop's default". Use this to
        keep benchmark-specific phrasing (e.g. "do not edit tests") out of
        the generic loop adapter.
        """
