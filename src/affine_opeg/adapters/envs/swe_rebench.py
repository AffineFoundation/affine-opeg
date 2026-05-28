"""SWE-rebench evaluation environment.

Encapsulates everything benchmark-specific:

  * task source — JSONL produced by ``scripts/prepare_swe_rebench.py``
    (column quirks like ``dockerhub_tag`` are already normalised by
    :class:`SweRebenchTaskSource`).
  * image / workspace — both live in ``task.meta`` (the prepare script
    fills ``workspace_path = "/testbed"`` and ``docker_image = <ref>``);
    falls back to ``/app`` for smoke tasks that bake the repo on
    ``python:3.11-slim``.
  * evaluator — applies the hidden ``test_patch``, runs the configured
    ``pytest`` invocation under the ``testbed`` conda env, parses the
    pytest summary into ``(passed, total)``.
  * patch extractor — ``git diff`` of code-extension files only.
  * system prompt — the SWE-bench-style "do not touch tests, smallest diff"
    overlay; consumed by the agent loop.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from affine_opeg.adapters.task_sources.swe_rebench import SweRebenchTaskSource
from affine_opeg.domain.errors import SandboxError
from affine_opeg.domain.ids import EnvName
from affine_opeg.domain.models import Task
from affine_opeg.domain.ports.env import Env, Evaluator, PatchExtractor, SandboxExec
from affine_opeg.domain.ports.task_source import TaskSource
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("env.swe_rebench")

_CODE_GLOBS = ("*.py", "*.go", "*.js", "*.ts", "*.rs", "*.java", "*.cpp", "*.c", "*.h")

_PYTEST_WRAP = (
    "if [ -f /opt/conda/etc/profile.d/conda.sh ]; then "
    "source /opt/conda/etc/profile.d/conda.sh && "
    "conda activate testbed >/dev/null 2>&1 || true; fi"
)

_SYSTEM_PROMPT = """\
You are a software engineering agent solving a real GitHub PR task.

Your workspace is {workspace} — the repository root. Use the available
tools (shell, read_file, write_file, edit_file, list_files) to:

  1. Read the relevant source files to understand the codebase.
  2. Make minimal, focused changes that directly address the task.
  3. Modify ONLY source code files. Do NOT touch tests, fixtures, or
     configuration files.
  4. Keep changes contained: prefer the smallest diff that resolves
     the issue. Do not refactor unrelated code.

When you believe the task is complete, stop calling tools and reply
with a brief summary of what you changed. The framework will extract
the diff from the workspace automatically — you do NOT need to print
the patch yourself.
"""


class PytestEvaluator:
    """SWE-rebench reward grader.

    Steps: apply test_patch (if any) → run test_command under the testbed
    conda env → parse pytest summary. Score is the simple ``passed/total``
    fraction; ``0/0`` is treated as score=0 so a crashed run is not
    silently rewarded.
    """

    def __init__(self, test_command: str, test_patch: str | None) -> None:
        self._test_command = test_command
        self._test_patch = test_patch

    async def evaluate(self, sandbox: SandboxExec) -> dict[str, Any]:
        if self._test_patch:
            await sandbox.apply_patch("/tmp/test.patch", self._test_patch)
        script = (
            f"{_PYTEST_WRAP} && "
            f"cd {shlex.quote(sandbox.workspace_path)} && "
            f"{self._test_command} || true"
        )
        rc, out, err = await sandbox.exec(script, timeout=600)
        passed, total = _parse_pytest(out + "\n" + err)
        score = (passed / total) if total > 0 else 0.0
        return {
            "score": score,
            "tests_passed": passed,
            "tests_total": total,
            "exit_code": rc,
            "raw_output": (out + err)[-4096:],
        }


class GitDiffPatchExtractor:
    """``git diff --cached`` over a fixed list of code globs.

    Restricted to source extensions so the diff doesn't pick up stray
    artefacts (`*.pyc`, temporary install state) the agent's actions may
    have left behind.
    """

    def __init__(self, code_globs: tuple[str, ...] = _CODE_GLOBS) -> None:
        self._globs = code_globs

    async def extract(self, sandbox: SandboxExec) -> str:
        glob_args = " ".join(f"'{g}'" for g in self._globs)
        rc, out, _ = await sandbox.exec(
            f"cd {shlex.quote(sandbox.workspace_path)} && "
            f"git add -A && git diff --cached -- {glob_args}",
            timeout=60,
        )
        if rc != 0:
            return ""
        patch = out.lstrip()
        return patch.rstrip("\n") + "\n" if patch else ""


class SweRebenchEnv:
    """Env binding for SWE-rebench (v1) — also handles ``swe-rebench:*`` slices."""

    name = "swe-rebench"

    def matches(self, env_name: EnvName) -> bool:
        base = str(env_name).split(":", 1)[0]
        return base == self.name

    def task_source(self, env_name: EnvName, path: Path) -> TaskSource:
        return SweRebenchTaskSource(env_name, path)

    def image(self, task: Task) -> str:
        image = task.meta.get("docker_image") or task.meta.get("image")
        if not image:
            raise SandboxError(
                f"task {task.env_name}:{task.task_id} missing docker_image in meta",
            )
        return str(image)

    def workspace_path(self, task: Task) -> str:
        # SWE-rebench images bake the repo at /testbed. Smoke tasks
        # (helloworld.jsonl on python:3.11-slim) fall back to /app.
        return str(task.meta.get("workspace_path") or "/app")

    def evaluator(self, task: Task) -> Evaluator:
        ht = task.hidden_tests or {}
        cmd = ht.get("test_command") or task.meta.get("test_command") or "pytest -q"
        return PytestEvaluator(test_command=cmd, test_patch=ht.get("test_patch"))

    def patch_extractor(self, task: Task) -> PatchExtractor:
        return GitDiffPatchExtractor()

    def system_prompt(self, task: Task) -> str | None:
        return _SYSTEM_PROMPT


# --------------------------------------------------------------------------- #


def _parse_pytest(output: str) -> tuple[int, int]:
    """Heuristic pytest summary parser.

    Scans the last ~50 lines for ``N passed`` / ``N failed`` / ``N error``
    tokens. Returns ``(passed, total)``; ``(0, 0)`` if no summary found, in
    which case caller treats score=0.
    """
    passed = failed = errored = 0
    for line in output.splitlines()[-50:]:
        if m := re.search(r"(\d+)\s+passed", line):
            passed = int(m.group(1))
        if m := re.search(r"(\d+)\s+failed", line):
            failed = int(m.group(1))
        if m := re.search(r"(\d+)\s+error", line):
            errored = int(m.group(1))
    total = passed + failed + errored
    return passed, total
