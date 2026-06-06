"""Verifiers-backed task source.

Turns the dataset that a verifiers ``Environment`` exposes into ``Task`` rows.
Each dataset row already carries everything a rollout needs — the chat
``prompt``, the gold ``answer``, and an opaque ``info`` blob — so we stash
those verbatim under ``task.meta`` and the agent loop reconstructs the
verifiers ``RolloutInput`` from them at rollout time. No container, no
test_patch, no docker image: rollout *and* scoring both live inside the env.

``task_id`` is the row index in the (deterministic) dataset, which is what
makes ingestion and sampling agree on the same id space.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from affine_opeg.adapters.verifiers_runtime import env_id_of, load_verifiers_env
from affine_opeg.domain.errors import TaskSourceError
from affine_opeg.domain.ids import EnvName, TaskId
from affine_opeg.domain.models import Task


class VerifiersTaskSource:
    """One env_name namespace (``verifiers:<env_id>``) -> Task rows."""

    def __init__(self, env_name: EnvName, **env_args: Any) -> None:
        self.env_name = env_name
        self._env_id = env_id_of(env_name)
        self._env_args = env_args
        self._rows: list[dict[str, Any]] | None = None

    def _load_rows(self) -> list[dict[str, Any]]:
        if self._rows is None:
            env = load_verifiers_env(self._env_id, **self._env_args)
            ds = _resolve_dataset(env)
            if ds is None:
                raise TaskSourceError(f"verifiers env {self._env_id!r} has no dataset")
            self._rows = list(ds)
        return self._rows

    async def task_count(self) -> int:
        return len(self._load_rows())

    async def load_task(self, task_id: TaskId) -> Task:
        rows = self._load_rows()
        tid = int(task_id)
        if tid < 0 or tid >= len(rows):
            raise TaskSourceError(f"task {task_id} out of range for {self.env_name}")
        return self._build_task(tid, rows[tid])

    async def iter_tasks(
        self, *, start: TaskId | None = None, end: TaskId | None = None,
    ) -> AsyncIterator[Task]:
        rows = self._load_rows()
        lo = int(start) if start is not None else 0
        hi = int(end) if end is not None else len(rows)
        for tid in range(max(lo, 0), min(hi, len(rows))):
            yield self._build_task(tid, rows[tid])

    # ---- helpers --------------------------------------------------------- #

    def _build_task(self, task_id: int, row: dict[str, Any]) -> Task:
        prompt = row.get("prompt")
        answer = row.get("answer")
        info = row.get("info") or {}
        example_id = row.get("example_id", task_id)
        return Task(
            env_name=self.env_name,
            task_id=TaskId(task_id),
            repo="",
            base_commit="",
            problem=_problem_text(prompt),
            hidden_tests={},  # scoring is the env's rubric, not a hidden test suite
            meta={
                "verifiers_env_id": self._env_id,
                "prompt": prompt,
                "answer": answer,
                "info": info,
                "example_id": example_id,
            },
        )


def _resolve_dataset(env: Any) -> Any:
    """Return the env's task dataset.

    Train-oriented envs expose rows via ``get_dataset()``; eval-only
    benchmarks (math500, mmlu-pro, ifeval, …) leave ``dataset`` unset and
    expose rows via ``get_eval_dataset()`` instead. Try train first, fall
    back to eval — that single fallback is what lets one adapter cover both
    families of hub env.
    """
    for meth in ("get_dataset", "get_eval_dataset"):
        fn = getattr(env, meth, None)
        if fn is None:
            continue
        try:
            ds = fn()
        except Exception:  # noqa: BLE001 — env raises when that split is unset
            continue
        if ds is not None:
            return ds
    return None


def _problem_text(prompt: Any) -> str:
    """Best-effort human-readable text for the ``problem`` column.

    ``prompt`` is normally a list of chat messages; we join the user-visible
    content. Falls back to ``str(prompt)`` for plain-string prompts.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") in ("user", "system"):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
        if parts:
            return "\n\n".join(parts)
    return str(prompt) if prompt is not None else ""
