"""JSONL-backed task source.

Loads tasks from a local JSONL file — one ``{task_id, repo, base_commit,
problem, hidden_tests, ...}`` object per line. The intent is that operators
prepare the dataset outside the runtime (HuggingFace dump, R2 sync, manual
hand-curate) and feed the resulting file into the system via
``afr admin load-tasks``.

This keeps the producer / api container free of heavy dataset SDKs and makes
the input fully reproducible (one JSONL is checksum-able).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from affine_opeg.domain.errors import TaskSourceError
from affine_opeg.domain.ids import EnvName, TaskId
from affine_opeg.domain.models import Task


class JsonlTaskSource:
    """One task per line. Field names match :class:`Task`."""

    def __init__(self, env_name: EnvName, path: str | Path) -> None:
        self.env_name = env_name
        self._path = Path(path)
        if not self._path.is_file():
            raise TaskSourceError(f"file not found: {self._path}")
        self._index: dict[int, int] | None = None     # task_id -> byte offset

    async def task_count(self) -> int:
        count = 0
        with self._path.open() as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    async def load_task(self, task_id: TaskId) -> Task:
        if self._index is None:
            self._index = self._build_index()
        offset = self._index.get(int(task_id))
        if offset is None:
            raise TaskSourceError(f"task {task_id} not found in {self._path.name}")
        with self._path.open() as f:
            f.seek(offset)
            raw = json.loads(f.readline())
        return self._build_task(raw)

    async def iter_tasks(
        self, *, start: TaskId | None = None, end: TaskId | None = None,
    ) -> AsyncIterator[Task]:
        lo = int(start) if start is not None else None
        hi = int(end) if end is not None else None
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = self._task_id_of(raw)
                if lo is not None and tid < lo:
                    continue
                if hi is not None and tid >= hi:
                    continue
                yield self._build_task(raw)

    # ---- helpers --------------------------------------------------------- #

    def _build_index(self) -> dict[int, int]:
        idx: dict[int, int] = {}
        offset = 0
        with self._path.open("rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped:
                    try:
                        obj = json.loads(stripped)
                        idx[self._task_id_of(obj)] = offset
                    except json.JSONDecodeError:
                        pass
                offset += len(line)
        return idx

    @staticmethod
    def _task_id_of(raw: dict[str, Any]) -> int:
        # Accept both numeric ``task_id`` and string ``instance_id`` (hashed)
        if "task_id" in raw:
            return int(raw["task_id"])
        if "id" in raw:
            try:
                return int(raw["id"])
            except (TypeError, ValueError) as e:
                raise TaskSourceError(f"non-numeric id: {raw['id']!r}") from e
        raise TaskSourceError("row missing task_id/id field")

    def _build_task(self, raw: dict[str, Any]) -> Task:
        task_id = self._task_id_of(raw)
        return Task(
            env_name=self.env_name,
            task_id=TaskId(task_id),
            repo=raw.get("repo", ""),
            base_commit=raw.get("base_commit", ""),
            problem=raw.get("problem") or raw.get("problem_statement") or "",
            hidden_tests=_extract_hidden_tests(raw),
            difficulty=raw.get("difficulty"),
            meta=_extract_meta(raw),
        )


def _extract_hidden_tests(raw: dict[str, Any]) -> dict[str, Any]:
    """Pull SWE-bench-style fields under a single ``hidden_tests`` payload.

    ``setup_files`` is also lifted — ``{rel_path: content}`` mappings the
    sandbox materialises into the workspace before the agent runs. This is
    how from-scratch smoke tasks seed the workspace when the docker image
    does not already contain a repo (real SWE-rebench images do).
    """
    fields = ("test_patch", "augmented_test_patch", "fail_to_pass",
              "pass_to_pass", "test_command", "setup_files", "setup_commands")
    return {k: raw[k] for k in fields if k in raw}


def _extract_meta(raw: dict[str, Any]) -> dict[str, Any]:
    """Surface fields the sandbox / agent need at runtime."""
    meta: dict[str, Any] = {}
    for key in ("docker_image", "dockerhub_tag", "repo_language",
                "test_command", "instance_id", "version", "workspace_path"):
        if key in raw:
            meta[key] = raw[key]
    # Normalise the image-name field.
    if "docker_image" not in meta and "dockerhub_tag" in meta:
        meta["docker_image"] = meta["dockerhub_tag"]
    return meta
