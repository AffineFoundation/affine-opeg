"""Task source factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from affine_opeg.adapters.task_sources.jsonl import JsonlTaskSource
from affine_opeg.adapters.task_sources.swe_rebench import SweRebenchTaskSource
from affine_opeg.domain.errors import TaskSourceError
from affine_opeg.domain.ids import EnvName
from affine_opeg.domain.ports.task_source import TaskSource


def build_task_source(kind: str, env_name: EnvName, **kwargs: Any) -> TaskSource:
    if kind == "swe_rebench":
        return SweRebenchTaskSource(env_name, Path(kwargs["path"]))
    if kind == "jsonl":
        return JsonlTaskSource(env_name, Path(kwargs["path"]))
    raise TaskSourceError(f"unknown task source kind: {kind}")


def known_kinds() -> list[str]:
    return ["swe_rebench", "jsonl"]
