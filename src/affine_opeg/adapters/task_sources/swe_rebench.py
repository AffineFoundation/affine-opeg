"""SWE-rebench task source.

SWE-rebench rows ship with a slightly different field layout than vanilla
SWE-bench (``dockerhub_tag`` instead of ``docker_image``, sometimes a string
``instance_id`` as the canonical key). This thin wrapper around
:class:`JsonlTaskSource` normalises those quirks at load time so downstream
code never has to special-case them.

Usage:
    afr admin load-tasks swe-rebench:python /data/swe-rebench-python.jsonl
"""

from __future__ import annotations

from pathlib import Path

from affine_opeg.adapters.task_sources.jsonl import JsonlTaskSource
from affine_opeg.domain.ids import EnvName


class SweRebenchTaskSource(JsonlTaskSource):
    """SWE-rebench-flavoured JSONL loader."""

    def __init__(self, env_name: EnvName, path: str | Path) -> None:
        super().__init__(env_name, path)
