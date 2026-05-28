"""Env registry — env_name → Env adapter.

Lookup is prefix-based: ``swe-rebench:python``, ``swe-rebench:hard``, …
all route to the single ``SweRebenchEnv`` instance. New benchmarks register
themselves here.
"""

from __future__ import annotations

from affine_opeg.adapters.envs.swe_rebench import SweRebenchEnv
from affine_opeg.domain.errors import TaskSourceError
from affine_opeg.domain.ids import EnvName
from affine_opeg.domain.ports.env import Env

_ENVS: list[Env] = [
    SweRebenchEnv(),
]


def get_env(env_name: EnvName) -> Env:
    """Resolve the Env handling ``env_name``. Raises if none match."""
    for env in _ENVS:
        if env.matches(env_name):
            return env
    known = ", ".join(e.name for e in _ENVS)
    raise TaskSourceError(f"no Env registered for {env_name!r} (known: {known})")


def register_env(env: Env) -> None:
    """Add an Env adapter. Inserted at the front so user-registered envs
    can override built-in defaults (rare, but useful for tests)."""
    _ENVS.insert(0, env)


def known_envs() -> list[str]:
    return [e.name for e in _ENVS]
