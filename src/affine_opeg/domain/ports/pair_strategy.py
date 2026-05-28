"""Pair selection strategy port.

Stateless function-like protocol. New strategies are added by writing a new
adapter and registering it under a name; no main-flow change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from affine_opeg.domain.models import PairCandidate, Rollout


@runtime_checkable
class PairStrategy(Protocol):
    """Selects pairs out of one ``same-(task, teacher)`` rollout group."""

    name: str
    version: str

    def select(self, group: list[Rollout]) -> list[PairCandidate]:
        """Return zero or more pair candidates. Must NOT mutate input."""
