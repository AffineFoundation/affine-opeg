"""Trajectory normalizer port.

One implementation per teacher family. Routing from ``teacher.model_family`` to
the right normalizer happens in the registry, not in business code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from affine_opeg.domain.models import NormalizedTrajectory, RawTrajectory


@runtime_checkable
class TrajectoryNormalizer(Protocol):
    """Convert a teacher-specific raw trace to the canonical schema."""

    family: str
    schema_version: str

    def normalize(self, raw: RawTrajectory) -> NormalizedTrajectory:
        """Raises NormalizationError if parsing of the raw trace fails."""
