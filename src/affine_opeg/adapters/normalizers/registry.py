"""Per-family normalizer registry.

Default family is ``affent`` — every trajectory we collect via the affent
agent loop carries ``family="affent"`` and the unified SSE schema. Per-family
normalizers (Claude, Qwen, ...) are kept as fallbacks for trajectories
collected outside the affent path (e.g. third-party imports).
"""

from __future__ import annotations

from affine_opeg.adapters.normalizers.affent import AffentNormalizer
from affine_opeg.adapters.normalizers.claude import ClaudeNormalizer
from affine_opeg.domain.errors import NormalizationError
from affine_opeg.domain.ports.normalizer import TrajectoryNormalizer

_REGISTRY: dict[str, TrajectoryNormalizer] = {
    "affent": AffentNormalizer(),
    "claude": ClaudeNormalizer(),
}


def get_normalizer(family: str) -> TrajectoryNormalizer:
    try:
        return _REGISTRY[family]
    except KeyError as e:
        raise NormalizationError(f"no normalizer registered for family: {family}") from e


def known_families() -> list[str]:
    return sorted(_REGISTRY)
