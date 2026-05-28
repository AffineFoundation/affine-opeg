"""Teacher provider registry.

All frontier teachers in production are served through OpenRouter and
therefore share the same OpenAI-compat adapter. The registry is still
keyed on ``Teacher.provider`` so future custom endpoints (Chutes-self,
Targon, internal staging) can register a different factory without
touching the application layer.
"""

from __future__ import annotations

from typing import Callable

from affine_opeg.adapters.teachers.openai_compat import OpenAICompatTeacherClient
from affine_opeg.domain.models import Teacher
from affine_opeg.domain.ports.teacher import TeacherProvider

ProviderFactory = Callable[[Teacher], TeacherProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register(provider: str) -> Callable[[ProviderFactory], ProviderFactory]:
    def _wrap(fn: ProviderFactory) -> ProviderFactory:
        _REGISTRY[provider] = fn
        return fn

    return _wrap


@register("openrouter")
@register("openai_compat")
def _make_openai_compat(teacher: Teacher) -> TeacherProvider:
    return OpenAICompatTeacherClient(teacher)


def build_teacher_provider(teacher: Teacher) -> TeacherProvider:
    """Instantiate a teacher adapter for the given Teacher domain object."""
    try:
        factory = _REGISTRY[teacher.provider]
    except KeyError as e:
        raise ValueError(f"no teacher provider registered for: {teacher.provider}") from e
    return factory(teacher)


def known_providers() -> list[str]:
    return sorted(_REGISTRY)
