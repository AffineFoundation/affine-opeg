"""Teacher provider port.

Each frontier teacher (claude, gpt, qwen, ...) has one adapter implementing
this Protocol. The adapter is stateless w.r.t. business data — it only knows
about the API contract and provider-specific tool/reasoning format.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from affine_opeg.domain.models import ChatRequest, RawAssistantMessage, Teacher


@runtime_checkable
class TeacherProvider(Protocol):
    """One frontier teacher endpoint."""

    teacher: Teacher

    async def chat(self, req: ChatRequest) -> RawAssistantMessage:
        """Execute one chat completion. Raises domain TeacherError on failure."""

    async def health_check(self) -> bool:
        """Cheap synthetic call to verify the endpoint is reachable."""


@runtime_checkable
class TeacherRegistry(Protocol):
    """Discover and instantiate teacher providers by name."""

    def get(self, teacher_name: str) -> TeacherProvider: ...
    def list_active(self) -> list[TeacherProvider]: ...
