"""Shared scaffolding for teacher adapters.

Sub-classes only need to override ``_native_chat`` — the conversion to/from
the normalized ``ChatRequest`` / ``RawAssistantMessage`` plus retry/ratelimit
plumbing happens here.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from affine_opeg.domain.errors import TeacherError, TeacherRateLimited, TeacherTimeout
from affine_opeg.domain.models import ChatRequest, RawAssistantMessage, Teacher


class BaseTeacherClient:
    """Common semantics: rate limit, retry, latency accounting."""

    def __init__(
        self,
        teacher: Teacher,
        *,
        max_concurrency: int = 8,
        timeout_s: int = 600,
        max_retries: int = 4,
    ) -> None:
        self.teacher = teacher
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    async def chat(self, req: ChatRequest) -> RawAssistantMessage:
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_random_exponential(min=1, max=30),
                retry=retry_if_exception_type((TeacherRateLimited, TeacherTimeout)),
                reraise=True,
            ):
                with attempt:
                    started = time.monotonic()
                    try:
                        raw = await asyncio.wait_for(self._native_chat(req), timeout=self._timeout_s)
                    except asyncio.TimeoutError as e:
                        raise TeacherTimeout(str(e)) from e
                    latency_ms = int((time.monotonic() - started) * 1000)
                    return self._to_normalized(raw, latency_ms=latency_ms)
            raise TeacherError("retries exhausted")  # unreachable

    async def health_check(self) -> bool:
        try:
            await self._native_health()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ----- subclass hooks ----- #

    async def _native_chat(self, req: ChatRequest) -> Any:
        raise NotImplementedError

    async def _native_health(self) -> None:
        raise NotImplementedError

    def _to_normalized(self, raw: Any, *, latency_ms: int) -> RawAssistantMessage:
        raise NotImplementedError
