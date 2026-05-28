"""Generic OpenAI-compat teacher adapter.

Speaks ``/v1/chat/completions`` against any OpenAI-compatible gateway
(OpenRouter, Chutes, vLLM, OpenAI itself, ...). Used for health checks
and any out-of-affent-loop diagnostics; the producer's hot path runs
through affent which speaks the same protocol natively.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from affine_opeg.adapters.teachers.base import BaseTeacherClient
from affine_opeg.domain.errors import (
    TeacherBadResponse,
    TeacherRateLimited,
    TeacherTimeout,
)
from affine_opeg.domain.models import (
    ChatRequest,
    RawAssistantMessage,
    Teacher,
)


class OpenAICompatTeacherClient(BaseTeacherClient):
    """Health-check / smoke-test path. NOT on the rollout hot path."""

    family = "openai_compat"

    def __init__(self, teacher: Teacher, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(teacher, **kwargs)
        self._api_key = api_key or os.environ.get(teacher.api_key_env, "")
        if not self._api_key:
            raise RuntimeError(f"missing api key in env var: {teacher.api_key_env}")
        self._client = httpx.AsyncClient(
            base_url=teacher.endpoint.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # OpenRouter requires HTTP-Referer / X-Title for analytics;
                # they're optional but recommended.
                "HTTP-Referer": "https://affine.io",
                "X-Title": "affine-cortex-generator",
            },
            timeout=httpx.Timeout(self._timeout_s, connect=10),
        )

    async def _native_chat(self, req: ChatRequest) -> dict[str, Any]:
        body = self._serialize_request(req)
        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise TeacherTimeout(str(e)) from e
        except httpx.HTTPError as e:
            raise TeacherBadResponse(str(e)) from e

        if resp.status_code == 429:
            raise TeacherRateLimited(resp.text)
        if resp.status_code >= 500:
            raise TeacherTimeout(resp.text)
        if resp.status_code >= 400:
            raise TeacherBadResponse(f"{resp.status_code}: {resp.text}")
        try:
            return resp.json()
        except ValueError as e:
            raise TeacherBadResponse(f"non-json response: {resp.text[:200]}") from e

    async def _native_health(self) -> None:
        # OpenRouter exposes /models for inventory; works as a cheap GET.
        resp = await self._client.get("/models", timeout=10)
        if resp.status_code >= 500:
            raise TeacherTimeout(f"models endpoint returned {resp.status_code}")
        if resp.status_code >= 400:
            raise TeacherBadResponse(f"models endpoint returned {resp.status_code}")

    def _serialize_request(self, req: ChatRequest) -> dict[str, Any]:
        """OpenAI-style ``messages`` payload."""
        messages: list[dict[str, Any]] = []
        for m in req.messages:
            if m.role == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content or "",
                })
                continue
            entry: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.role == "assistant" and m.tool_calls:
                entry["tool_calls"] = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                } for tc in m.tool_calls]
            messages.append(entry)
        body: dict[str, Any] = {
            "model": self.teacher.teacher_name,
            "messages": messages,
            "temperature": req.temperature,
        }
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.tools:
            body["tools"] = req.tools
        if req.stop:
            body["stop"] = req.stop
        body.update(req.extra_provider_args)
        return body

    def _to_normalized(self, raw: dict[str, Any], *, latency_ms: int) -> RawAssistantMessage:
        choices = raw.get("choices") or [{}]
        msg = choices[0].get("message", {})
        usage = raw.get("usage", {})
        return RawAssistantMessage(
            teacher_name=self.teacher.teacher_name,
            raw=raw,
            text=msg.get("content"),
            finish_reason=choices[0].get("finish_reason"),
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
