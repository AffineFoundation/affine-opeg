"""Verifiers trajectory normalizer.

The verifiers agent loop packs raw OpenAI chat messages (prompt + completion)
into ``RawTrajectory.steps``; this normalizer maps each one to a
``NormalizedMessage``. Output schema is identical to the affent / per-family
normalizers, so downstream eval is agnostic to how the rollout was produced.
"""

from __future__ import annotations

import json
from typing import Any

from affine_opeg.domain.errors import NormalizationError
from affine_opeg.domain.models import (
    NormalizedMessage,
    NormalizedTrajectory,
    RawTrajectory,
    ToolCall,
)

SCHEMA_VERSION = "1.0-verifiers"

_ROLES = {"system", "user", "assistant", "tool"}


class VerifiersNormalizer:
    family = "verifiers"
    schema_version = SCHEMA_VERSION

    def normalize(self, raw: RawTrajectory) -> NormalizedTrajectory:
        if raw.family != "verifiers":
            raise NormalizationError(f"expected family=verifiers, got {raw.family}")
        messages: list[NormalizedMessage] = []
        for idx, msg in enumerate(raw.steps):
            role = msg.get("role")
            try:
                if role == "tool":
                    messages.append(self._tool(msg))
                elif role in ("system", "user", "assistant"):
                    messages.append(self._chat(role, msg))
                else:
                    raise NormalizationError(f"unknown role: {role!r}")
            except NormalizationError:
                raise
            except Exception as e:  # noqa: BLE001
                raise NormalizationError(f"message {idx} ({role}) parse failed: {e}") from e
        return NormalizedTrajectory(
            schema_version=SCHEMA_VERSION,
            messages=messages,
            reward_breakdown=raw.reward_breakdown,
            teacher_meta=raw.agent_meta,
        )

    def _chat(self, role: str, msg: dict[str, Any]) -> NormalizedMessage:
        tool_calls = [self._tool_call(tc) for tc in (msg.get("tool_calls") or [])]
        return NormalizedMessage(
            role=role,  # type: ignore[arg-type]
            content=_as_text(msg.get("content")),
            reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
            tool_calls=tool_calls,
        )

    def _tool(self, msg: dict[str, Any]) -> NormalizedMessage:
        return NormalizedMessage(
            role="tool",
            content=_as_text(msg.get("content")) or "",
            tool_call_id=msg.get("tool_call_id") or "",
        )

    @staticmethod
    def _tool_call(tc: dict[str, Any]) -> ToolCall:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        args = _safe_json(raw_args)
        return ToolCall(
            id=tc.get("id", ""),
            name=fn.get("name", ""),
            arguments=args if isinstance(args, dict) else {"_raw": args},
            malformed=not isinstance(args, dict),
        )


def _as_text(content: Any) -> str | None:
    """OpenAI ``content`` may be a string or a list of content parts."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts)
    return str(content)


def _safe_json(s: Any) -> Any:
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s) if s else {}
    except Exception:  # noqa: BLE001
        return {"_raw": s}
