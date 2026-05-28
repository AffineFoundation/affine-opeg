"""Claude trajectory normalizer.

Input shape (``RawTrajectory.steps`` for family='claude'):
    Each step represents one agent loop turn::

        {
            "kind": "user" | "assistant" | "tool_result",
            "content_blocks": [...],          # Anthropic response or tool-result blocks
            "tool_call_id": "..." | null,
        }

Output: ``NormalizedTrajectory`` with system/user/assistant/tool messages and
reasoning lifted out of assistant content where present.
"""

from __future__ import annotations

import re
from typing import Any

from affine_opeg.domain.errors import NormalizationError
from affine_opeg.domain.models import (
    NormalizedMessage,
    NormalizedTrajectory,
    RawTrajectory,
    ToolCall,
)

SCHEMA_VERSION = "1.0"
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", flags=re.DOTALL | re.IGNORECASE)


class ClaudeNormalizer:
    family = "claude"
    schema_version = SCHEMA_VERSION

    def normalize(self, raw: RawTrajectory) -> NormalizedTrajectory:
        if raw.family != "claude":
            raise NormalizationError(f"expected family=claude, got {raw.family}")
        messages: list[NormalizedMessage] = []
        for idx, step in enumerate(raw.steps):
            kind = step.get("kind")
            try:
                if kind == "user":
                    messages.append(self._user_message(step))
                elif kind == "tool_result":
                    messages.append(self._tool_message(step))
                elif kind == "assistant":
                    messages.append(self._assistant_message(step))
                else:
                    raise NormalizationError(f"unknown step kind: {kind!r}")
            except NormalizationError:
                raise
            except Exception as e:  # noqa: BLE001
                raise NormalizationError(f"step {idx} ({kind}) parse failed: {e}") from e
        return NormalizedTrajectory(
            schema_version=SCHEMA_VERSION,
            messages=messages,
            reward_breakdown=raw.reward_breakdown,
            agent_trace=None,
            teacher_meta=raw.agent_meta,
        )

    # --------- per-kind handlers --------- #

    def _user_message(self, step: dict[str, Any]) -> NormalizedMessage:
        text = step.get("text")
        if text is None:
            blocks = step.get("content_blocks") or []
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return NormalizedMessage(role="user", content=text or "")

    def _tool_message(self, step: dict[str, Any]) -> NormalizedMessage:
        tool_call_id = step.get("tool_call_id")
        if not tool_call_id:
            raise NormalizationError("tool_result step missing tool_call_id")
        text = step.get("text")
        if text is None:
            blocks = step.get("content_blocks") or []
            parts: list[str] = []
            for b in blocks:
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    inner = b.get("content")
                    if isinstance(inner, str):
                        parts.append(inner)
                    elif isinstance(inner, list):
                        parts.extend(x.get("text", "") for x in inner if x.get("type") == "text")
            text = "\n".join(p for p in parts if p)
        return NormalizedMessage(role="tool", tool_call_id=tool_call_id, content=text or "")

    def _assistant_message(self, step: dict[str, Any]) -> NormalizedMessage:
        blocks = step.get("content_blocks") or []
        texts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for b in blocks:
            kind = b.get("type")
            if kind == "text":
                t = b.get("text", "")
                # Extract Claude-style <thinking>...</thinking> as reasoning.
                without_thinking, found = _THINKING_RE.subn("", t)
                for m in _THINKING_RE.finditer(t):
                    reasoning_parts.append(m.group(1).strip())
                if found and not without_thinking.strip():
                    continue
                texts.append(without_thinking.strip() if found else t)
            elif kind == "thinking":  # newer Claude extended-thinking blocks
                reasoning_parts.append(b.get("thinking") or b.get("text", ""))
            elif kind == "tool_use":
                tool_calls.append(self._tool_call(b))
            else:
                # Unknown block — keep as malformed evidence in reasoning so
                # downstream observers can spot drift, instead of dropping silently.
                reasoning_parts.append(f"[unhandled block type={kind}]")

        return NormalizedMessage(
            role="assistant",
            content="\n".join(t for t in texts if t) or None,
            reasoning="\n".join(p for p in reasoning_parts if p) or None,
            tool_calls=tool_calls,
        )

    def _tool_call(self, block: dict[str, Any]) -> ToolCall:
        tool_id = block.get("id")
        name = block.get("name")
        if not tool_id or not name:
            raise NormalizationError(f"tool_use block missing id/name: {block!r}")
        args = block.get("input")
        if isinstance(args, dict):
            return ToolCall(id=tool_id, name=name, arguments=args)
        # Anthropic occasionally returns a JSON string for input.
        if isinstance(args, str):
            import orjson
            try:
                parsed = orjson.loads(args)
                if isinstance(parsed, dict):
                    return ToolCall(id=tool_id, name=name, arguments=parsed)
            except orjson.JSONDecodeError:
                pass
            return ToolCall(id=tool_id, name=name, arguments={"_raw": args}, malformed=True)
        return ToolCall(id=tool_id, name=name, arguments={}, malformed=True)
