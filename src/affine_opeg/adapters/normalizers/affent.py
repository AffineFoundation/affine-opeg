"""affent trajectory normalizer.

Since the agent loop is run through ``affent``, every teacher's output reaches
us through the same unified SSE schema. The trajectory we receive in
``RawTrajectory.steps`` is already structured into per-turn dicts by
``adapters/sandboxes/affent_loop.py::_events_to_steps``; this normalizer's
only job is to convert those dicts into ``NormalizedMessage`` instances.

The output schema is identical to the per-family normalizers (Claude / Qwen
/ ...), so downstream eval doesn't care which teacher produced the rollout.
"""

from __future__ import annotations

from affine_opeg.domain.errors import NormalizationError
from affine_opeg.domain.models import (
    NormalizedMessage,
    NormalizedTrajectory,
    RawTrajectory,
    ToolCall,
)

SCHEMA_VERSION = "1.0-affent"


class AffentNormalizer:
    family = "affent"
    schema_version = SCHEMA_VERSION

    def normalize(self, raw: RawTrajectory) -> NormalizedTrajectory:
        if raw.family != "affent":
            raise NormalizationError(f"expected family=affent, got {raw.family}")
        messages: list[NormalizedMessage] = []
        for idx, step in enumerate(raw.steps):
            kind = step.get("kind")
            try:
                if kind == "user":
                    messages.append(NormalizedMessage(role="user", content=step.get("text", "")))
                elif kind == "assistant":
                    messages.append(self._assistant(step))
                elif kind == "tool_result":
                    messages.append(self._tool_result(step))
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
            teacher_meta=raw.agent_meta,
        )

    def _assistant(self, step: dict) -> NormalizedMessage:
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {"_raw": tc.get("arguments")},
                malformed=not isinstance(tc.get("arguments"), dict),
            )
            for tc in step.get("tool_calls") or []
        ]
        return NormalizedMessage(
            role="assistant",
            content=step.get("text"),
            reasoning=step.get("reasoning"),
            tool_calls=tool_calls,
        )

    def _tool_result(self, step: dict) -> NormalizedMessage:
        return NormalizedMessage(
            role="tool",
            tool_call_id=step.get("tool_call_id") or "",
            content=step.get("text", ""),
        )
