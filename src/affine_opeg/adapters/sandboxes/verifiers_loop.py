"""Verifiers (PrimeIntellect) agent loop.

This is the ``AgentLoopFn`` for the container-free PI pathway. Where the
affent loop drives a tool-using agent inside a docker sandbox, this loop
hands the whole rollout to a verifiers ``Environment``:

    1. Reconstruct the ``RolloutInput`` from ``task.meta`` (stashed there by
       :class:`VerifiersTaskSource`).
    2. ``env.rollout(input, client, model, sampling_args)`` — verifiers makes
       the (multi-turn, possibly tool-using) LLM calls against the teacher's
       OpenAI-compatible endpoint and returns a ``state`` dict.
    3. ``env.rubric.score_rollout(state)`` — the env's own rubric writes
       ``reward`` / ``metrics`` into ``state`` in place.
    4. Pack the full message list (prompt + completion) as raw chat-message
       dicts into ``RawTrajectory.steps`` with ``family="verifiers"``; the
       :class:`VerifiersNormalizer` maps those to ``NormalizedMessage``.

The sandbox argument is a :class:`NullSandbox` — only ``sandbox.task`` is
read; there is no container to exec into.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import verifiers as vf

from affine_opeg.adapters.verifiers_runtime import load_verifiers_env
from affine_opeg.domain.errors import TeacherError
from affine_opeg.domain.models import RawTrajectory, Task, Teacher
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("verifiers_loop")


@dataclass(frozen=True)
class VerifiersLoopConfig:
    max_tokens: int = 16384
    # verifiers reads the bearer key from an env var *name* (not the value).
    # We honour the teacher's ``api_key_env`` so creds stay out of payloads.
    client_type: str = "openai_chat_completions"


class VerifiersAgentLoop:
    """``AgentLoopFn``: (teacher, sandbox, params) -> RawTrajectory."""

    def __init__(self, cfg: VerifiersLoopConfig | None = None) -> None:
        self.cfg = cfg or VerifiersLoopConfig()

    async def __call__(self, teacher: Teacher, sandbox: Any, params: Any) -> RawTrajectory:
        task: Task = sandbox.task
        env_id = task.meta.get("verifiers_env_id")
        if not env_id:
            raise TeacherError(
                f"task {task.env_name}:{task.task_id} missing verifiers_env_id in meta",
            )

        # verifiers needs the key present in the process env under this name.
        if not os.environ.get(teacher.api_key_env):
            raise TeacherError(f"missing api key env var: {teacher.api_key_env}")

        env = load_verifiers_env(env_id)
        served_model = (teacher.meta or {}).get("served_model") or str(teacher.teacher_name)

        client = vf.ClientConfig(
            api_base_url=teacher.endpoint,
            api_key_var=teacher.api_key_env,
            client_type=self.cfg.client_type,
        )
        sampling_args = self._sampling_args(params, teacher)
        rollout_input = self._rollout_input(task)

        state = await env.rollout(rollout_input, client, served_model, sampling_args)
        await env.rubric.score_rollout(state)

        prompt_msgs = state.get("prompt") or rollout_input.get("prompt") or []
        completion = state.get("completion") or []
        steps = [_msg_to_dict(m) for m in prompt_msgs] + [_msg_to_dict(m) for m in completion]

        reward = state.get("reward")
        rmetrics = state.get("metrics") or {}
        usage = _usage_dict(state.get("usage"))
        reward_breakdown: dict[str, Any] = {
            "score": float(reward) if reward is not None else 0.0,
        }
        for k, v in rmetrics.items():
            reward_breakdown.setdefault(k, v)

        agent_meta: dict[str, Any] = {
            "env_id": env_id,
            "model": served_model,
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "usage": usage,
            "num_messages": len(steps),
        }
        if not completion:
            log.warning(
                "verifiers.empty_completion",
                env_id=env_id, task_id=task.task_id, reward=reward,
            )

        return RawTrajectory(
            teacher_name=teacher.teacher_name,
            family="verifiers",
            steps=steps,
            reward_breakdown=reward_breakdown,
            agent_meta=agent_meta,
        )

    # ---- helpers --------------------------------------------------------- #

    def _sampling_args(self, params: Any, teacher: Teacher | None = None) -> dict[str, Any]:
        # Clamp max_tokens to the teacher's server-side completion cap when set
        # (teacher.meta["max_completion_tokens"]). Some chutes serve a lower
        # ceiling (e.g. MiniMax-M2.5-TEE allows 8192) and 400 on larger requests
        # -> empty completion -> reward 0.
        max_tokens = self.cfg.max_tokens
        if teacher is not None:
            cap = (teacher.meta or {}).get("max_completion_tokens")
            if isinstance(cap, int) and cap > 0:
                max_tokens = min(max_tokens, cap)
        args: dict[str, Any] = {"max_tokens": max_tokens}
        temp = getattr(params, "temperature", None)
        if temp is not None and temp >= 0:
            args["temperature"] = float(temp)
        top_p = getattr(params, "top_p", None)
        if top_p is not None and top_p >= 0:
            args["top_p"] = float(top_p)
        return args

    @staticmethod
    def _rollout_input(task: Task) -> dict[str, Any]:
        meta = task.meta
        return {
            "prompt": meta.get("prompt"),
            "answer": meta.get("answer"),
            "info": meta.get("info") or {},
            "example_id": meta.get("example_id", int(task.task_id)),
        }


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """verifiers may return pydantic AssistantMessage/ToolMessage objects."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if isinstance(msg, dict):
        return msg
    return {"role": "assistant", "content": str(msg)}


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif isinstance(usage, str):
        try:
            usage = json.loads(usage)
        except Exception:  # noqa: BLE001
            return {}
    elif not isinstance(usage, dict):
        usage = getattr(usage, "__dict__", None)
    return usage if isinstance(usage, dict) else {}
