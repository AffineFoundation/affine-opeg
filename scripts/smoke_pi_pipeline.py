"""Layer-3 smoke: the FULL verifiers pathway through the real opeg adapters.

Unlike ``smoke_pi_rollout.py`` (which hand-builds the env), this drives the
exact production adapters that the producer wires in ``AFR_ROLLOUT_MODE=
verifiers`` — only Postgres is bypassed:

    VerifiersTaskSource  -> Task (prompt/answer/info in meta)
    NullSandboxFactory   -> NullSandbox(task)
    VerifiersAgentLoop   -> RawTrajectory (real LLM rollout + rubric score)
    VerifiersNormalizer  -> NormalizedTrajectory
    compress_json        -> the exact bytes the publisher would persist

Run dry (prints the task scaffold and exits) or live (real rollout):

    . .venv/bin/activate
    CHUTES_BASE_URL=https://llm.chutes.ai/v1 \
    CHUTES_API_KEY=cpk_... \
    CHUTES_MODEL=<served-model-id> \
        python scripts/smoke_pi_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import os

from affine_opeg.adapters.normalizers.registry import get_normalizer
from affine_opeg.adapters.sandboxes.null_sandbox import NullSandboxFactory
from affine_opeg.adapters.sandboxes.verifiers_loop import (
    VerifiersAgentLoop,
    VerifiersLoopConfig,
)
from affine_opeg.adapters.task_sources.verifiers import VerifiersTaskSource
from affine_opeg.domain.ids import EnvName, TaskId, TeacherName
from affine_opeg.domain.models import Teacher
from affine_opeg.infrastructure.compression import compress_json

ENV_NAME = EnvName("verifiers:gsm8k")


def _teacher(endpoint: str, model: str) -> Teacher:
    return Teacher(
        teacher_name=TeacherName("smoke-teacher"),
        model_family="qwen",
        provider="chutes",
        endpoint=endpoint,
        api_key_env="CHUTES_API_KEY",
        tool_format="openai_json",
        reasoning_format="none",
        context_window=32768,
        meta={"served_model": model},
    )


class _Params:
    """Minimal RolloutParams stand-in (the loop only reads these)."""

    temperature = 0.7
    top_p = 0.95
    seed = 0
    max_steps = 1


async def main() -> None:
    ts = VerifiersTaskSource(ENV_NAME)
    count = await ts.task_count()
    task = await ts.load_task(TaskId(0))

    print("=" * 72)
    print(f"TASK SOURCE: {ENV_NAME}  | tasks={count}")
    print(f"TASK 0  env_id={task.meta['verifiers_env_id']}  example_id={task.meta.get('example_id')}")
    print("-" * 72)
    print("PROMPT (messages):")
    print(json.dumps(task.meta["prompt"], ensure_ascii=False, indent=2))
    print("GOLD ANSWER:", task.meta.get("answer"))
    print("=" * 72)

    base_url = os.environ.get("CHUTES_BASE_URL")
    api_key = os.environ.get("CHUTES_API_KEY")
    model = os.environ.get("CHUTES_MODEL")
    if not (base_url and api_key and model):
        print("\n[dry-run] CHUTES_BASE_URL / CHUTES_API_KEY / CHUTES_MODEL not set.")
        print("Adapter chain verified up to rollout. Set the 3 vars to run live.")
        return

    teacher = _teacher(base_url, model)
    loop = VerifiersAgentLoop(VerifiersLoopConfig(max_tokens=2048))
    normalizer = get_normalizer("verifiers")

    async with NullSandboxFactory().acquire(task) as sb:
        raw = await loop(teacher, sb, _Params())

    norm = normalizer.normalize(raw)
    blob, sha = compress_json(norm.model_dump(mode="json"))

    print("\n----- RAW TRAJECTORY (family={}) -----".format(raw.family))
    print("REWARD_BREAKDOWN:", json.dumps(raw.reward_breakdown, ensure_ascii=False))
    print("AGENT_META:", json.dumps(raw.agent_meta, ensure_ascii=False))
    print("STEPS (raw chat messages):", len(raw.steps))

    print("\n----- NORMALIZED TRAJECTORY -----")
    print("schema_version:", norm.schema_version)
    for i, m in enumerate(norm.messages):
        body = m.content if m.content is not None else ""
        tc = f"  +{len(m.tool_calls)} tool_calls" if m.tool_calls else ""
        print(f"  [{i}] {m.role:9s} {str(body)[:200]!r}{tc}")
    print("reward_breakdown:", json.dumps(norm.reward_breakdown, ensure_ascii=False))

    print("\n----- PERSIST PAYLOAD (what the publisher stores) -----")
    print(f"compressed bytes: {len(blob)}   sha256: {sha[:16]}…")
    print("reward:", norm.reward_breakdown.get("score"))


if __name__ == "__main__":
    asyncio.run(main())
