"""Layer-2 smoke: one real PI (verifiers) rollout, scored, normalized.

Bypasses Postgres and the producer loop entirely. It exercises the exact
core that VerifiersAgentLoop + VerifiersNormalizer will run in production:

    load real env -> build RolloutInput -> rollout (real LLM) ->
    score_rollout (real rubric) -> map completion to NormalizedMessage.

Endpoint is parameterized via env vars so the same script runs dry (no
creds -> prints the scaffold and exits) or live (creds -> real rollout):

    CHUTES_BASE_URL   e.g. https://<slug>.chutes.ai/v1  or  https://llm.chutes.ai/v1
    CHUTES_API_KEY    bearer key
    CHUTES_MODEL      served model id

Run:
    . .venv/bin/activate
    CHUTES_BASE_URL=... CHUTES_API_KEY=... CHUTES_MODEL=... \
        python scripts/smoke_pi_rollout.py
"""

from __future__ import annotations

import asyncio
import json
import os

import verifiers as vf
from verifiers.envs.singleturn_env import SingleTurnEnv

from affine_opeg.domain.models import NormalizedMessage, ToolCall


def build_gsm8k_env() -> SingleTurnEnv:
    """A real gsm8k SingleTurnEnv (same shape as the PI hub env): real
    dataset rows, real LLM call at rollout time, real 0/1 correctness rubric."""
    ds = vf.load_example_dataset("gsm8k", split="test", n=5)
    parser = vf.Parser(extract_fn=vf.extract_boxed_answer)

    def correct_answer(parser, completion, answer, **_):  # noqa: ANN001
        return 1.0 if parser.parse_answer(completion) == answer else 0.0

    rubric = vf.Rubric(funcs=[correct_answer], parser=parser)
    return SingleTurnEnv(
        dataset=ds,
        system_prompt="Solve the problem. Put the final answer in \\boxed{}.",
        parser=parser,
        rubric=rubric,
    )


def rollout_input_from_row(row: dict) -> dict:
    """Exactly what VerifiersAgentLoop reconstructs from task.meta."""
    return {
        "prompt": row["prompt"],
        "example_id": row.get("example_id", 0),
        "answer": row["answer"],
        "info": row.get("info", {}),
    }


def _to_dict(msg):  # noqa: ANN001
    """verifiers may return pydantic AssistantMessage/ToolMessage objects."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return msg


def normalize_completion(completion) -> list[NormalizedMessage]:
    """Minimal VerifiersNormalizer: OpenAI chat messages -> NormalizedMessage."""
    out: list[NormalizedMessage] = []
    for msg in completion or []:
        msg = _to_dict(msg)
        role = msg.get("role")
        if role == "tool":
            out.append(NormalizedMessage(
                role="tool", content=str(msg.get("content", "")),
                tool_call_id=msg.get("tool_call_id", "")))
        else:
            tcs = [
                ToolCall(id=tc.get("id", ""),
                         name=tc.get("function", {}).get("name", ""),
                         arguments=_safe_json(tc.get("function", {}).get("arguments")))
                for tc in (msg.get("tool_calls") or [])
            ]
            out.append(NormalizedMessage(
                role=role, content=msg.get("content"), tool_calls=tcs))
    return out


def _safe_json(s):  # noqa: ANN001
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s) if s else {}
    except Exception:  # noqa: BLE001
        return {"_raw": s}


async def main() -> None:
    env = build_gsm8k_env()
    row = env.get_dataset()[0]
    ri = rollout_input_from_row(row)

    print("=" * 70)
    print("ENV READY:", type(env).__name__, "| dataset cols:", env.get_dataset().column_names)
    print("PROMPT:", json.dumps(ri["prompt"], ensure_ascii=False, indent=2))
    print("GOLD ANSWER:", ri["answer"])
    print("=" * 70)

    base_url = os.environ.get("CHUTES_BASE_URL")
    api_key = os.environ.get("CHUTES_API_KEY")
    model = os.environ.get("CHUTES_MODEL")
    if not (base_url and api_key and model):
        print("\n[dry-run] CHUTES_BASE_URL / CHUTES_API_KEY / CHUTES_MODEL not set.")
        print("Scaffold verified: env + RolloutInput ready. Set the 3 vars to run live.")
        return

    # verifiers 0.1.14 uses its own client abstraction. api_key_var is the
    # ENV VAR NAME it reads the key from (not the key itself).
    client = vf.ClientConfig(api_base_url=base_url, api_key_var="CHUTES_API_KEY")
    sampling_args = {"temperature": 1.0, "max_tokens": 2048}

    state = await env.rollout(ri, client, model, sampling_args)
    await env.rubric.score_rollout(state)   # writes reward/metrics into state in place

    completion = [_to_dict(m) for m in (state["completion"] or [])]
    reward = state.get("reward")
    metrics = state.get("metrics")
    usage = state.get("usage")
    norm = normalize_completion(completion)

    print("\n----- LIVE ROLLOUT RESULT -----")
    print("COMPLETION:", json.dumps(completion, ensure_ascii=False, indent=2)[:4000])
    print("REWARD:", reward)
    print("METRICS:", metrics)
    print("USAGE:", usage)
    print("NORMALIZED MESSAGES:",
          json.dumps([m.model_dump() for m in norm], ensure_ascii=False, indent=2)[:2000])
    print("\nREWARD_BREAKDOWN (would persist):",
          json.dumps({"score": float(reward) if reward is not None else None,
                      **(metrics or {})}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
