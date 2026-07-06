"""Shared verifiers (PrimeIntellect) environment runtime.

A verifiers ``Environment`` bundles dataset + rollout + rubric behind one
uniform interface, so a single adapter family unlocks every env on the PI
Environments Hub. Both the task source (needs the dataset) and the agent
loop (needs ``rollout`` + ``rubric``) resolve the *same* env object through
:func:`load_verifiers_env`, which caches per ``env_id`` so a producer doesn't
rebuild the dataset on every rollout.

``env_name`` convention: ``verifiers:<env_id>`` (e.g. ``verifiers:gsm8k``).
``<env_id>`` is what gets handed to :func:`vf.load_environment`.

Most hub envs install as a package (``vf-install <env_id>``) and load via
``vf.load_environment``. A small set of *builtin* builders is kept for envs
we want runnable without a hub package — currently just ``gsm8k`` (built from
the bundled example dataset), which is our smoke target.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import verifiers as vf

from affine_opeg.domain.ids import EnvName
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("verifiers_runtime")

# env_id -> builder(**env_args) -> verifiers.Environment
_BUILTIN_BUILDERS: dict[str, Callable[..., Any]] = {}

# Resolved env cache. Keyed by (env_id, frozen-args) so a producer reuses the
# same dataset/rubric across thousands of rollouts. verifiers envs are
# stateless across ``rollout`` calls (state lives in the returned dict), so
# sharing one instance is safe.
_ENV_CACHE: dict[tuple, Any] = {}
_ENV_LOCK = threading.Lock()


def env_id_of(env_name: EnvName | str) -> str:
    """``verifiers:gsm8k`` -> ``gsm8k``. Bare names pass through unchanged."""
    s = str(env_name)
    return s.split(":", 1)[1] if ":" in s else s


def register_builtin(env_id: str, builder: Callable[..., Any]) -> None:
    """Register an in-process env builder (no hub package required)."""
    _BUILTIN_BUILDERS[env_id] = builder


def load_verifiers_env(env_id: str, **env_args: Any) -> Any:
    """Resolve (and cache) a verifiers ``Environment`` for ``env_id``.

    Builtins win over the hub so our smoke target is reproducible offline;
    everything else routes through ``vf.load_environment``.
    """
    key = (env_id, tuple(sorted(env_args.items())))
    with _ENV_LOCK:
        cached = _ENV_CACHE.get(key)
        if cached is not None:
            return cached

    # Hub-first: a real installed env package (e.g. the full 7473-row gsm8k,
    # math500, mmlu-pro, …) always wins over a builtin. Builtins exist only
    # as an offline fallback for envs not installed from the hub — currently
    # just the truncated gsm8k smoke scaffold.
    builder = _BUILTIN_BUILDERS.get(env_id)
    try:
        log.info("verifiers.env.load_hub", env_id=env_id)
        env = vf.load_environment(env_id, **env_args)
    except Exception as exc:  # noqa: BLE001 — hub package may be absent
        if builder is None:
            raise
        log.info("verifiers.env.build_builtin", env_id=env_id, hub_error=str(exc)[:120])
        env = builder(**env_args)

    with _ENV_LOCK:
        _ENV_CACHE[key] = env
    return env


# --------------------------------------------------------------------------- #
# Builtin builders
# --------------------------------------------------------------------------- #


def _build_gsm8k(n: int = 200, split: str = "test") -> Any:
    """A real gsm8k ``SingleTurnEnv`` from the bundled example dataset.

    Same shape as the PI hub env: real dataset rows, real LLM call at rollout
    time, real 0/1 boxed-answer correctness rubric. Mirrors the long-standing
    ``scripts/smoke_pi_rollout.py`` scaffold.
    """
    from verifiers.envs.singleturn_env import SingleTurnEnv

    ds = vf.load_example_dataset("gsm8k", split=split, n=n)
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


register_builtin("gsm8k", _build_gsm8k)


def _build_phybench(use_think: bool = False) -> Any:
    """PHYBench physics-reasoning env with a continuous EED reward.

    The PI hub ``phybench`` wheel is broken — it ships only ``phybench.py``,
    whose ``from phybench.eed import EED`` refers to a module that was never
    packaged, so ``vf.load_environment('phybench')`` fails. We reconstruct the
    exact env here: the ungated ``Eureka-Lab/PHYBench`` dataset (1000 physics
    problems, LaTeX answers) + a vendored EED (Expression Edit Distance)
    scorer. EED is continuous in [0,1], which reliably yields within-cell
    reward variance (frontier models score mid-range — SOTA ~37% exact).

    Registered as a builtin so ``load_verifiers_env('phybench')`` falls back
    here after the hub load raises.

    ``use_think=False`` on purpose: the plain boxed ``Parser`` extracts the
    ``\\boxed{}`` answer whether or not the model emits ``<think>`` tags, so it
    is robust across our mixed teacher fleet. ``ThinkParser`` returns nothing
    (reward 0) for teachers whose chat template strips think tags (Qwen3,
    DeepSeek-R1 style) — verified: plain boxed completion scores 0 under
    ThinkParser but 1.0 under Parser.
    """
    from datasets import load_dataset
    from verifiers.envs.singleturn_env import SingleTurnEnv
    from verifiers.utils.data_utils import (
        BOXED_SYSTEM_PROMPT,
        THINK_BOXED_SYSTEM_PROMPT,
        extract_boxed_answer,
    )

    from affine_opeg.adapters.envs_builtin.phybench_eed import EED

    dataset = load_dataset("Eureka-Lab/PHYBench", split="train")
    dataset = dataset.filter(lambda x: x["answer"] != "")
    dataset = dataset.rename_column("content", "question")
    split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]

    if use_think:
        system_prompt = THINK_BOXED_SYSTEM_PROMPT
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        system_prompt = BOXED_SYSTEM_PROMPT
        parser = vf.Parser(extract_fn=extract_boxed_answer)

    def eed_reward_func(completion, answer, **kwargs):  # noqa: ANN001
        response = parser.parse_answer(completion) or ""
        score, _rel, _tree, _dist = EED(answer, response)
        return score / 100.0  # EED returns 0-100

    def accuracy_reward_func(completion, answer, **kwargs):  # noqa: ANN001
        response = parser.parse_answer(completion) or ""
        if "$$" in response:
            response = response.split("$$")[-1].strip()
        return float(int(response == answer))

    rubric = vf.Rubric(
        funcs=[eed_reward_func, accuracy_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.5, 0.2],
    )
    return SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )


register_builtin("phybench", _build_phybench)
