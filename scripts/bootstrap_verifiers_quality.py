"""Bootstrap the 5 quality verifiers (PI hub) envs into PG + a cross-env list.

Registers each env + bulk-inserts its tasks, ensures the chutes teacher exists,
then seeds one small cross-env sampling_list (1 task/env, 1 sample/cell) so a
producer run proves multi-env generation end to end.

Run against a migrated DB:
    AFR_DB__... python scripts/bootstrap_verifiers_quality.py
"""

from __future__ import annotations

import asyncio

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg import SqlAlchemyMetadataStore
from affine_opeg.adapters.task_sources.verifiers import VerifiersTaskSource
from affine_opeg.application.bootstrap import init_sampling_list, load_env_and_tasks
from affine_opeg.domain.ids import EnvName, SamplingListName, TaskId, TeacherName
from affine_opeg.domain.models import Teacher
from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import get_sessionmaker

# PI hub, pure-LLM, individual-level rubric (score_rollout-compatible).
# NOTE: aime2024 is excluded — its rubric is group-level only, so it needs
# the group-scoring path (env.run_group / score_rollouts), a future addition.
QUALITY_ENVS = ["gsm8k", "math500", "mmlu-pro", "hendrycks-math", "ifeval"]
TEACHER_NAME = TeacherName("chutes-qwen3-32b")
LIST_NAME = SamplingListName("quality-smoke")


async def main() -> None:
    cfg = load_config()
    sm = get_sessionmaker(cfg)
    md = SqlAlchemyMetadataStore(sm)

    for env_id in QUALITY_ENVS:
        env_name = EnvName(f"verifiers:{env_id}")
        ts = VerifiersTaskSource(env_name)
        _, inserted = await load_env_and_tasks(
            md, env_name=env_name, dataset=env_id, dataset_version="hub",
            task_source=ts, actor="bootstrap",
        )
        total = await ts.task_count()
        print(f"{env_id:10s} tasks_total={total:6d} inserted={inserted}")

    teacher = Teacher(
        teacher_name=TEACHER_NAME, model_family="qwen", provider="chutes",
        endpoint="https://llm.chutes.ai/v1", api_key_env="CHUTES_API_KEY",
        tool_format="openai_json", reasoning_format="thinking_tag",
        context_window=32768, meta={"served_model": "Qwen/Qwen3-32B-TEE"},
    )
    async with md.unit_of_work() as uow:
        await uow.teachers.upsert(teacher)
        await uow.commit()
    print(f"teacher upserted: {TEACHER_NAME}")

    # one single list spanning all 5 envs (task 0 each, 1 sample) -> 5 cells,
    # so a single producer bound to this list generates one rollout per env.
    cells = await init_sampling_list(
        md, list_name=LIST_NAME,
        env_names=[EnvName(f"verifiers:{e}") for e in QUALITY_ENVS],
        teacher_names=[TEACHER_NAME],
        target_samples_per_cell=1, task_id_filter=(0, 1),
        description="cross-env quality smoke", actor="bootstrap",
    )
    print(f"seeded list {LIST_NAME} with {cells} cells across {len(QUALITY_ENVS)} envs")


if __name__ == "__main__":
    asyncio.run(main())
