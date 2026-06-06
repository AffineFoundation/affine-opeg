"""One-shot bootstrap for the verifiers:gsm8k PG smoke.

Registers the env + loads tasks (from the verifiers dataset), upserts a chutes
teacher, and seeds a small sampling_list. Run once against a migrated DB, then
start the producer with AFR_ROLLOUT_MODE=verifiers AFR_SAMPLING_LIST=gsm8k-smoke.
"""

from __future__ import annotations

import asyncio

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg import SqlAlchemyMetadataStore
from affine_opeg.adapters.task_sources.verifiers import VerifiersTaskSource
from affine_opeg.application.bootstrap import init_sampling_list, load_env_and_tasks
from affine_opeg.domain.ids import EnvName, SamplingListName, TeacherName
from affine_opeg.domain.models import Teacher
from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import get_sessionmaker

ENV_NAME = EnvName("verifiers:gsm8k")
TEACHER_NAME = TeacherName("chutes-qwen3-32b")
LIST_NAME = SamplingListName("gsm8k-smoke")
N_TASKS = 2            # task_id 0..1
TARGET_PER_CELL = 1    # 1 sample per (task, teacher) cell -> 2 rollouts total


async def main() -> None:
    cfg = load_config()
    sm = get_sessionmaker(cfg)
    metadata = SqlAlchemyMetadataStore(sm)

    # 1) env + tasks
    ts = VerifiersTaskSource(ENV_NAME)
    env_rows, inserted = await load_env_and_tasks(
        metadata,
        env_name=ENV_NAME,
        dataset="gsm8k",
        dataset_version="example",
        task_source=ts,
        actor="bootstrap",
    )
    print(f"env upserted={env_rows}  tasks inserted={inserted}")

    # 2) teacher (chutes, OpenAI-chat compatible; verifiers reads the key
    #    from the CHUTES_API_KEY env var named below)
    teacher = Teacher(
        teacher_name=TEACHER_NAME,
        model_family="qwen",
        provider="chutes",
        endpoint="https://llm.chutes.ai/v1",
        api_key_env="CHUTES_API_KEY",
        tool_format="openai_json",
        reasoning_format="thinking_tag",
        context_window=32768,
        meta={"served_model": "Qwen/Qwen3-32B-TEE"},
    )
    async with metadata.unit_of_work() as uow:
        await uow.teachers.upsert(teacher)
        await uow.commit()
    print(f"teacher upserted: {TEACHER_NAME}")

    # 3) sampling list (small)
    cells = await init_sampling_list(
        metadata,
        list_name=LIST_NAME,
        env_names=[ENV_NAME],
        teacher_names=[TEACHER_NAME],
        target_samples_per_cell=TARGET_PER_CELL,
        task_id_filter=(0, N_TASKS),
        description="verifiers gsm8k PG smoke",
        actor="bootstrap",
    )
    print(f"sampling_list {LIST_NAME} created with {cells} cells")


if __name__ == "__main__":
    asyncio.run(main())
