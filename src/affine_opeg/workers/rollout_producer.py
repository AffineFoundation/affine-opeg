"""Rollout producer worker entrypoint.

Lifecycle:
    1. Load config and configure logging + DB event sink.
    2. Discover an active ``sampling_list`` via env var ``AFR_SAMPLING_LIST``
       (a single producer process is bound to one list).
    3. Compose adapters: DockerSandboxFactory + AffentAgentLoop +
       AffentNormalizer.
    4. Run ``producer_loop`` until SIGINT / SIGTERM. Heartbeat + metrics flush
       + event sink run as siblings.

No R2 writes from this process: rollouts persist inline in
``PG.rollouts.extra_compressed``. The publisher reads from PG and
materialises shards to R2 on its own cadence.
"""

from __future__ import annotations

import asyncio
import os
import signal

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg import SqlAlchemyMetadataStore
from affine_opeg.adapters.normalizers.registry import get_normalizer
from affine_opeg.adapters.sandboxes.affent_loop import AffentAgentLoop, AffentLoopConfig
from affine_opeg.adapters.sandboxes.docker_sandbox import DockerSandboxFactory
from affine_opeg.application.producer_loop import (
    ProducerConfig,
    ProducerDeps,
    run_producer_loop,
)
from affine_opeg.domain.ids import EnvName, SamplingListName, TeacherName
from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import get_sessionmaker
from affine_opeg.infrastructure.event_sink import QueuedDbEventSink
from affine_opeg.infrastructure.logging import configure_logging, get_logger, register_db_sink
from affine_opeg.infrastructure.metrics import metrics_flush_loop
from affine_opeg.workers.heartbeat import heartbeat_loop

log = get_logger("producer")


def _csv_env(name: str) -> list[str] | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


async def main() -> None:
    cfg = load_config()
    configure_logging(cfg, service="producer")
    sm = get_sessionmaker(cfg)
    sink = QueuedDbEventSink()
    register_db_sink(sink)

    list_name = os.environ.get("AFR_SAMPLING_LIST")
    if not list_name:
        log.error("producer.missing_sampling_list",
                  msg="set AFR_SAMPLING_LIST=<list_name> to bind this producer")
        raise SystemExit(2)

    metadata = SqlAlchemyMetadataStore(sm)
    sandbox_factory = DockerSandboxFactory(
        max_concurrent=cfg.rollout.max_concurrent_episodes,
    )
    agent_loop = AffentAgentLoop(AffentLoopConfig(
        max_turns=cfg.rollout.max_steps,
    ))
    normalizer = get_normalizer("affent")

    worker_id = f"producer-{os.environ.get('HOSTNAME', 'local')}-{os.getpid()}"
    deps = ProducerDeps(
        metadata=metadata,
        sandbox=sandbox_factory,
        normalizer=normalizer,
        agent_loop=agent_loop,
        producer_id=worker_id,
    )

    p_envs = _csv_env("AFR_ENV_NAMES")
    p_teachers = _csv_env("AFR_TEACHER_NAMES")
    producer_cfg = ProducerConfig(
        list_name=SamplingListName(list_name),
        env_names=[EnvName(e) for e in p_envs] if p_envs else None,
        teacher_names=[TeacherName(t) for t in p_teachers] if p_teachers else None,
        temperature_min=cfg.rollout.temperature_min,
        temperature_max=cfg.rollout.temperature_max,
        max_steps=cfg.rollout.max_steps,
        max_concurrent_episodes=cfg.rollout.max_concurrent_episodes,
        per_teacher_concurrency=cfg.rollout.per_teacher_concurrency,
    )

    stop = asyncio.Event()

    def _stop(*_a: object) -> None:
        log.info("producer.stop_signal")
        stop.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, _stop)

    aux = [
        asyncio.create_task(sink.run(sm), name="event_sink"),
        asyncio.create_task(metrics_flush_loop(sm), name="metrics"),
        asyncio.create_task(
            heartbeat_loop(sm, worker_id=worker_id, role="producer", version=cfg.version),
            name="heartbeat",
        ),
    ]

    log.info("producer.ready", worker_id=worker_id, list_name=list_name)
    try:
        await run_producer_loop(deps, producer_cfg, stop)
    finally:
        log.info("producer.stopping")
        for t in aux:
            t.cancel()
        await asyncio.gather(*aux, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
