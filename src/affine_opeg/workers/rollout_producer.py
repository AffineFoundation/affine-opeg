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
    BackendRouter,
    ProducerConfig,
    ProducerDeps,
    RolloutBackend,
    VERIFIERS_BACKEND,
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


def _build_verifiers_backend(max_concurrency: int, *, claim_env_like: str | None = None):
    """Construct the verifiers backend, importing its adapters lazily.

    The verifiers adapters import the ``verifiers`` package; a SWE-only
    producer container may not have it installed. Importing here (only when
    the backend is actually enabled) keeps that container booting.
    """
    from affine_opeg.adapters.sandboxes.null_sandbox import NullSandboxFactory
    from affine_opeg.adapters.sandboxes.verifiers_loop import (
        VerifiersAgentLoop,
        VerifiersLoopConfig,
    )

    return RolloutBackend(
        name=VERIFIERS_BACKEND,
        sandbox=NullSandboxFactory(max_concurrent=max_concurrency),
        agent_loop=VerifiersAgentLoop(VerifiersLoopConfig()),
        normalizer=get_normalizer("verifiers"),
        max_concurrency=max_concurrency,
        claim_env_like=claim_env_like,
    )


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

    # One producer can host two rollout backends and route each task by env:
    #   - ``affent`` (default): docker sandbox + affent agent + testbed grade,
    #     for SWE-style envs (memory-heavy — capped by max_concurrent_episodes).
    #   - ``verifiers``: container-free PI pathway (NullSandbox + verifiers
    #     env owns rollout *and* rubric). Cheap on local memory — its own
    #     (usually higher) cap is ``rollout.verifiers_concurrency``.
    # ``AFR_ROLLOUT_MODE=verifiers`` forces a verifiers-only producer (no SWE
    # backend); otherwise SWE is always present and verifiers is added when
    # ``verifiers_concurrency > 0``. Mixing lets a flood of cheap verifiers
    # episodes run alongside a memory-safe number of SWE sandboxes.
    mode = os.environ.get("AFR_ROLLOUT_MODE", "affent").strip().lower()
    backends: list[RolloutBackend] = []
    if mode == "verifiers":
        backends.append(_build_verifiers_backend(cfg.rollout.max_concurrent_episodes))
        default_backend = VERIFIERS_BACKEND
    elif mode == "affent":
        # Mixed when verifiers is also enabled: each backend claims only its
        # own family (SWE = NOT verifiers:%, verifiers = verifiers:%) so the
        # slow sandbox backlog can't starve verifiers in the shared claim order.
        mixed = cfg.rollout.verifiers_concurrency > 0
        backends.append(RolloutBackend(
            name="affent",
            sandbox=DockerSandboxFactory(max_concurrent=cfg.rollout.max_concurrent_episodes),
            agent_loop=AffentAgentLoop(AffentLoopConfig(max_turns=cfg.rollout.max_steps)),
            normalizer=get_normalizer("affent"),
            max_concurrency=cfg.rollout.max_concurrent_episodes,
            claim_env_not_like="verifiers:%" if mixed else None,
        ))
        default_backend = "affent"
        if mixed:
            backends.append(_build_verifiers_backend(
                cfg.rollout.verifiers_concurrency, claim_env_like="verifiers:%"))
    else:
        log.error("producer.bad_rollout_mode", mode=mode,
                  msg="AFR_ROLLOUT_MODE must be 'affent' or 'verifiers'")
        raise SystemExit(2)

    router = BackendRouter(backends, default=default_backend)
    log.info("producer.rollout_backends",
             backends={b.name: b.max_concurrency for b in backends},
             default=default_backend)

    worker_id = f"producer-{os.environ.get('HOSTNAME', 'local')}-{os.getpid()}"
    deps = ProducerDeps(
        metadata=metadata,
        router=router,
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
