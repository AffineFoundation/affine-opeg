"""High-level producer loop.

Orchestrates the choice of *what to sample next* against the open cells of
one ``sampling_list``. Composition:

    1. Poll ``sampling_progress`` for cells with ``collected < target``
       (atomic claim via FOR UPDATE SKIP LOCKED).
    2. For each claimed cell, draw ``sample_idx = collected`` (so re-claims
       on retry don't collide on the rollout business key), and call
       ``generate_rollout`` with a randomised temperature drawn from the
       configured range.
    3. On successful insert, increment ``collected``.
    4. Sleep briefly when no cells are open; respect ``stop`` signal between
       iterations so SIGTERM is honored promptly.

Concurrency is enforced both per-teacher (a Semaphore avoids hammering one
provider) and globally (Semaphore on docker sandboxes). The use case takes
those bounds via :class:`ProducerConfig`.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from affine_opeg.application.generate_rollouts import (
    AgentLoopFn,
    GenerateRolloutDeps,
    RolloutParams,
    generate_rollout,
)
from affine_opeg.domain.ids import (
    EnvName,
    SamplingListName,
    TaskId,
    TeacherName,
)
from affine_opeg.domain.models import SamplingProgress
from affine_opeg.domain.ports import (
    MetadataStore,
    SandboxFactory,
    TrajectoryNormalizer,
)
from affine_opeg.infrastructure.logging import get_logger, trace_context
from affine_opeg.infrastructure.metrics import metrics

log = get_logger("producer.loop")

# Backend family name for verifiers (container-free) envs.
VERIFIERS_BACKEND = "verifiers"


@dataclass(frozen=True)
class RolloutBackend:
    """A (sandbox, agent_loop, normalizer) triple plus its concurrency cap.

    One producer can host several backends and route each task to the right
    one by env (see :class:`BackendRouter`). The cap is enforced by a
    per-backend semaphore so the memory-heavy SWE sandbox backend stays
    within the host's RAM budget while cheap verifiers (NullSandbox) episodes
    run at a much higher cap — without either family stealing the other's
    slots.
    """

    name: str
    sandbox: SandboxFactory
    agent_loop: AgentLoopFn
    normalizer: TrajectoryNormalizer
    max_concurrency: int
    # Claim filter so this backend pulls only its own cells (set only when
    # multiple backends coexist). ``verifiers:%`` for the verifiers backend;
    # ``NOT LIKE verifiers:%`` for the default/SWE backend.
    claim_env_like: str | None = None
    claim_env_not_like: str | None = None


class BackendRouter:
    """Route a task's env to its rollout backend.

    ``verifiers:*`` envs go to the ``verifiers`` backend (when registered);
    everything else falls to ``default`` (the SWE/affent sandbox backend).
    """

    def __init__(self, backends: list[RolloutBackend], default: str) -> None:
        self._by_name = {b.name: b for b in backends}
        if default not in self._by_name:
            raise ValueError(f"default backend {default!r} not in {list(self._by_name)}")
        self._default = default

    def backend_for(self, env_name: EnvName) -> RolloutBackend:
        family = str(env_name).split(":", 1)[0]
        if family == VERIFIERS_BACKEND and VERIFIERS_BACKEND in self._by_name:
            return self._by_name[VERIFIERS_BACKEND]
        return self._by_name[self._default]

    @property
    def backends(self) -> list[RolloutBackend]:
        return list(self._by_name.values())

    @property
    def total_concurrency(self) -> int:
        return sum(b.max_concurrency for b in self._by_name.values())


@dataclass(frozen=True)
class ProducerConfig:
    list_name: SamplingListName
    env_names: list[EnvName] | None = None
    teacher_names: list[TeacherName] | None = None
    temperature_min: float = 1.0
    temperature_max: float = 1.8
    top_p: float = 0.95
    max_steps: int = 40
    batch_size: int = 8                       # cells claimed per poll
    poll_idle_sleep_s: float = 5.0
    max_concurrent_episodes: int = 16
    per_teacher_concurrency: int = 16
    seed: int | None = None


@dataclass
class ProducerDeps:
    metadata: MetadataStore
    router: BackendRouter
    producer_id: str = "producer-unknown"
    rng: random.Random = field(default_factory=random.Random)


async def run_producer_loop(
    deps: ProducerDeps,
    cfg: ProducerConfig,
    stop: asyncio.Event,
) -> None:
    """Long-running loop. Returns only when ``stop`` is set."""
    if cfg.seed is not None:
        deps.rng.seed(cfg.seed)
    # One semaphore per backend, sized to that backend's cap. Created here
    # (inside the running loop) rather than at wiring time. The overall
    # in-flight bound is the sum of the per-backend caps — claiming more
    # than that can't help since every episode must hold a backend slot.
    backend_sems = {
        b.name: asyncio.Semaphore(b.max_concurrency) for b in deps.router.backends
    }
    per_teacher_sems: dict[TeacherName, asyncio.Semaphore] = {}

    def _sem_for(t: TeacherName) -> asyncio.Semaphore:
        sem = per_teacher_sems.get(t)
        if sem is None:
            sem = asyncio.Semaphore(cfg.per_teacher_concurrency)
            per_teacher_sems[t] = sem
        return sem

    in_flight: set[asyncio.Task] = set()
    # Per-backend in-flight counts so we claim each backend's cells up to its
    # own cap, independently. Without this, the global ``collected ASC`` claim
    # order lets a large slow backend (SWE sandboxes) monopolise every batch
    # and starve the cheap verifiers backend even when it has free capacity.
    in_flight_by_backend: dict[str, int] = {b.name: 0 for b in deps.router.backends}

    def _dispatch(cell: SamplingProgress, backend: RolloutBackend) -> None:
        t = asyncio.create_task(
            _run_cell(deps, cfg, cell, backend,
                      backend_sems[backend.name], _sem_for(cell.teacher_name)),
            name=f"rollout:{backend.name}:{cell.task_id}:{cell.teacher_name}",
        )
        in_flight.add(t)
        in_flight_by_backend[backend.name] += 1

        def _done(task: asyncio.Task, _name: str = backend.name) -> None:
            in_flight.discard(task)
            in_flight_by_backend[_name] -= 1
            if task.exception() is not None:
                log.warning("producer.task_crashed", error=str(task.exception()))

        t.add_done_callback(_done)

    log.info("producer.loop.start", list_name=cfg.list_name, **_loop_meta(cfg))

    # Run the random-cell scheduler in the background. It tops up the
    # active cell pool periodically — independent of whether the
    # producer is busy claiming or idle — so the pool always grows when
    # ``target_active_cells`` allows. The producer itself only consumes;
    # the scheduler decides which (env, task, teacher) triples enter
    # sampling_progress next.
    scheduler_task = asyncio.create_task(
        _scheduler_background(deps, cfg, stop), name="cell-scheduler",
    )

    while not stop.is_set():
        claimed = 0
        any_spare = False
        # Claim per backend up to its own spare capacity, filtered to that
        # backend's env family — so neither family starves the other.
        for backend in deps.router.backends:
            spare = backend.max_concurrency - in_flight_by_backend[backend.name]
            if spare <= 0:
                continue
            any_spare = True
            cells = await _claim_cells(
                deps.metadata, cfg, backend=backend, batch_size=min(spare, cfg.batch_size),
            )
            for cell in cells:
                _dispatch(cell, backend)
            claimed += len(cells)

        if claimed == 0:
            if not any_spare and in_flight:
                # Every backend is saturated — wait for a slot to free up.
                await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            else:
                # Spare capacity but no eligible cells yet (pool still
                # filling) — idle briefly, honouring the stop signal.
                try:
                    await asyncio.wait_for(stop.wait(), timeout=cfg.poll_idle_sleep_s)
                except asyncio.TimeoutError:
                    pass

    if in_flight:
        log.info("producer.loop.drain", in_flight=len(in_flight))
        await asyncio.gather(*in_flight, return_exceptions=True)
    scheduler_task.cancel()
    try:
        await scheduler_task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
    log.info("producer.loop.stopped")


async def _scheduler_background(
    deps: ProducerDeps, cfg: ProducerConfig, stop: asyncio.Event,
) -> None:
    """Fire ``_maybe_top_up_pool`` every ``poll_idle_sleep_s`` regardless
    of producer business — that's what makes the active cell count
    actually converge on ``target_active_cells`` instead of being stuck
    at whatever the admin first INSERTed.
    """
    # Initial top-up so the first claim has something to do.
    await _maybe_top_up_pool(deps, cfg)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=cfg.poll_idle_sleep_s)
        except asyncio.TimeoutError:
            pass
        if stop.is_set():
            return
        try:
            await _maybe_top_up_pool(deps, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("scheduler.background_err", error=str(exc)[:200])


async def _maybe_top_up_pool(deps: ProducerDeps, cfg: ProducerConfig) -> None:
    """Materialise random cells until the active pool hits its target.

    Only applies when the list's config carries a ``pool`` block;
    list_name without one (legacy admin-seeded lists) is a no-op.
    """
    from affine_opeg.application.cell_scheduler import maintain_active_pool
    # The scheduler needs raw SQLAlchemy sessionmaker access (its
    # queries don't fit cleanly into the existing repository ports —
    # they touch both sampling_lists.config and sampling_progress in
    # one short transaction). Reach through the concrete adapter; if
    # someone wires in a non-SqlAlchemy MetadataStore the scheduler
    # is just skipped.
    sessionmaker = getattr(deps.metadata, "_sessionmaker", None)
    if sessionmaker is None:
        return
    try:
        result = await maintain_active_pool(sessionmaker, str(cfg.list_name))
    except LookupError:
        return
    except ValueError as exc:
        # No pool block in config -> nothing to schedule, just sample
        # whatever rows the admin pre-seeded.
        log.debug("producer.no_pool", reason=str(exc))
        return
    if result.inserted or result.pool_exhausted:
        log.info(
            "producer.pool.topped_up",
            inserted=result.inserted,
            skipped_conflicts=result.skipped_conflicts,
            active_after=result.active_after,
            pool_exhausted=result.pool_exhausted,
        )


async def _claim_cells(
    metadata: MetadataStore,
    cfg: ProducerConfig,
    *,
    backend: RolloutBackend | None = None,
    batch_size: int | None = None,
) -> list[SamplingProgress]:
    async with metadata.unit_of_work() as uow:
        cells = await uow.sampling_lists.claim_next_cell(
            cfg.list_name,
            env_names=cfg.env_names,
            teacher_names=cfg.teacher_names,
            batch_size=batch_size if batch_size is not None else cfg.batch_size,
            env_name_like=backend.claim_env_like if backend else None,
            env_name_not_like=backend.claim_env_not_like if backend else None,
        )
    metrics().incr("producer.cells_claimed", value=len(cells))
    return cells


async def _run_cell(
    deps: ProducerDeps,
    cfg: ProducerConfig,
    cell: SamplingProgress,
    backend: RolloutBackend,
    backend_sem: asyncio.Semaphore,
    teacher_sem: asyncio.Semaphore,
) -> None:
    """Produce one rollout for the slot ``claim_next_cell`` handed us.

    The cell already had its ``attempts`` counter bumped in the same
    transaction that returned this row, so ``sample_idx = attempts - 1``
    is uniquely ours for the duration of this attempt. Success bumps
    ``collected`` (inside ``generate_rollout`` together with the
    rollouts row insert); failure leaves ``collected`` alone so the cell
    can be re-claimed up to its attempt budget.
    """
    sample_idx = cell.attempts - 1
    params = RolloutParams(
        env_name=cell.env_name, task_id=cell.task_id,
        teacher_name=cell.teacher_name, sample_idx=sample_idx,
        temperature=deps.rng.uniform(cfg.temperature_min, cfg.temperature_max),
        top_p=cfg.top_p,
        seed=deps.rng.randint(0, 2**31 - 1),
        max_steps=cfg.max_steps,
        list_name=cell.list_name,
    )
    with trace_context(
        rollout_key=f"{cell.env_name}:{cell.task_id}:{cell.teacher_name}:{sample_idx}",
        list_name=str(cell.list_name),
    ):
        async with backend_sem, teacher_sem:
            try:
                gen_deps = GenerateRolloutDeps(
                    metadata=deps.metadata,
                    sandbox=backend.sandbox,
                    normalizer=backend.normalizer,
                    agent_loop=backend.agent_loop,
                )
                rollout = await generate_rollout(gen_deps, params, producer_id=deps.producer_id)
            except Exception as exc:  # noqa: BLE001
                log.error("producer.cell_failed",
                          error=str(exc),
                          env_name=cell.env_name, task_id=cell.task_id,
                          teacher_name=cell.teacher_name, sample_idx=sample_idx)
                metrics().incr("producer.cell_failed",
                               labels={"teacher": str(cell.teacher_name)})
                return

        # ``attempts`` was bumped at claim time (slot reservation).
        # ``collected`` is bumped inside ``generate_rollout`` when a
        # status='ok' rollout is persisted — same transaction as the
        # insert, so a crash between the two is impossible. Below is
        # only a metrics hop.
        if rollout.status.value == "ok":
            metrics().incr("producer.cell_completed",
                           labels={"teacher": str(cell.teacher_name)})


def _loop_meta(cfg: ProducerConfig) -> dict:
    return {
        "env_names": [str(e) for e in (cfg.env_names or [])] or None,
        "teacher_names": [str(t) for t in (cfg.teacher_names or [])] or None,
        "batch_size": cfg.batch_size,
        "global_concurrency": cfg.max_concurrent_episodes,
        "per_teacher_concurrency": cfg.per_teacher_concurrency,
    }
