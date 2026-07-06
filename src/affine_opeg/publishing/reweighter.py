"""Yield-driven pool reweighter — auto-cut dud envs, weight survivors by yield.

Recomputes ``sampling_lists.config.pool.env_weights`` from the rolling
per-env *yield* (fraction of cells whose within-cell reward carries
variance, i.e. that can become a publishable task). The scheduler
(``cell_scheduler.maintain_active_pool``) turns those weights into per-env
active-cell targets, so this is the knob that decides how the producer's
compute is split across envs.

This module is called two ways:
  * inline from ``publisher_loop`` on a throttle (no cron, ships in the
    image — see ``AFR_REWEIGHT_*`` env vars), and
  * from the ``scripts/reweight_pool.py`` CLI for manual dry-runs.

Policy (per env, over the last ``window_hours``)
------------------------------------------------
  * ``yield``   = (# cells with reward std >= min_std) / (# cells with
                  >= 2 ok rollouts). A cell is (env, task, teacher).
  * **Warm-up guard.** If the env has < ``min_cells`` cells it is NOT cut;
    it gets a neutral warm-up score so it keeps sampling. Stops an env
    that got unlucky early (or a freshly integrated one) from being zeroed
    forever with no chance to recover.
  * **Auto-cut.** Else if ``yield < cut_yield`` the score is 0.
  * **Aggressive weighting.** Else score = ``yield ** alpha``.

Final weight = ``floor + score``, then raised to ``pins[env]`` if pinned
(a per-env raw-weight floor, e.g. keep swe-rebench sampled despite low
yield). The scheduler normalises weights to shares, so absolute scale is
free.
"""

from __future__ import annotations

import json
import os

from sqlalchemy import text

# Cells with >=2 ok rollouts and reward std below this are "degenerate"
# (no distillation signal). Mirrors AFR_PUBLISH_MIN_REWARD_STD default.
DEFAULT_MIN_STD = 0.05


def compute_weights(
    envs: list[str],
    stats: dict[str, tuple[float, int]],
    *,
    cut_yield: float,
    alpha: float,
    min_cells: int,
    floor: float,
    pins: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Return (env_weights, per-env status label). Pure — unit-testable.

    ``stats`` maps env -> (yield, n_cells). ``pins`` maps env -> minimum
    raw weight (applied after the yield formula).
    """
    pins = pins or {}
    # Warm-up score = median yield of the envs that DO have enough data,
    # so a data-poor env samples at a typical rate (not a dud rate).
    mature_yields = sorted(y for e, (y, n) in stats.items() if n >= min_cells)
    if mature_yields:
        mid = len(mature_yields) // 2
        warmup_yield = (
            mature_yields[mid]
            if len(mature_yields) % 2
            else (mature_yields[mid - 1] + mature_yields[mid]) / 2
        )
    else:
        warmup_yield = cut_yield  # no data anywhere -> sample everything equally-ish

    weights: dict[str, float] = {}
    labels: dict[str, str] = {}
    for e in envs:
        y, n = stats.get(e, (0.0, 0))
        if n < min_cells:
            score = max(warmup_yield, cut_yield) ** alpha
            label = f"warmup(n={n}<{min_cells})"
        elif y < cut_yield:
            score = 0.0
            label = f"CUT(yield={y:.0%})"
        else:
            score = y ** alpha
            label = f"yield={y:.0%}"
        w = floor + score
        pin = pins.get(e)
        if pin is not None and pin > w:
            w = pin
            label += f" pinned>={pin:g}"
        weights[e] = round(w, 6)
        labels[e] = label
    return weights, labels


async def _env_yield(
    session, envs: list[str], window_hours: int, min_std: float
) -> dict[str, tuple[float, int]]:
    """Return {env: (yield, n_cells)} over the rolling window (SQLAlchemy session)."""
    rows = (await session.execute(
        text("""
        with cell as (
          select env_name, task_id, teacher_name,
                 count(*) n, stddev_pop(reward) sd
          from rollouts
          where status = 'ok' and reward is not null
            and created_at > now() - make_interval(hours => :wh)
            and env_name = any(:envs)
          group by 1, 2, 3
          having count(*) >= 2
        )
        select env_name,
               count(*) n_cells,
               count(*) filter (where sd >= :mstd) var_ok
        from cell group by 1
        """),
        {"wh": window_hours, "envs": envs, "mstd": min_std},
    )).all()
    out: dict[str, tuple[float, int]] = {}
    for r in rows:
        n = int(r.n_cells)
        y = (int(r.var_ok) / n) if n else 0.0
        out[r.env_name] = (y, n)
    return out


async def reweight_pool(
    sm, list_name: str, *,
    window_hours: int = 48,
    cut_yield: float = 0.10,
    alpha: float = 2.0,
    min_cells: int = 30,
    min_std: float = DEFAULT_MIN_STD,
    floor: float = 0.03,
    pins: dict[str, float] | None = None,
    apply: bool = True,
) -> tuple[dict[str, float], dict[str, str]]:
    """Compute (and optionally write) ``config.pool.env_weights`` for a list.

    Returns (weights, labels). ``sm`` is an async sessionmaker.
    """
    async with sm() as session:
        row = (await session.execute(
            text("select config from sampling_lists where list_name = :ln"),
            {"ln": list_name},
        )).first()
        if row is None:
            raise LookupError(f"no sampling_lists row: {list_name}")
        config = row[0]
        if isinstance(config, str):
            config = json.loads(config)
        pool = config.get("pool") or {}
        envs = list(dict.fromkeys(pool.get("envs") or []))
        if not envs:
            raise ValueError("pool.envs empty")

        stats = await _env_yield(session, envs, window_hours, min_std)
        weights, labels = compute_weights(
            envs, stats,
            cut_yield=cut_yield, alpha=alpha,
            min_cells=min_cells, floor=floor, pins=pins,
        )
        if apply:
            pool["env_weights"] = weights
            config["pool"] = pool
            await session.execute(
                text("update sampling_lists set config = :c where list_name = :ln"),
                {"c": json.dumps(config), "ln": list_name},
            )
            await session.commit()
    return weights, labels


def pins_from_env() -> dict[str, float]:
    """Parse ``AFR_REWEIGHT_PINS`` = ``env=weight,env2=weight2`` (raw-weight floors).

    Default keeps swe-rebench sampled (a strategically useful coding signal)
    despite a low measured yield. Set ``AFR_REWEIGHT_PINS=`` (empty) to drop
    all pins.
    """
    raw = os.environ.get("AFR_REWEIGHT_PINS")
    if raw is None:
        return {"swe-rebench": 0.10}  # ~8-9% share; overridable
    pins: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            pins[k.strip()] = float(v)
        except ValueError:
            continue
    return pins


def params_from_env() -> dict:
    """Reweighter knobs from ``AFR_REWEIGHT_*`` env vars."""
    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, "").strip() or default)
        except ValueError:
            return default

    def _i(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, "").strip() or default)
        except ValueError:
            return default

    return dict(
        list_name=os.environ.get("AFR_REWEIGHT_LIST_NAME", "").strip() or "smoke-01",
        window_hours=_i("AFR_REWEIGHT_WINDOW_H", 48),
        cut_yield=_f("AFR_REWEIGHT_CUT_YIELD", 0.10),
        alpha=_f("AFR_REWEIGHT_ALPHA", 2.0),
        min_cells=_i("AFR_REWEIGHT_MIN_CELLS", 30),
        floor=_f("AFR_REWEIGHT_FLOOR", 0.03),
        interval_s=_f("AFR_REWEIGHT_INTERVAL_S", 7200.0),
        enabled=(os.environ.get("AFR_REWEIGHT_ENABLED", "1").strip() not in ("0", "false", "no", "")),
    )
