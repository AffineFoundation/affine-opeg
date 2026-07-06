"""Yield-driven pool reweighter — auto-cut dud envs, weight survivors by yield.

Recomputes ``sampling_lists.config.pool.env_weights`` from the rolling
per-env *yield* (fraction of cells whose within-cell reward carries
variance, i.e. that can become a publishable task). The scheduler
(``cell_scheduler.maintain_active_pool``) turns those weights into per-env
active-cell targets, so this is the knob that decides how the producer's
compute is split across envs.

Policy
------
For each env in the pool, over the last ``--window-hours``:

  * ``yield``   = (# cells with reward std >= --min-std) / (# cells with
                  >= 2 ok rollouts). A "cell" is (env, task, teacher).
  * ``n_cells`` = # cells with >= 2 ok rollouts (the denominator).

Then:
  * **Warm-up guard (cold start).** If ``n_cells < --min-cells`` the env
    hasn't been sampled enough to judge — it is NOT cut. It gets a neutral
    warm-up score so it keeps sampling until it has data. This is what
    stops an env that got unlucky early (or a freshly integrated env) from
    being zeroed forever with no chance to recover.
  * **Auto-cut.** Else if ``yield < --cut-yield`` the env's score is 0.
  * **Aggressive weighting.** Else score = ``yield ** --alpha`` (alpha>1
    concentrates budget on the highest-yield envs).

Final stored weight = ``--floor + score``. The floor is a small constant
so even a cut env keeps a non-zero recovery probe (share ~= floor/total).
The scheduler normalises weights to shares, so absolute scale is free.

Usage
-----
    AFR_DB__HOST=... AFR_DB__PASSWORD=... python scripts/reweight_pool.py \
        --list-name smoke-01                 # dry-run: print proposed table
    ... python scripts/reweight_pool.py --list-name smoke-01 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import asyncpg

# Cells with >=2 ok rollouts and reward std below this are "degenerate"
# (no distillation signal). Mirrors AFR_PUBLISH_MIN_REWARD_STD default.
DEFAULT_MIN_STD = 0.05


def _conn_kwargs() -> dict:
    return dict(
        host=os.environ["AFR_DB__HOST"],
        port=int(os.environ.get("AFR_DB__PORT", "5432")),
        user=os.environ["AFR_DB__USER"],
        password=os.environ["AFR_DB__PASSWORD"],
        database=os.environ["AFR_DB__NAME"],
    )


async def _env_yield(conn, envs: list[str], window_hours: int, min_std: float) -> dict[str, tuple[float, int]]:
    """Return {env: (yield, n_cells)} over the rolling window."""
    rows = await conn.fetch(
        """
        with cell as (
          select env_name, task_id, teacher_name,
                 count(*) n, stddev_pop(reward) sd
          from rollouts
          where status = 'ok' and reward is not null
            and created_at > now() - make_interval(hours => $1)
            and env_name = any($2::text[])
          group by 1, 2, 3
          having count(*) >= 2
        )
        select env_name,
               count(*) n_cells,
               count(*) filter (where sd >= $3) var_ok
        from cell group by 1
        """,
        window_hours, envs, min_std,
    )
    out: dict[str, tuple[float, int]] = {}
    for r in rows:
        n = int(r["n_cells"])
        y = (int(r["var_ok"]) / n) if n else 0.0
        out[r["env_name"]] = (y, n)
    return out


def compute_weights(
    envs: list[str],
    stats: dict[str, tuple[float, int]],
    *,
    cut_yield: float,
    alpha: float,
    min_cells: int,
    floor: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Return (env_weights, per-env status label). Pure — unit-testable."""
    # Warm-up score = median yield of the envs that DO have enough data
    # (so a data-poor env samples at a typical rate, not a dud rate).
    mature_yields = sorted(
        y for e, (y, n) in stats.items() if n >= min_cells
    )
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
            labels[e] = f"warmup(n={n}<{min_cells})"
        elif y < cut_yield:
            score = 0.0
            labels[e] = f"CUT(yield={y:.0%})"
        else:
            score = y ** alpha
            labels[e] = f"yield={y:.0%}"
        weights[e] = round(floor + score, 6)
    return weights, labels


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-name", required=True)
    # 48h: recent enough to reflect current steady state, long enough to be
    # stable. Avoid very long windows until any one-off backlog-flush bursts
    # (which run at low variance and depress yield) age out of the window.
    ap.add_argument("--window-hours", type=int, default=48)
    ap.add_argument("--cut-yield", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--min-std", type=float, default=DEFAULT_MIN_STD)
    ap.add_argument("--floor", type=float, default=0.03)
    ap.add_argument("--apply", action="store_true", help="write config (default: dry-run)")
    args = ap.parse_args()

    conn = await asyncpg.connect(**_conn_kwargs())
    try:
        row = await conn.fetchrow(
            "select config from sampling_lists where list_name = $1", args.list_name
        )
        if row is None:
            raise SystemExit(f"no sampling_lists row: {args.list_name}")
        config = row["config"]
        if isinstance(config, str):
            config = json.loads(config)
        pool = config.get("pool") or {}
        envs = list(dict.fromkeys(pool.get("envs") or []))
        if not envs:
            raise SystemExit("pool.envs empty")

        stats = await _env_yield(conn, envs, args.window_hours, args.min_std)
        weights, labels = compute_weights(
            envs, stats,
            cut_yield=args.cut_yield, alpha=args.alpha,
            min_cells=args.min_cells, floor=args.floor,
        )
        total = sum(weights.values()) or 1.0
        old = pool.get("env_weights") or {}

        print(f"list={args.list_name} window={args.window_hours}h alpha={args.alpha} "
              f"cut<{args.cut_yield:.0%} min_cells={args.min_cells} floor={args.floor}")
        print(f"{'env':34s} {'status':22s} {'share':>7s} {'was':>7s}")
        os_total = sum(old.values()) or 1.0
        for e in sorted(envs, key=lambda x: weights[x], reverse=True):
            share = weights[e] / total
            was = (old.get(e, 0.0) / os_total) if old else float("nan")
            was_s = f"{was:6.1%}" if old else "   -  "
            print(f"  {e:32s} {labels[e]:22s} {share:6.1%} {was_s}")

        if not args.apply:
            print("\n(dry-run — pass --apply to write env_weights into config.pool)")
            return

        pool["env_weights"] = weights
        config["pool"] = pool
        await conn.execute(
            "update sampling_lists set config = $2 where list_name = $1",
            args.list_name, json.dumps(config),
        )
        print("\nAPPLIED: config.pool.env_weights updated.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
