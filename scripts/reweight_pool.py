"""CLI for the yield-driven pool reweighter (dry-run by default).

The reweighter also runs automatically inside ``publisher_loop`` (Stage 3,
no cron needed) — this script is for manual inspection / one-off applies.
Core logic lives in ``affine_opeg.publishing.reweighter``.

Usage
-----
    python scripts/reweight_pool.py --list-name smoke-01            # dry-run
    python scripts/reweight_pool.py --list-name smoke-01 --apply
"""

from __future__ import annotations

import argparse
import asyncio

from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import get_sessionmaker
from affine_opeg.publishing.reweighter import (
    DEFAULT_MIN_STD,
    pins_from_env,
    reweight_pool,
)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-name", required=True)
    # 48h: recent enough to reflect current steady state, long enough to be
    # stable. Avoid very long windows until one-off backlog-flush bursts
    # (low variance, depress yield) age out of the window.
    ap.add_argument("--window-hours", type=int, default=48)
    ap.add_argument("--cut-yield", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--min-std", type=float, default=DEFAULT_MIN_STD)
    ap.add_argument("--floor", type=float, default=0.03)
    ap.add_argument("--no-pins", action="store_true", help="ignore AFR_REWEIGHT_PINS (e.g. swe floor)")
    ap.add_argument("--apply", action="store_true", help="write config (default: dry-run)")
    args = ap.parse_args()

    sm = get_sessionmaker(load_config())
    pins = {} if args.no_pins else pins_from_env()
    weights, labels = await reweight_pool(
        sm, args.list_name,
        window_hours=args.window_hours, cut_yield=args.cut_yield,
        alpha=args.alpha, min_cells=args.min_cells, min_std=args.min_std,
        floor=args.floor, pins=pins, apply=args.apply,
    )
    total = sum(weights.values()) or 1.0
    print(f"list={args.list_name} window={args.window_hours}h alpha={args.alpha} "
          f"cut<{args.cut_yield:.0%} min_cells={args.min_cells} floor={args.floor} pins={pins or '{}'}")
    print(f"{'env':34s} {'status':26s} {'share':>7s}")
    for e in sorted(weights, key=lambda x: weights[x], reverse=True):
        print(f"  {e:32s} {labels[e]:26s} {weights[e] / total:6.1%}")
    print("\nAPPLIED." if args.apply else "\n(dry-run — pass --apply to write env_weights)")


if __name__ == "__main__":
    asyncio.run(main())
