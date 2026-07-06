"""Bootstrap the PHYBench env into PG + add it to a sampling list's pool.

phybench loads via our builtin builder (the PI wheel is broken) — a
``SingleTurnEnv`` over the ungated ``Eureka-Lab/PHYBench`` dataset with a
continuous EED reward. This script:

  1. registers ``verifiers:phybench`` + bulk-inserts its tasks, and
  2. appends the env to ``sampling_lists.config.pool.envs`` and sets
     ``pool.task_id_pool[verifiers:phybench] = range(n)`` so the scheduler
     starts sampling it (the reweighter's warm-up guard gives it a fair
     trial before any yield-based cut).

Run against the migrated prod DB (inside the container, env has AFR_DB__*):
    python scripts/bootstrap_phybench.py --list-name smoke-01
    python scripts/bootstrap_phybench.py --list-name smoke-01 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json

from sqlalchemy import text

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg import SqlAlchemyMetadataStore
from affine_opeg.adapters.task_sources.verifiers import VerifiersTaskSource
from affine_opeg.application.bootstrap import load_env_and_tasks
from affine_opeg.domain.ids import EnvName
from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import get_sessionmaker

ENV_NAME = "verifiers:phybench"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-name", default="smoke-01")
    ap.add_argument("--apply", action="store_true", help="write tasks + pool (default: dry-run count)")
    args = ap.parse_args()

    cfg = load_config()
    sm = get_sessionmaker(cfg)
    md = SqlAlchemyMetadataStore(sm)

    env_name = EnvName(ENV_NAME)
    ts = VerifiersTaskSource(env_name)
    n = await ts.task_count()
    print(f"phybench task_count (train rows) = {n}")
    if n == 0:
        raise SystemExit("phybench enumerated 0 tasks — env load / EED broken?")

    if not args.apply:
        print("(dry-run — pass --apply to insert tasks + add to pool)")
        return

    _, inserted = await load_env_and_tasks(
        md, env_name=env_name, dataset="phybench",
        dataset_version="Eureka-Lab/PHYBench", task_source=ts, actor="bootstrap",
    )
    print(f"tasks inserted: {inserted} (env registered: {env_name})")

    # Add to the pool: envs list + task_id_pool. Weight is left to the
    # reweighter (warm-up until it has enough cells).
    async with sm() as s:
        row = (await s.execute(
            text("select config from sampling_lists where list_name=:ln"),
            {"ln": args.list_name},
        )).first()
        if row is None:
            raise SystemExit(f"no sampling_lists row: {args.list_name}")
        config = row[0]
        if isinstance(config, str):
            config = json.loads(config)
        pool = config.setdefault("pool", {})
        envs = list(pool.get("envs") or [])
        if ENV_NAME not in envs:
            envs.append(ENV_NAME)
        pool["envs"] = envs
        tip = pool.setdefault("task_id_pool", {})
        tip[ENV_NAME] = list(range(n))
        config["pool"] = pool
        await s.execute(
            text("update sampling_lists set config=:c where list_name=:ln"),
            {"c": json.dumps(config), "ln": args.list_name},
        )
        await s.commit()
    print(f"pool updated: {ENV_NAME} added with task_id_pool range(0,{n})")


if __name__ == "__main__":
    asyncio.run(main())
