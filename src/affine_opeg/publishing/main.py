"""``af servers generator-publisher`` — R2 rollout publisher loop.

Long-running service that periodically dumps successful rollouts to
parquet and uploads to the cortex-owned R2 bucket. Decoupled from the
producer process so publish cadence is independent of generation
throughput.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

import click


def _configure_root_logging(verbosity: int) -> None:
    """Minimal CLI-side log setup; ``configure_logging`` below replaces
    handlers for the structured publisher log."""
    level = {0: logging.CRITICAL + 1, 1: logging.INFO, 2: logging.DEBUG}.get(
        min(verbosity, 2), logging.INFO,
    )
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("-v", "--verbose", count=True, default=1,
              help="Increase logging verbosity (-v=INFO, -vv=DEBUG)")
@click.option("--interval-s", type=float, default=None,
              help="Publish cadence in seconds (overrides AFR_PUBLISH_INTERVAL_S)")
def main(verbose: int, interval_s: float | None) -> None:
    _configure_root_logging(verbose)
    from affine_opeg.infrastructure.config import load_config
    from affine_opeg.infrastructure.logging import configure_logging, get_logger
    from affine_opeg.publishing.publisher import publisher_loop

    cfg = load_config()
    configure_logging(cfg, service="publisher")
    log = get_logger("publisher.main")

    cadence = interval_s if interval_s is not None else float(
        os.environ.get("AFR_PUBLISH_INTERVAL_S", "300")
    )

    async def _run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, stop.set)
            except (NotImplementedError, RuntimeError):
                pass

        task = asyncio.create_task(publisher_loop(interval_s=cadence))
        await stop.wait()
        log.info("publisher.stopping")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


if __name__ == "__main__":
    main()
