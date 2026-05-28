"""``af servers generator`` — rollout producer entry point.

Runs the rollout producer loop. Configuration is
supplied via ``AFR_*`` environment variables — see
``affine_opeg.infrastructure.config`` for the supported keys:

    AFR_SAMPLING_LIST          required; which list this producer is bound to
    AFR_ENV_NAMES              optional CSV; restrict to these envs
    AFR_TEACHER_NAMES          optional CSV; restrict to these teachers
    AFR_DB__HOST/USER/PASSWORD/NAME
    AFR_BLOB__ENDPOINT/BUCKET/ACCESS_KEY/SECRET_KEY
    OPENROUTER_API_KEY         consumed by the OpenRouter teacher adapter

The inner loop owns its own asyncio event loop; cortex's CLI is just a
boot shim.
"""

from __future__ import annotations

import asyncio
import logging

import click


def _configure_root_logging(verbosity: int) -> None:
    """Minimal CLI-side log setup; the worker's structlog config takes
    over once ``rollout_producer.main`` runs ``configure_logging``."""
    level = {0: logging.CRITICAL + 1, 1: logging.INFO, 2: logging.DEBUG}.get(
        min(verbosity, 2), logging.INFO,
    )
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("-v", "--verbose", count=True, default=1,
              help="Increase logging verbosity (-v=INFO, -vv=DEBUG)")
def main(verbose: int) -> None:
    """Start the rollout generator worker."""
    _configure_root_logging(verbose)

    # Vendored entry point; resolves to cortex's own copy of the producer.
    from affine_opeg.workers.rollout_producer import main as _producer_main

    asyncio.run(_producer_main())


if __name__ == "__main__":
    main()
