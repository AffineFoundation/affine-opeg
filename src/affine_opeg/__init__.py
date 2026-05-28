"""Rollout generator service.

Long-running cortex service that produces rollouts (agent traces) and
persists them. The companion ``publishing`` subpackage exports those
rollouts to R2 parquet for downstream evaluators.

Submodules:
    workers/                producer entry points (rollout_producer)
    application/            producer loop + use case orchestration
    adapters/               sandboxes, teachers, normalizers, blob, metadata
    domain/                 typed models + Protocol ports
    infrastructure/         config, db, logging, metrics
    migrations/             alembic for the rollout schema (PG)
    publishing/             rollout export to R2 parquet
"""

__version__ = "0.1.0"
