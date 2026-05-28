"""Alembic env that wires to our pydantic settings + SQLAlchemy metadata."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg.orm import metadata as target_metadata
from affine_opeg.infrastructure.config import load_config
from affine_opeg.infrastructure.db import make_async_dsn

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)


def _get_dsn() -> str:
    settings = load_config()
    return make_async_dsn(settings.db)


def run_migrations_offline() -> None:
    context.configure(
        url=_get_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online_async() -> None:
    cfg = alembic_config.get_section(alembic_config.config_ini_section) or {}
    cfg["sqlalchemy.url"] = _get_dsn()
    engine = async_engine_from_config(cfg, prefix="sqlalchemy.")
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_migrations_online_async())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
