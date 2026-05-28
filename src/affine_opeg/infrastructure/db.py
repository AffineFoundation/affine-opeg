"""Async SQLAlchemy engine wiring.

Supports two auth modes:
    - ``password``: plain DSN (local dev, RDS w/o IAM)
    - ``iam``: short-lived RDS auth token, refreshed per-connection

The IAM path uses boto3's ``generate_db_auth_token`` and is signed every
time a new connection is opened by the pool. Combined with RDS Proxy, this
keeps the upstream connection count low while individual workers can fan
out asyncio tasks.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event as sa_event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from affine_opeg.infrastructure.config import AppConfig, DbConfig


def make_async_dsn(db: DbConfig) -> str:
    """Build an asyncpg DSN. For IAM mode this is a placeholder — actual
    token is injected on every connect via ``connect_args``."""
    sslmode = "" if db.ssl == "disable" else f"?ssl={db.ssl}"
    if db.auth == "iam":
        return f"postgresql+asyncpg://{db.user}@{db.host}:{db.port}/{db.name}{sslmode}"
    pw = db.password.get_secret_value()
    return f"postgresql+asyncpg://{db.user}:{pw}@{db.host}:{db.port}/{db.name}{sslmode}"


def _iam_password_provider(db: DbConfig, region: str):  # type: ignore[no-untyped-def]
    import boto3

    rds = boto3.client("rds", region_name=region)

    def _provider() -> str:
        return rds.generate_db_auth_token(
            DBHostname=db.host,
            Port=db.port,
            DBUsername=db.user,
            Region=region,
        )

    return _provider


# AppConfig is a pydantic model (mutable, unhashable), so the engine /
# sessionmaker cache cannot use ``functools.lru_cache``. We keep a single
# pair per process keyed by the cfg's identity — each process loads config
# once and reuses the same engine throughout.
_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _jsonb_encode(value):  # type: ignore[no-untyped-def]
    """asyncpg JSONB encoder that survives both call paths:

    * SQLAlchemy ORM JSONB columns hand us an already-``json.dumps``'d str.
    * Raw ``text(...)`` SQL with a dict bind goes straight to asyncpg with
      the dict intact.

    Returning ``str`` as-is and dumping anything else covers both without
    double-encoding (which would store a quoted JSON string in the column).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return json.dumps(value)


async def _init_asyncpg_codecs(conn):  # type: ignore[no-untyped-def]
    """Per-connection initialiser: register JSON/JSONB codecs.

    Without this, raw ``text(...)`` queries that bind ``dict`` to a JSONB
    column crash because asyncpg's default codec expects a string. With
    it, both ORM and raw SQL paths land the same value shape in the DB.
    """
    for type_name in ("jsonb", "json"):
        await conn.set_type_codec(
            type_name, encoder=_jsonb_encode, decoder=json.loads,
            schema="pg_catalog", format="text",
        )


def get_engine(cfg: AppConfig) -> AsyncEngine:
    global _engine
    if _engine is not None:
        return _engine
    db = cfg.db
    connect_args: dict[str, object] = {
        "server_settings": {"application_name": f"afr-{cfg.service}"},
        "timeout": db.statement_timeout_ms / 1000,
    }
    if db.auth == "iam":
        provider = _iam_password_provider(db, cfg.aws.region)
        connect_args["password"] = provider  # asyncpg accepts callable

    # ``json_serializer=json.dumps`` is required for asyncpg's JSONB codec —
    # SQLAlchemy hands dicts into the codec, which then calls ``.encode()``;
    # without a serializer that path crashes with AttributeError on dicts.
    # The asyncpg JSONB codec registered in `_init_asyncpg_codecs` already
    # handles dict <-> JSON. Adding SQLAlchemy's ``json_serializer`` would
    # cause double-encoding: SQLAlchemy dumps dict -> str, asyncpg dumps
    # str -> JSON-encoded string, leaving the column with a *string* value
    # instead of an object (``meta::jsonb ? 'key'`` returns false). One
    # serialiser is enough.
    _engine = create_async_engine(
        make_async_dsn(db),
        pool_size=db.pool_size,
        max_overflow=db.max_overflow,
        pool_recycle=db.pool_recycle,
        pool_pre_ping=db.pool_pre_ping,
        connect_args=connect_args,
        future=True,
    )

    @sa_event.listens_for(_engine.sync_engine, "connect")
    def _register_codecs(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        # ``dbapi_conn`` is SQLAlchemy's AsyncAdapt_asyncpg_connection;
        # ``run_async`` dispatches the supplied coroutine factory onto the
        # event loop the connection lives on.
        dbapi_conn.run_async(
            lambda raw: _init_asyncpg_codecs(raw)
        )

    return _engine


def get_sessionmaker(cfg: AppConfig) -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(cfg), expire_on_commit=False, class_=AsyncSession,
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope(cfg: AppConfig) -> AsyncIterator[AsyncSession]:
    """One transaction per scope. Commit on success, rollback on raise."""
    factory = get_sessionmaker(cfg)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
