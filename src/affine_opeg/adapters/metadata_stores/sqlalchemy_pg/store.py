"""MetadataStore + UnitOfWork composition.

Constructed once per process from infrastructure DI; injected into application
use cases as a single ``MetadataStore`` value.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import async_sessionmaker

from affine_opeg.adapters.metadata_stores.sqlalchemy_pg.repositories import (
    SaDeploymentRepository,
    SaPairRepository,
    SaPairSetRepository,
    SaRolloutRepository,
    SaSamplingListRepository,
    SaStudentRepository,
    SaStudentScoreRepository,
    SaTaskRepository,
    SaTeacherRepository,
)


class SqlAlchemyUnitOfWork:
    """Aggregate of repositories bound to one session/transaction."""

    def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
        self._session = session
        self.tasks = SaTaskRepository(session)
        self.teachers = SaTeacherRepository(session)
        self.rollouts = SaRolloutRepository(session)
        self.pair_sets = SaPairSetRepository(session)
        self.pairs = SaPairRepository(session)
        self.sampling_lists = SaSamplingListRepository(session)
        self.students = SaStudentRepository(session)
        self.student_scores = SaStudentScoreRepository(session)
        self.deployments = SaDeploymentRepository(session)

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()


class SqlAlchemyMetadataStore:
    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncIterator[SqlAlchemyUnitOfWork]:
        async with self._sessionmaker() as session:
            uow = SqlAlchemyUnitOfWork(session)
            try:
                yield uow
                await session.commit()
            except Exception:
                await session.rollback()
                raise
