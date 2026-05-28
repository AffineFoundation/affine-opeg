"""sampling_progress: attempts + published_at

Splits ``sampling_progress.collected`` into two counters and adds a
published flag so the publisher can run an incremental query:

* ``attempts`` — claim-time counter (assigns ``sample_idx``); previous
  ``collected`` semantics.
* ``collected`` — success counter (``status='ok'`` rollouts only);
  cell-frozen / publish decisions read this.
* ``published_at`` — timestamp the publisher set after committing the
  cell's shard to R2; ``IS NULL`` means "not yet published".

Backfill:
    attempts := old collected
    collected := COUNT(rollouts WHERE status='ok'
                                  AND sample_idx < target_samples)

A partial index on (last_updated) WHERE published_at IS NULL lets the
publisher's "what's new?" query stay O(pending) rather than scanning
all frozen rows.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sampling_progress",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "sampling_progress",
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    # Backfill 1: attempts inherits old collected (which was actually
    # the claim counter).
    op.execute("""
        UPDATE sampling_progress
        SET attempts = collected
    """)

    # Backfill 2: collected becomes the count of ok rollouts within the
    # configured cell window (sample_idx < target_samples). Slow but
    # one-shot.
    op.execute("""
        UPDATE sampling_progress sp
        SET collected = COALESCE(sub.ok_count, 0)
        FROM (
            SELECT env_name, task_id, teacher_name, COUNT(*) AS ok_count
            FROM rollouts
            WHERE status = 'ok'
            GROUP BY env_name, task_id, teacher_name
        ) sub
        WHERE sp.env_name = sub.env_name
          AND sp.task_id = sub.task_id
          AND sp.teacher_name = sub.teacher_name
    """)

    # Partial index: publisher only ever scans unpublished frozen-ish cells.
    op.create_index(
        "idx_sp_unpublished",
        "sampling_progress",
        ["last_updated"],
        postgresql_where=sa.text("published_at IS NULL"),
    )

    # Drop the server_default once the backfill is done — generator code
    # always writes attempts explicitly.
    op.alter_column("sampling_progress", "attempts", server_default=None)


def downgrade() -> None:
    op.drop_index("idx_sp_unpublished", table_name="sampling_progress")
    # Restore old collected semantics (claim counter) by overwriting
    # with attempts. Lossy: the success counter is gone afterwards.
    op.execute("UPDATE sampling_progress SET collected = attempts")
    op.drop_column("sampling_progress", "published_at")
    op.drop_column("sampling_progress", "attempts")
