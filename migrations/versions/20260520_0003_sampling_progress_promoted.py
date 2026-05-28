"""sampling_progress: promoted_at

Adds the third lifecycle timestamp on a cell row:

    attempts++        → producer claimed a slot
    collected++       → producer wrote a status='ok' rollout
    published_at      → publisher uploaded the parquet to the PRIVATE bucket
    promoted_at       → promoter copied the shard to the PUBLIC bucket
                        once ``mature_at`` (committed_at + 24h) elapsed

The split is what lets us physically isolate fresh cells (only in
``affine-distill-v2-private``, AK/SK required) from mature ones (mirrored
to ``affine-distill-v2-public``, miner-facing).

Partial index ``idx_sp_promote_pending`` keeps the promoter's "what's
ready to promote?" query O(pending) rather than scanning all published
rows.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sampling_progress",
        sa.Column("promoted_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    # Partial index on rows that are published but not yet promoted —
    # the promoter's incremental query touches only this slice.
    op.create_index(
        "idx_sp_promote_pending",
        "sampling_progress",
        ["published_at"],
        postgresql_where=sa.text("published_at IS NOT NULL AND promoted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_sp_promote_pending", table_name="sampling_progress")
    op.drop_column("sampling_progress", "promoted_at")
