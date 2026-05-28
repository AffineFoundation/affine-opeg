"""initial schema

Creates the full set of business tables. Field choices and naming intentionally
mirror affine-cortex equivalents (sample_results, miners, scores, ...) so the
mental model carries over.

Revision ID: 0001
Revises:
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    op.create_table(
        "environments",
        sa.Column("env_name", sa.Text(), primary_key=True),
        sa.Column("dataset", sa.Text(), nullable=False),
        sa.Column("dataset_version", sa.Text(), nullable=False),
        sa.Column("task_id_range", postgresql.INT4RANGE(), nullable=False),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "tasks",
        sa.Column("env_name", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("repo", sa.Text(), nullable=False),
        sa.Column("base_commit", sa.Text(), nullable=False),
        sa.Column("problem", sa.Text(), nullable=False),
        sa.Column("hidden_tests", postgresql.JSONB(), nullable=False),
        sa.Column("difficulty", sa.Text()),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("env_name", "task_id"),
        sa.ForeignKeyConstraint(["env_name"], ["environments.env_name"]),
    )
    op.create_index("idx_tasks_difficulty", "tasks", ["difficulty"], postgresql_where=sa.text("difficulty IS NOT NULL"))

    op.create_table(
        "teachers",
        sa.Column("teacher_name", sa.Text(), primary_key=True),
        sa.Column("model_family", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("api_key_env", sa.Text(), nullable=False),
        sa.Column("tool_format", sa.Text(), nullable=False),
        sa.Column("reasoning_format", sa.Text(), nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=False),
        sa.Column("price_per_mtoken_in", sa.Numeric(8, 4)),
        sa.Column("price_per_mtoken_out", sa.Numeric(8, 4)),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("registered_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "rollouts",
        sa.Column("rollout_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("env_name", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("teacher_name", sa.Text(), nullable=False),
        sa.Column("sample_idx", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("top_p", sa.Float()),
        sa.Column("seed", sa.BigInteger()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reward", sa.Float()),
        sa.Column("reward_breakdown", postgresql.JSONB()),
        sa.Column("steps", sa.Integer()),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("tokens_in", sa.Integer()),
        sa.Column("tokens_out", sa.Integer()),
        sa.Column("cost_usd", sa.Numeric(10, 6)),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("extra_compressed", postgresql.BYTEA(), nullable=False),
        sa.Column("extra_sha256", sa.Text(), nullable=False),
        sa.Column("blob_uri", sa.Text()),
        sa.Column("group_label", sa.Text()),
        sa.Column("producer_id", sa.Text(), nullable=False),
        sa.Column("trace_id", sa.Text()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["env_name", "task_id"], ["tasks.env_name", "tasks.task_id"]),
        sa.ForeignKeyConstraint(["teacher_name"], ["teachers.teacher_name"]),
        sa.UniqueConstraint("env_name", "task_id", "teacher_name", "sample_idx", name="uq_rollouts_business_key"),
    )
    op.create_index("idx_rollouts_eteach", "rollouts", ["env_name", "task_id", "teacher_name", "reward"])
    op.create_index("idx_rollouts_created", "rollouts", [sa.text("created_at DESC")])
    op.create_index("idx_rollouts_teacher", "rollouts", ["teacher_name", sa.text("created_at DESC")])
    op.create_index(
        "idx_rollouts_group",
        "rollouts",
        ["group_label", "env_name"],
        postgresql_where=sa.text("group_label IS NOT NULL"),
    )

    op.create_table(
        "teacher_api_calls",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("rollout_id", postgresql.UUID(as_uuid=True)),
        sa.Column("teacher_name", sa.Text(), nullable=False),
        sa.Column("step", sa.Integer()),
        sa.Column("request_meta", postgresql.JSONB(), nullable=False),
        sa.Column("response_meta", postgresql.JSONB()),
        sa.Column("tokens_in", sa.Integer()),
        sa.Column("tokens_out", sa.Integer()),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(10, 6)),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error", postgresql.JSONB()),
        sa.Column("trace_id", sa.Text()),
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["rollout_id"], ["rollouts.rollout_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["teacher_name"], ["teachers.teacher_name"]),
    )
    op.create_index("idx_tac_teacher_ts", "teacher_api_calls", ["teacher_name", sa.text("ts DESC")])
    op.create_index("idx_tac_rollout", "teacher_api_calls", ["rollout_id"])
    op.create_index(
        "idx_tac_failures",
        "teacher_api_calls",
        [sa.text("ts DESC")],
        postgresql_where=sa.text("status != 'ok'"),
    )

    op.create_table(
        "sampling_lists",
        sa.Column("list_name", sa.Text(), primary_key=True),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "sampling_progress",
        sa.Column("list_name", sa.Text(), nullable=False),
        sa.Column("env_name", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("teacher_name", sa.Text(), nullable=False),
        sa.Column("target_samples", sa.Integer(), nullable=False),
        sa.Column("collected", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_updated", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("list_name", "env_name", "task_id", "teacher_name"),
        sa.ForeignKeyConstraint(["list_name"], ["sampling_lists.list_name"]),
        sa.ForeignKeyConstraint(["env_name", "task_id"], ["tasks.env_name", "tasks.task_id"]),
        sa.ForeignKeyConstraint(["teacher_name"], ["teachers.teacher_name"]),
    )
    op.create_index(
        "idx_sampling_progress_open",
        "sampling_progress",
        ["list_name"],
        postgresql_where=sa.text("collected < target_samples"),
    )

    op.create_table(
        "pair_sets",
        sa.Column("pair_set_name", sa.Text(), primary_key=True),
        sa.Column("source_filter", postgresql.JSONB(), nullable=False),
        sa.Column("strategy", postgresql.JSONB(), nullable=False),
        sa.Column("rollouts_scanned", sa.Integer()),
        sa.Column("pairs_created", sa.Integer()),
        sa.Column("groups_summary", postgresql.JSONB()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("triggered_by", sa.Text(), nullable=False),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("trace_id", sa.Text()),
    )

    op.create_table(
        "pairs",
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("pair_set_name", sa.Text(), nullable=False),
        sa.Column("env_name", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("teacher_name", sa.Text(), nullable=False),
        sa.Column("win_rollout", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("lose_rollout", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reward_win", sa.Float(), nullable=False),
        sa.Column("reward_lose", sa.Float(), nullable=False),
        sa.Column("reward_gap", sa.Float(), nullable=False),
        sa.Column("selection_meta", postgresql.JSONB()),
        sa.ForeignKeyConstraint(["pair_set_name"], ["pair_sets.pair_set_name"]),
        sa.ForeignKeyConstraint(["env_name", "task_id"], ["tasks.env_name", "tasks.task_id"]),
        sa.ForeignKeyConstraint(["win_rollout"], ["rollouts.rollout_id"]),
        sa.ForeignKeyConstraint(["lose_rollout"], ["rollouts.rollout_id"]),
        sa.CheckConstraint("win_rollout != lose_rollout", name="ck_pairs_distinct_rollouts"),
    )
    op.create_index("idx_pairs_set", "pairs", ["pair_set_name"])
    op.create_index("idx_pairs_set_env", "pairs", ["pair_set_name", "env_name"])
    op.create_index("idx_pairs_set_teacher", "pairs", ["pair_set_name", "teacher_name"])
    op.create_index("idx_pairs_win", "pairs", ["win_rollout"])
    op.create_index("idx_pairs_lose", "pairs", ["lose_rollout"])

    op.create_table(
        "student_submissions",
        sa.Column("student_name", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("hf_repo", sa.Text()),
        sa.Column("arch", sa.Text(), nullable=False),
        sa.Column("param_b", sa.Float()),
        sa.Column("model_hash", sa.Text(), nullable=False),
        sa.Column("template_check", sa.Text()),
        sa.Column("is_valid", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("invalid_reason", sa.Text()),
        sa.Column("submitted_by", sa.Text(), nullable=False),
        sa.Column("submitted_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("student_name", "revision"),
    )
    op.create_index("idx_students_hash", "student_submissions", ["model_hash"])
    op.create_index(
        "idx_students_valid",
        "student_submissions",
        ["is_valid", sa.text("submitted_at DESC")],
    )

    op.create_table(
        "anti_copy_results",
        sa.Column("student_name", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("round_ts", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("is_copy", sa.Boolean(), nullable=False),
        sa.Column("copied_from", postgresql.JSONB()),
        sa.Column("detection_meta", postgresql.JSONB()),
        sa.PrimaryKeyConstraint("student_name", "revision", "round_ts"),
        sa.ForeignKeyConstraint(
            ["student_name", "revision"],
            ["student_submissions.student_name", "student_submissions.revision"],
        ),
    )
    op.create_index(
        "idx_anticopy_latest",
        "anti_copy_results",
        ["student_name", "revision", sa.text("round_ts DESC")],
    )

    op.create_table(
        "student_scores",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_name", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("pair_set_name", sa.Text(), nullable=False),
        sa.Column("mean_score", sa.Float(), nullable=False),
        sa.Column("win_rate", sa.Float(), nullable=False),
        sa.Column("scores_by_env", postgresql.JSONB(), nullable=False),
        sa.Column("scores_by_teacher", postgresql.JSONB(), nullable=False),
        sa.Column("scores_by_difficulty", postgresql.JSONB()),
        sa.Column("pairs_evaluated", sa.Integer(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("forward_engine", sa.Text(), nullable=False),
        sa.Column("total_forward_tokens", sa.BigInteger()),
        sa.Column("total_compute_seconds", sa.Float()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("latest_marker", sa.Text()),
        sa.Column("triggered_by", sa.Text(), nullable=False),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("trace_id", sa.Text()),
        sa.PrimaryKeyConstraint("run_id", "student_name", "revision"),
        sa.ForeignKeyConstraint(
            ["student_name", "revision"],
            ["student_submissions.student_name", "student_submissions.revision"],
        ),
        sa.ForeignKeyConstraint(["pair_set_name"], ["pair_sets.pair_set_name"]),
    )
    op.create_index(
        "idx_scores_pairset_finished",
        "student_scores",
        ["pair_set_name", sa.text("finished_at DESC")],
    )
    op.create_index(
        "idx_scores_student_finished",
        "student_scores",
        ["student_name", "revision", sa.text("finished_at DESC")],
    )
    # Partial unique: at most one LATEST row per (student, revision, pair_set)
    op.create_index(
        "uq_student_latest",
        "student_scores",
        ["student_name", "revision", "pair_set_name"],
        unique=True,
        postgresql_where=sa.text("latest_marker = 'LATEST'"),
    )

    op.create_table(
        "student_pair_scores",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_name", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ce_win", sa.Float(), nullable=False),
        sa.Column("ce_lose", sa.Float(), nullable=False),
        sa.Column("tokens_win", sa.Integer(), nullable=False),
        sa.Column("tokens_lose", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("ce_per_message_win", postgresql.JSONB()),
        sa.Column("ce_per_message_lose", postgresql.JSONB()),
        sa.PrimaryKeyConstraint("run_id", "student_name", "revision", "pair_id"),
        sa.ForeignKeyConstraint(["pair_id"], ["pairs.pair_id"]),
    )
    op.create_index("idx_pair_scores_pair", "student_pair_scores", ["pair_id"])
    op.create_index(
        "idx_pair_scores_ranked",
        "student_pair_scores",
        ["run_id", "student_name", "revision", sa.text("score DESC")],
    )

    op.create_table(
        "student_deployments",
        sa.Column("deployment_id", sa.Text(), primary_key=True),
        sa.Column("student_name", sa.Text(), nullable=False),
        sa.Column("revision", sa.Text(), nullable=False),
        sa.Column("engine", sa.Text(), nullable=False),
        sa.Column("host", sa.Text(), nullable=False),
        sa.Column("gpu_indices", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("base_url", sa.Text()),
        sa.Column("instance_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_retry_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("last_health_check_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("meta", postgresql.JSONB()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["student_name", "revision"],
            ["student_submissions.student_name", "student_submissions.revision"],
        ),
    )
    op.create_index(
        "idx_deployments_status",
        "student_deployments",
        ["status", "last_health_check_at"],
    )
    op.create_index(
        "idx_deployments_student",
        "student_deployments",
        ["student_name", "revision"],
    )

    op.create_table(
        "service_heartbeats",
        sa.Column("worker_id", sa.Text(), primary_key=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("host", sa.Text()),
        sa.Column("pid", sa.Integer()),
        sa.Column("version", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("current_task", postgresql.JSONB()),
        sa.Column("last_seen", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("meta", postgresql.JSONB()),
    )

    op.create_table(
        "system_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("service", sa.Text(), nullable=False),
        sa.Column("level", sa.Text(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("message", sa.Text()),
        sa.Column("context", postgresql.JSONB()),
        sa.Column("trace_id", sa.Text()),
        sa.Column("rollout_id", postgresql.UUID(as_uuid=True)),
        sa.Column("pair_id", postgresql.UUID(as_uuid=True)),
        sa.Column("run_id", postgresql.UUID(as_uuid=True)),
    )
    op.create_index("idx_sysev_ts", "system_events", [sa.text("ts DESC")])
    op.create_index("idx_sysev_svc_lvl", "system_events", ["service", "level", sa.text("ts DESC")])
    op.create_index("idx_sysev_event", "system_events", ["event", sa.text("ts DESC")])

    op.create_table(
        "metrics_minutely",
        sa.Column("ts_minute", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("labels", postgresql.JSONB(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("ts_minute", "metric", "labels"),
    )
    op.create_index(
        "idx_metrics_metric_ts",
        "metrics_minutely",
        ["metric", sa.text("ts_minute DESC")],
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("entity_kind", sa.Text()),
        sa.Column("entity_id", sa.Text()),
        sa.Column("payload", postgresql.JSONB()),
        sa.Column("trace_id", sa.Text()),
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_audit_action_ts", "audit_log", ["action", sa.text("ts DESC")])
    op.create_index("idx_audit_entity", "audit_log", ["entity_kind", "entity_id"])


def downgrade() -> None:
    for tbl in [
        "audit_log",
        "metrics_minutely",
        "system_events",
        "service_heartbeats",
        "student_deployments",
        "student_pair_scores",
        "student_scores",
        "anti_copy_results",
        "student_submissions",
        "pairs",
        "pair_sets",
        "sampling_progress",
        "sampling_lists",
        "teacher_api_calls",
        "rollouts",
        "teachers",
        "tasks",
        "environments",
    ]:
        op.drop_table(tbl)
