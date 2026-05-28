"""SQLAlchemy ORM tables.

This module is **only** imported by:
    - the SQLAlchemy repository implementations in this package, and
    - ``migrations/env.py`` (for autogenerate diffing).

Application/domain code does not touch ORM rows directly.
"""

from __future__ import annotations

from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import INT4RANGE, JSONB, UUID

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _ts(*, server_default: bool = False) -> Column:
    """Timestamp column shorthand."""
    if server_default:
        return Column("__placeholder__", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    return Column("__placeholder__", TIMESTAMP(timezone=True))


environments = Table(
    "environments", metadata,
    Column("env_name", Text, primary_key=True),
    Column("dataset", Text, nullable=False),
    Column("dataset_version", Text, nullable=False),
    Column("task_id_range", INT4RANGE, nullable=False),
    Column("meta", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
)

tasks = Table(
    "tasks", metadata,
    Column("env_name", Text, nullable=False),
    Column("task_id", Integer, nullable=False),
    Column("repo", Text, nullable=False),
    Column("base_commit", Text, nullable=False),
    Column("problem", Text, nullable=False),
    Column("hidden_tests", JSONB, nullable=False),
    Column("difficulty", Text),
    Column("meta", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    PrimaryKeyConstraint("env_name", "task_id"),
    ForeignKeyConstraint(["env_name"], ["environments.env_name"]),
)

teachers = Table(
    "teachers", metadata,
    Column("teacher_name", Text, primary_key=True),
    Column("model_family", Text, nullable=False),
    Column("provider", Text, nullable=False),
    Column("endpoint", Text, nullable=False),
    Column("api_key_env", Text, nullable=False),
    Column("tool_format", Text, nullable=False),
    Column("reasoning_format", Text, nullable=False),
    Column("context_window", Integer, nullable=False),
    Column("price_per_mtoken_in", Numeric(8, 4)),
    Column("price_per_mtoken_out", Numeric(8, 4)),
    Column("active", Boolean, nullable=False, server_default=text("true")),
    Column("meta", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("registered_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
)

rollouts = Table(
    "rollouts", metadata,
    Column("rollout_id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("env_name", Text, nullable=False),
    Column("task_id", Integer, nullable=False),
    Column("teacher_name", Text, nullable=False),
    Column("sample_idx", Integer, nullable=False),
    Column("temperature", Float, nullable=False),
    Column("top_p", Float),
    Column("seed", BigInteger),
    Column("status", Text, nullable=False),
    Column("reward", Float),
    Column("reward_breakdown", JSONB),
    Column("steps", Integer),
    Column("latency_ms", Integer),
    Column("tokens_in", Integer),
    Column("tokens_out", Integer),
    Column("cost_usd", Numeric(10, 6)),
    Column("schema_version", Text, nullable=False),
    Column("extra_compressed", LargeBinary, nullable=False),
    Column("extra_sha256", Text, nullable=False),
    Column("blob_uri", Text),
    Column("group_label", Text),
    Column("producer_id", Text, nullable=False),
    Column("trace_id", Text),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    ForeignKeyConstraint(["env_name", "task_id"], ["tasks.env_name", "tasks.task_id"]),
    ForeignKeyConstraint(["teacher_name"], ["teachers.teacher_name"]),
    UniqueConstraint("env_name", "task_id", "teacher_name", "sample_idx", name="uq_rollouts_business_key"),
)

teacher_api_calls = Table(
    "teacher_api_calls", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("rollout_id", UUID(as_uuid=True), ForeignKey("rollouts.rollout_id", ondelete="CASCADE")),
    Column("teacher_name", Text, nullable=False),
    Column("step", Integer),
    Column("request_meta", JSONB, nullable=False),
    Column("response_meta", JSONB),
    Column("tokens_in", Integer),
    Column("tokens_out", Integer),
    Column("latency_ms", Integer, nullable=False),
    Column("cost_usd", Numeric(10, 6)),
    Column("status", Text, nullable=False),
    Column("error", JSONB),
    Column("trace_id", Text),
    Column("ts", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
)

sampling_lists = Table(
    "sampling_lists", metadata,
    Column("list_name", Text, primary_key=True),
    Column("config", JSONB, nullable=False),
    Column("description", Text),
    Column("created_by", Text, nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
)

sampling_progress = Table(
    "sampling_progress", metadata,
    Column("list_name", Text, nullable=False),
    Column("env_name", Text, nullable=False),
    Column("task_id", Integer, nullable=False),
    Column("teacher_name", Text, nullable=False),
    Column("target_samples", Integer, nullable=False),
    # attempts := number of times claim_next_cell handed out a sample_idx
    # for this cell (success or failure). Assigns the unique sample_idx
    # column on the rollouts row.
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    # collected := number of status='ok' rollouts persisted; the cell
    # is considered "finished" when collected >= target_samples (the
    # publish trigger) or attempts exhausts the max budget.
    Column("collected", Integer, nullable=False, server_default=text("0")),
    # published_at := timestamp the publisher uploaded the cell's
    # immutable shard to the PRIVATE bucket. Used by the publisher's
    # incremental query.
    Column("published_at", TIMESTAMP(timezone=True), nullable=True),
    # promoted_at := timestamp the promoter copied the shard from the
    # private bucket to the PUBLIC (miner-facing) bucket after the
    # maturation window elapsed. NULL means "still fresh, not yet
    # mirrored to public".
    Column("promoted_at", TIMESTAMP(timezone=True), nullable=True),
    Column("last_updated", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    PrimaryKeyConstraint("list_name", "env_name", "task_id", "teacher_name"),
)

pair_sets = Table(
    "pair_sets", metadata,
    Column("pair_set_name", Text, primary_key=True),
    Column("source_filter", JSONB, nullable=False),
    Column("strategy", JSONB, nullable=False),
    Column("rollouts_scanned", Integer),
    Column("pairs_created", Integer),
    Column("groups_summary", JSONB),
    Column("status", Text, nullable=False),
    Column("triggered_by", Text, nullable=False),
    Column("started_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    Column("finished_at", TIMESTAMP(timezone=True)),
    Column("trace_id", Text),
)

pairs = Table(
    "pairs", metadata,
    Column("pair_id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("pair_set_name", Text, ForeignKey("pair_sets.pair_set_name"), nullable=False),
    Column("env_name", Text, nullable=False),
    Column("task_id", Integer, nullable=False),
    Column("teacher_name", Text, nullable=False),
    Column("win_rollout", UUID(as_uuid=True), ForeignKey("rollouts.rollout_id"), nullable=False),
    Column("lose_rollout", UUID(as_uuid=True), ForeignKey("rollouts.rollout_id"), nullable=False),
    Column("reward_win", Float, nullable=False),
    Column("reward_lose", Float, nullable=False),
    Column("reward_gap", Float, nullable=False),
    Column("selection_meta", JSONB),
    CheckConstraint("win_rollout != lose_rollout", name="distinct_rollouts"),
)
Index("idx_pairs_set", pairs.c.pair_set_name)
Index("idx_pairs_set_env", pairs.c.pair_set_name, pairs.c.env_name)
Index("idx_pairs_set_teacher", pairs.c.pair_set_name, pairs.c.teacher_name)

student_submissions = Table(
    "student_submissions", metadata,
    Column("student_name", Text, nullable=False),
    Column("revision", Text, nullable=False),
    Column("hf_repo", Text),
    Column("arch", Text, nullable=False),
    Column("param_b", Float),
    Column("model_hash", Text, nullable=False),
    Column("template_check", Text),
    Column("is_valid", Boolean, nullable=False, server_default=text("true")),
    Column("invalid_reason", Text),
    Column("submitted_by", Text, nullable=False),
    Column("submitted_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    Column("meta", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    PrimaryKeyConstraint("student_name", "revision"),
)

anti_copy_results = Table(
    "anti_copy_results", metadata,
    Column("student_name", Text, nullable=False),
    Column("revision", Text, nullable=False),
    Column("round_ts", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    Column("is_copy", Boolean, nullable=False),
    Column("copied_from", JSONB),
    Column("detection_meta", JSONB),
    PrimaryKeyConstraint("student_name", "revision", "round_ts"),
    ForeignKeyConstraint(
        ["student_name", "revision"],
        ["student_submissions.student_name", "student_submissions.revision"],
    ),
)

student_scores = Table(
    "student_scores", metadata,
    Column("run_id", UUID(as_uuid=True), nullable=False),
    Column("student_name", Text, nullable=False),
    Column("revision", Text, nullable=False),
    Column("pair_set_name", Text, ForeignKey("pair_sets.pair_set_name"), nullable=False),
    Column("mean_score", Float, nullable=False),
    Column("win_rate", Float, nullable=False),
    Column("scores_by_env", JSONB, nullable=False),
    Column("scores_by_teacher", JSONB, nullable=False),
    Column("scores_by_difficulty", JSONB),
    Column("pairs_evaluated", Integer, nullable=False),
    Column("config", JSONB, nullable=False),
    Column("forward_engine", Text, nullable=False),
    Column("total_forward_tokens", BigInteger),
    Column("total_compute_seconds", Float),
    Column("status", Text, nullable=False),
    Column("latest_marker", Text),
    Column("triggered_by", Text, nullable=False),
    Column("started_at", TIMESTAMP(timezone=True)),
    Column("finished_at", TIMESTAMP(timezone=True)),
    Column("trace_id", Text),
    PrimaryKeyConstraint("run_id", "student_name", "revision"),
    ForeignKeyConstraint(
        ["student_name", "revision"],
        ["student_submissions.student_name", "student_submissions.revision"],
    ),
)

student_pair_scores = Table(
    "student_pair_scores", metadata,
    Column("run_id", UUID(as_uuid=True), nullable=False),
    Column("student_name", Text, nullable=False),
    Column("revision", Text, nullable=False),
    Column("pair_id", UUID(as_uuid=True), ForeignKey("pairs.pair_id"), nullable=False),
    Column("ce_win", Float, nullable=False),
    Column("ce_lose", Float, nullable=False),
    Column("tokens_win", Integer, nullable=False),
    Column("tokens_lose", Integer, nullable=False),
    Column("score", Float, nullable=False),
    Column("ce_per_message_win", JSONB),
    Column("ce_per_message_lose", JSONB),
    PrimaryKeyConstraint("run_id", "student_name", "revision", "pair_id"),
)

student_deployments = Table(
    "student_deployments", metadata,
    Column("deployment_id", Text, primary_key=True),
    Column("student_name", Text, nullable=False),
    Column("revision", Text, nullable=False),
    Column("engine", Text, nullable=False),
    Column("host", Text, nullable=False),
    Column("gpu_indices", ARRAY(Integer), nullable=False),
    Column("base_url", Text),
    Column("instance_count", Integer, nullable=False, server_default=text("0")),
    Column("status", Text, nullable=False),
    Column("consecutive_failures", Integer, nullable=False, server_default=text("0")),
    Column("next_retry_at", TIMESTAMP(timezone=True)),
    Column("last_health_check_at", TIMESTAMP(timezone=True)),
    Column("meta", JSONB),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    ForeignKeyConstraint(
        ["student_name", "revision"],
        ["student_submissions.student_name", "student_submissions.revision"],
    ),
)

service_heartbeats = Table(
    "service_heartbeats", metadata,
    Column("worker_id", Text, primary_key=True),
    Column("role", Text, nullable=False),
    Column("host", Text),
    Column("pid", Integer),
    Column("version", Text),
    Column("status", Text, nullable=False),
    Column("current_task", JSONB),
    Column("last_seen", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    Column("meta", JSONB),
)

system_events = Table(
    "system_events", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("ts", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
    Column("service", Text, nullable=False),
    Column("level", Text, nullable=False),
    Column("event", Text, nullable=False),
    Column("message", Text),
    Column("context", JSONB),
    Column("trace_id", Text),
    Column("rollout_id", UUID(as_uuid=True)),
    Column("pair_id", UUID(as_uuid=True)),
    Column("run_id", UUID(as_uuid=True)),
)

metrics_minutely = Table(
    "metrics_minutely", metadata,
    Column("ts_minute", TIMESTAMP(timezone=True), nullable=False),
    Column("metric", Text, nullable=False),
    Column("labels", JSONB, nullable=False),
    Column("value", Float, nullable=False),
    PrimaryKeyConstraint("ts_minute", "metric", "labels"),
)

audit_log = Table(
    "audit_log", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("actor", Text, nullable=False),
    Column("action", Text, nullable=False),
    Column("entity_kind", Text),
    Column("entity_id", Text),
    Column("payload", JSONB),
    Column("trace_id", Text),
    Column("ts", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
)
