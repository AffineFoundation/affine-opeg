"""Configuration loading.

Layered precedence (later overrides earlier):
    1. defaults in pydantic models
    2. conf/base.yaml
    3. conf/env/{AFR_ENV}.yaml
    4. environment variables (AFR_* prefix)

Secrets are never stored in yaml. yaml fields like ``api_key_env`` name the env
var to read at runtime. The settings object is loaded once per process and
treated as immutable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONF_DIR = Path(os.environ.get("AFR_CONF_DIR", "conf"))


class DbConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "afr"
    password: SecretStr = SecretStr("afr")
    name: str = "afr"
    auth: Literal["password", "iam"] = "password"
    ssl: Literal["disable", "prefer", "require"] = "prefer"
    pool_size: int = 10
    max_overflow: int = 10
    pool_recycle: int = 600
    pool_pre_ping: bool = True
    statement_timeout_ms: int = 60_000


class BlobConfig(BaseModel):
    """R2 / S3-compatible store used by the publisher + promoter.

    Two buckets to physically isolate fresh from mature data:

    * ``bucket`` (private): the publisher writes per-cell parquets +
      manifest here as soon as a cell is frozen. Only the validator
      (which holds AK/SK) can read this bucket. Fresh data stays here
      during the maturation window.
    * ``public_bucket``: the promoter copies a cell's shard + appends
      the manifest entry here only after ``mature_at`` (committed_at
      + maturation_window_s) has elapsed. Designed to be served
      anonymously (the bucket has public read enabled in the R2
      console) so miners can pull mature shards without credentials.

    The generator never writes to R2 — rollouts persist inline in
    ``PG.rollouts.extra_compressed``. Both buckets are written from
    the ``af servers generator-publisher`` service.
    """

    endpoint: str | None = None
    bucket: str = "affine-distill-v2-private"          # fresh side
    public_bucket: str = "affine-distill-v2-public"    # mature mirror
    access_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    region: str = "us-east-1"
    prefix: str = ""


class AwsConfig(BaseModel):
    region: str = "us-west-2"


class RolloutConfig(BaseModel):
    samples_per_pair: int = 8
    temperature_min: float = 0.7
    temperature_max: float = 1.2
    max_steps: int = 40
    # Concurrency cap for the *sandbox* (SWE/affent) backend. Each episode
    # is a docker sandbox running pip/compile/test, so this is the
    # memory-bound knob — keep it within the host's RAM budget.
    max_concurrent_episodes: int = 32
    # Concurrency cap for the container-free verifiers backend. These
    # episodes only make remote LLM calls (NullSandbox), so they cost ~0
    # local memory and can run far higher than the sandbox cap. 0 disables
    # the verifiers backend entirely (sandbox-only producer).
    verifiers_concurrency: int = 0
    per_teacher_concurrency: int = 16
    blob_archive_async: bool = True


class EvalConfig(BaseModel):
    default_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    default_max_seq_len: int = 32_768
    max_batch_tokens: int = 65_536
    mask_reasoning: bool = True


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # ``json`` is a reserved-ish name on pydantic BaseModel; expose as
    # ``as_json`` instead. yaml keeps the friendlier ``json: true`` via alias.
    as_json: bool = Field(default=True, alias="json")
    db_sink_min_level: Literal["INFO", "WARNING", "ERROR"] = "WARNING"

    model_config = {"populate_by_name": True}


class AppConfig(BaseSettings):
    """Top-level configuration, populated from layered sources."""

    model_config = SettingsConfigDict(
        env_prefix="AFR_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "local"
    service: str = "api"
    version: str = Field(default_factory=lambda: os.environ.get("AFR_GIT_SHA", "dev"))

    db: DbConfig = Field(default_factory=DbConfig)
    blob: BlobConfig = Field(default_factory=BlobConfig)
    aws: AwsConfig = Field(default_factory=AwsConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"Expected mapping at top of {path}, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _env_overrides(prefix: str = "AFR_", delim: str = "__") -> dict[str, Any]:
    """Materialise AFR_* env vars into a nested dict mirroring AppConfig.

    ``AFR_DB__HOST=foo`` becomes ``{"db": {"host": "foo"}}``. Both flat
    (single segment) and nested (``__`` delimited) keys are honoured.
    """
    out: dict[str, Any] = {}
    for raw_key, val in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        path = raw_key[len(prefix):].lower().split(delim)
        cursor = out
        for segment in path[:-1]:
            nxt = cursor.get(segment)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[segment] = nxt
            cursor = nxt
        cursor[path[-1]] = val
    return out


@lru_cache(maxsize=1)
def load_config(env: str | None = None) -> AppConfig:
    """Load and cache configuration. Call once at process startup.

    Precedence (lowest → highest):
        1. pydantic model defaults
        2. conf/base.yaml
        3. conf/env/{AFR_ENV}.yaml
        4. AFR_* environment variables
    """
    env_name = env or os.environ.get("AFR_ENV", "local")
    base = _load_yaml(CONF_DIR / "base.yaml")
    env_overlay = _load_yaml(CONF_DIR / "env" / f"{env_name}.yaml")
    merged = _deep_merge(base, env_overlay)
    merged = _deep_merge(merged, _env_overrides())
    merged.setdefault("env", env_name)
    return AppConfig(**merged)
