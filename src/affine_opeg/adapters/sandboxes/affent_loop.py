"""affent-driven agent loop.

This adapter is the ``AgentLoopFn`` used by ``application.generate_rollouts``:

    1. Copy the prebuilt ``affent-static`` binary into the sandbox container.
    2. Stage prompt + system prompt as base64 files inside the container.
    3. ``docker exec affentctl run --workspace /app --base-url ... --trace ...``.
    4. Copy the trace JSONL back to the host, parse it.
    5. Build a ``RawTrajectory`` with family ``affent`` plus the trace events
       in ``steps``. Reward is computed via ``sandbox.run_hidden_tests`` and
       lives in ``reward_breakdown``.

The trace is uniform across teachers — affent is the single LLM client —
so the normalizer downstream does NOT need per-family branches. See
``adapters/normalizers/affent.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affine_opeg.adapters.envs.registry import get_env
from affine_opeg.domain.errors import SandboxError, TaskSourceError, TeacherError
from affine_opeg.domain.models import RawTrajectory, Task, Teacher
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("affent_loop")

# Generic fallback prompt. Per-benchmark phrasing belongs to the bound Env
# (see ``Env.system_prompt`` — SweRebenchEnv supplies its own).
DEFAULT_SYSTEM_PROMPT = """\
You are an autonomous agent operating inside a sandbox at {workspace}.

Use the available tools (shell, read_file, write_file, edit_file,
list_files) to complete the task. Stop calling tools and reply with a
short summary when you believe the work is done.
"""


def _affent_binary_path() -> str:
    for p in ("/usr/local/bin/affent-static", os.path.expanduser("~/affent-static")):
        if os.path.isfile(p):
            return p
    return "affent-static"


@dataclass(frozen=True)
class AffentLoopConfig:
    binary_path: str = ""
    max_turns: int = 80
    max_call_timeout: str = "5m"
    skip_deltas: bool = True
    host_trace_dir: str = "/tmp/afr-affent-traces"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    extra_args: tuple[str, ...] = ()


class AffentAgentLoop:
    """``AgentLoopFn`` implementation. Stateless; safe to share."""

    def __init__(self, cfg: AffentLoopConfig | None = None) -> None:
        self.cfg = cfg or AffentLoopConfig()
        self._binary = self.cfg.binary_path or _affent_binary_path()
        Path(self.cfg.host_trace_dir).mkdir(parents=True, exist_ok=True)

    async def __call__(self, teacher: Teacher, sandbox, params) -> RawTrajectory:  # type: ignore[no-untyped-def]
        """Conforms to ``AgentLoopFn``: (teacher, sandbox, params) -> RawTrajectory."""
        if sandbox.container_id is None:
            raise SandboxError("affent loop requires a container-backed sandbox")
        api_key = os.environ.get(teacher.api_key_env, "")
        if not api_key:
            raise TeacherError(f"missing api key env var: {teacher.api_key_env}")

        prompt = _build_user_prompt(sandbox.task)
        system = self._resolve_system_prompt(sandbox.task)

        workspace = sandbox.workspace_path
        await self._install_binary(sandbox.container_id)
        await self._stage_files(sandbox.container_id, prompt, system, workspace=workspace)
        rc, stderr = await self._run_affent(
            sandbox.container_id, teacher, api_key, params, workspace=workspace,
        )
        trace_text = await self._fetch_trace(sandbox.container_id)
        events = _parse_trace_events(trace_text)
        steps, agent_meta = _events_to_steps(events, user_prompt=prompt)
        agent_meta["exit_code"] = rc
        # Capture affent's stderr tail so post-hoc debugging doesn't require
        # docker access. Useful when affent fails before any SSE event lands.
        if stderr:
            agent_meta["affent_stderr_tail"] = stderr[-2000:]
        if not events:
            log.warning("affent.no_events",
                        rollout_key=f"{sandbox.task.env_name}:{sandbox.task.task_id}",
                        exit_code=rc, stderr_tail=stderr[-400:])

        reward = await sandbox.run_hidden_tests()

        return RawTrajectory(
            teacher_name=teacher.teacher_name,
            family="affent",
            steps=steps,
            reward_breakdown=reward,
            agent_meta=agent_meta,
        )

    # ---- installation / staging ---- #

    async def _install_binary(self, container_id: str) -> None:
        if not os.path.isfile(self._binary):
            raise SandboxError(f"affent binary not found at {self._binary}")
        rc, _, err = await _docker("cp", self._binary, f"{container_id}:/usr/local/bin/affentctl", timeout=30)
        if rc != 0:
            raise SandboxError(f"affent install failed: {err.decode(errors='replace')[:300]}")
        # smoke test
        rc2, _, _ = await _docker_exec(container_id, "affentctl help", timeout=10)
        if rc2 not in (0, 2):  # flag.FlagSet -h exits 2
            raise SandboxError("affent binary smoke test failed")

    def _resolve_system_prompt(self, task: Task) -> str:
        """Pull the env-specific system prompt if registered, else use the
        loop's configured fallback. ``{workspace}`` is substituted by the
        caller at stage time."""
        try:
            override = get_env(task.env_name).system_prompt(task)
        except TaskSourceError:
            override = None
        return override or self.cfg.system_prompt

    async def _stage_files(
        self, container_id: str, prompt: str, system: str, *, workspace: str,
    ) -> None:
        await _write_file_in_container(container_id, "/tmp/affent_prompt.txt", prompt)
        rendered = system.format(workspace=workspace) if "{workspace}" in system else system
        await _write_file_in_container(container_id, "/tmp/affent_system.txt", rendered)

    async def _run_affent(
        self, container_id: str, teacher: Teacher, api_key: str, params,  # type: ignore[no-untyped-def]
        *, workspace: str,
    ) -> tuple[int, str]:
        # Model identifier passed to the OpenAI-compatible endpoint. We
        # prefer ``teacher.meta['served_model']`` so PG ``teacher_name``
        # can stay a short human-readable label (e.g. ``qwen3-32b``)
        # while the endpoint may serve a fully-qualified id
        # (``Qwen/Qwen3-32B-TEE``). Fall back to teacher_name when the
        # meta key is absent for backward compat.
        served_model = (teacher.meta or {}).get("served_model") or teacher.teacher_name
        cmd = (
            f"cd {shlex.quote(workspace)} && affentctl run "
            f"--workspace {shlex.quote(workspace)} "
            f"--base-url {shlex.quote(teacher.endpoint)} "
            f"--model {shlex.quote(served_model)} "
            "--prompt @/tmp/affent_prompt.txt "
            "--system-prompt @/tmp/affent_system.txt "
            "--trace /tmp/affent_trace.jsonl "
            + ("--trace-skip-deltas " if self.cfg.skip_deltas else "")
            + f"--max-turns {params.max_steps or self.cfg.max_turns} "
            + f"--max-call-timeout {self.cfg.max_call_timeout} "
        )
        # Forward sampling knobs so N rollouts of the same (task, teacher)
        # actually diverge — temperature and seed are the producer's lever
        # for reward variance. Both are optional; negative / None means
        # "let the upstream pick its default" (matches affentctl's CLI).
        if params.temperature is not None and params.temperature >= 0:
            cmd += f"--temperature {float(params.temperature)} "
        if getattr(params, "top_p", None) is not None and params.top_p >= 0:
            cmd += f"--top-p {float(params.top_p)} "
        if getattr(params, "seed", None) is not None and params.seed >= 0:
            cmd += f"--seed {int(params.seed)} "
        cmd += "--quiet"
        if self.cfg.extra_args:
            cmd += " " + " ".join(shlex.quote(a) for a in self.cfg.extra_args)
        rc, _out, err = await _docker_exec(
            container_id, cmd,
            env={"AFFENTCTL_API_KEY": api_key},
            timeout=self.cfg.max_turns * 90,   # rough upper bound
        )
        return rc, err.decode(errors="replace")

    async def _fetch_trace(self, container_id: str) -> str:
        host_path = Path(self.cfg.host_trace_dir) / f"{container_id[:12]}.jsonl"
        rc, _out, _err = await _docker(
            "cp", f"{container_id}:/tmp/affent_trace.jsonl", str(host_path), timeout=30,
        )
        if rc == 0:
            return host_path.read_text(errors="replace")
        # Fallback to ``docker exec cat`` — useful when the trace file
        # exists but cp races with container cleanup.
        rc2, out, _ = await _docker_exec(container_id, "cat /tmp/affent_trace.jsonl 2>/dev/null || true", timeout=30)
        return out.decode(errors="replace") if rc2 == 0 else ""


# ----- helpers ----- #


def _build_user_prompt(task: Task) -> str:
    lines: list[str] = []
    if task.repo:
        lines.append(f"Repository: {task.repo}")
    if (lang := task.meta.get("repo_language")):
        lines.append(f"Language: {lang}")
    if lines:
        lines.append("")
    lines.append("## Issue / PR Description")
    lines.append("")
    lines.append(task.problem.strip())

    # Experimental: per-task hints for the cells where every baseline
    # rollout collapsed to reward=0. Goal of the A/B is to learn whether
    # a focused pointer materially improves success rate — i.e. whether
    # the model lacks capability or just the right starting context.
    hint = _HINTS_BY_TASK.get(int(task.task_id))
    if hint:
        lines.append("")
        lines.append("## Hint")
        lines.append(hint.strip())
    return "\n".join(lines)


_HINTS_BY_TASK: dict[int, str] = {
    19: (
        "This is a Python package (mbed-tools). The fix is in "
        "src/mbed_tools/build/_internal/cmake_file.py — specifically the "
        "function that builds the macro definition for the -D compile "
        "flag. Today it wraps the entire token in outer double quotes "
        "('-D\"NAME=VALUE\"') which breaks when VALUE itself contains "
        "double quotes. Produce '-DNAME=VALUE' instead; any inner quotes "
        "in VALUE must be escaped, not used as the outer wrapper. "
        "Before running tests, run `pip install -e .` in /testbed. "
        "Failing test: tests/build/_internal/test_cmake_file.py::"
        "TestRendersCMakeListsFile::test_returns_quoted_content."
    ),
    4613: (
        "The fix is in src/auditwheel/policy.py. There is a regex that "
        "detects libpython shared libraries and filters them out of the "
        "list of libs to bundle. The current pattern matches single-digit "
        "minor versions (e.g. libpython3.\\d.so) so it misses libpython3.10, "
        "libpython3.11 etc. Make the regex accept one or more digits "
        "(libpython3.\\d+.so). The failing test is "
        "tests/unit/test_policy.py::TestLddTreeExternalReferences::"
        "test_filter_libs — it expects libpython3.* in the filtered-out set."
    ),
    506: (
        "The fix is in xclim/subset.py — the subset_gridpoint function. "
        "Refactor it so the lon and lat arguments accept both a scalar and "
        "an array-like sequence of points. When given arrays the function "
        "should return a result indexed by the new 'site' dimension. Also "
        "add an `add_distance` keyword: when True the result must include a "
        "`distance` data variable (haversine distance to the gridpoint). "
        "The failing tests parametrize lon/lat as (scalar, scalar) and "
        "(list-of-2, list-of-2), and parametrize add_distance over True/False."
    ),
    3753: (
        "The fix is in aqt/metadata.py — show_list and getList chain. When "
        "the network fetch raises ArchiveConnectionError or "
        "ArchiveDownloadError during a checksum lookup, the current code "
        "swallows the exception and proceeds, leading to "
        "`output.extend(None)` further down the call chain. Re-raise the "
        "exception (or return early with an explicit error) before "
        "anything attempts to .extend() the None result. The failing test "
        "is tests/test_list.py::test_show_list_bad_connection_for_checksum."
    ),
    4299: (
        "The fix is in src/pdm/cli/commands/add.py (the dependency-group "
        "resolution path used by `pdm add -d`). When the project still has "
        "a legacy [tool.pdm.dev-dependencies] table, the code currently "
        "picks up a group name from the new [dependency-groups] table; it "
        "must instead use the legacy group name in that case. The failing "
        "test is tests/cli/test_add.py::test_add_dev_dependency_with_"
        "existing_editables_group; it adds an editable to an existing "
        "'editables' legacy group and expects subsequent `pdm add -d` to "
        "extend that same group rather than create a duplicate."
    ),
    # ---- Round-2 hints: partial-success cells (variance-injection) ----
    5741: (
        "The bug is in the Postgres dialect of sqlglot. The keyword "
        "'character varying' (and bare 'character') must be tokenised / "
        "parsed as VARCHAR / CHAR — Postgres treats them as exact "
        "aliases. The failing test exercises BOTH inline column types "
        "and CAST/NULL expressions (NULL::character varying), so make "
        "sure the recogniser handles 'character varying' anywhere a "
        "data type can appear, with or without a length specifier. Look "
        "at sqlglot/dialects/postgres.py — likely a TOKENS mapping or "
        "keyword table."
    ),
    6018: (
        "The bug is in sqlglot's BigQuery parser. The ARRAY(...) "
        "constructor must accept a SELECT subquery (e.g. "
        "ARRAY(SELECT x FROM UNNEST([0,1]) AS x)), not just a literal "
        "expression list. Look in sqlglot/dialects/bigquery.py or "
        "sqlglot/parser.py where ARRAY is handled: when the token after "
        "ARRAY( is SELECT, parse the body as a subquery and wrap it as "
        "an array. Cover both the simple `ARRAY(SELECT x FROM ...)` "
        "and the projection variant `ARRAY(SELECT x * 2 FROM ...)`."
    ),
    1105: (
        "These ansible-builder v3 failures are about Containerfile "
        "header defaults. In ansible_builder/main.py (or wherever the v3 "
        "Containerfile template is emitted), v3 must default to "
        "`WORKDIR /runner` and `USER 1000:0` (non-root user, primary "
        "group 0 for OpenShift compatibility). Also add an opt-in "
        "`--relax-passwd-perms` flag — when set it emits "
        "`RUN chmod g=u /etc/passwd /etc/group`. Touch only the v3 code "
        "path; v1/v2 defaults must stay as they are. The failing tests "
        "live in test/integration/test_create.py::test_v3_*."
    ),
}


async def _docker(*args: str, timeout: int = 30, input_: bytes | None = None) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE if input_ is not None else None,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=input_), timeout=timeout)
    except asyncio.TimeoutError as e:
        proc.kill(); await proc.wait()
        raise SandboxError(f"docker {args[0]} timed out after {timeout}s") from e
    return proc.returncode or 0, stdout, stderr


async def _docker_exec(
    container_id: str, script: str,
    *, env: dict[str, str] | None = None, timeout: int = 60,
) -> tuple[int, bytes, bytes]:
    args = ["exec"]
    if env:
        for k, v in env.items():
            args.extend(["-e", f"{k}={v}"])
    args.extend([container_id, "bash", "-c", script])
    return await _docker(*args, timeout=timeout)


async def _write_file_in_container(container_id: str, path: str, content: str) -> None:
    b64 = base64.b64encode(content.encode()).decode()
    rc, _out, err = await _docker_exec(
        container_id, f"echo '{b64}' | base64 -d > {shlex.quote(path)}", timeout=15,
    )
    if rc != 0:
        raise SandboxError(f"failed to write {path}: {err.decode(errors='replace')[:200]}")


def _parse_trace_events(jsonl_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _events_to_steps(
    events: list[dict[str, Any]], *, user_prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fold affent SSE events into per-turn ``steps`` and aggregate meta.

    Each ``step`` in the output is one normalized turn:
        - ``{"kind": "user", "text": "..."}``
        - ``{"kind": "assistant", "text": "...", "reasoning": "...",
              "tool_calls": [{"id", "name", "arguments"}]}``
        - ``{"kind": "tool_result", "tool_call_id": "...", "text": "...",
              "exit_code": int}``
    """
    steps: list[dict[str, Any]] = []
    cur_assistant_text: list[str] = []
    cur_assistant_thinking: list[str] = []
    cur_tool_calls: list[dict[str, Any]] = []
    tokens_in = tokens_out = model_calls = 0
    errors: list[str] = []
    saw_user = False

    def _flush_assistant() -> None:
        if cur_assistant_text or cur_assistant_thinking or cur_tool_calls:
            steps.append({
                "kind": "assistant",
                "text": "".join(cur_assistant_text).strip() or None,
                "reasoning": "".join(cur_assistant_thinking).strip() or None,
                "tool_calls": list(cur_tool_calls),
            })
        cur_assistant_text.clear()
        cur_assistant_thinking.clear()
        cur_tool_calls.clear()

    for ev in events:
        t = ev.get("type", "")
        d = ev.get("data") or {}
        if t == "user.message":
            _flush_assistant()
            saw_user = True
            steps.append({"kind": "user", "text": d.get("text", "")})
        elif t == "thinking.delta":
            cur_assistant_thinking.append(d.get("delta", ""))
        elif t == "thinking.done":
            full = d.get("text")
            if full is not None:
                cur_assistant_thinking[:] = [full]
        elif t == "message.delta":
            cur_assistant_text.append(d.get("delta", ""))
        elif t == "message.done":
            full = d.get("text")
            if full is not None:
                cur_assistant_text[:] = [full]
            model_calls += 1
        elif t == "tool.request":
            cur_tool_calls.append({
                "id": d.get("call_id", ""),
                "name": d.get("tool", ""),
                "arguments": d.get("args", {}),
            })
        elif t == "tool.result":
            _flush_assistant()
            steps.append({
                "kind": "tool_result",
                "tool_call_id": d.get("call_id", ""),
                "text": d.get("result_summary", ""),
                "exit_code": int(d.get("exit_code", 0)),
            })
        elif t == "usage":
            tokens_in += int(d.get("input_tokens", 0))
            tokens_out += int(d.get("output_tokens", 0))
        elif t == "error":
            errors.append(str(d.get("message", "")))
    _flush_assistant()

    # Guarantee a leading user step even if affent never emitted one.
    if not saw_user:
        steps.insert(0, {"kind": "user", "text": user_prompt})

    meta = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model_calls": model_calls,
        "errors": errors,
    }
    return steps, meta
