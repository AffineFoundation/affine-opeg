"""Docker-backed sandbox.

A pure container shell: brings a docker image up, exposes ``workspace_path``
and a narrow exec/write/apply-patch surface, tears the container down on
exit. All benchmark-specific logic (how to grade, what counts as the agent
patch, which workspace path to mount) is delegated to an
:class:`affine_opeg.domain.ports.env.Env` adapter resolved per-task.

Steps inside ``setup``:

    1. ``docker pull`` the task image (cached on the host).
    2. ``docker run -d --memory`` to start a long-sleeping container.
    3. Prepare the workspace (mkdir, materialise ``setup_files``, ensure
       git is installed and the dir is a repo).
    4. Apply a network blocklist + sanitize git so the agent cannot reach
       the internet or recover the gold patch from git history.

``run_hidden_tests`` / ``extract_patch`` forward to the per-task
:class:`Evaluator` / :class:`PatchExtractor` produced by the bound Env.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shlex
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from affine_opeg.adapters.envs.registry import get_env
from affine_opeg.domain.errors import SandboxError, SandboxTimeout
from affine_opeg.domain.models import Task
from affine_opeg.domain.ports.env import Env, Evaluator, PatchExtractor
from affine_opeg.infrastructure.logging import get_logger

log = get_logger("sandbox.docker")

_DOCKER_PULL_TIMEOUT = 600
_DEFAULT_MEMORY = "4g"
_DEFAULT_TIMEOUT_S = 1800
_PULL_RETRIES = 4
_PULL_BACKOFF_S = 10

# Global semaphore to serialise ``docker pull`` calls across all
# concurrent sandboxes in this process. containerd's overlayfs
# snapshotter does NOT handle concurrent layer extraction gracefully —
# two pulls touching the same base layer race on the snapshotter's
# work dir and one (or both) fails with "failed to extract layer ...
# to overlayfs as extract-...". Keeping pulls serial avoids that
# entirely; cost is at most a few minutes per fresh image, which is
# unavoidable anyway.
_PULL_LOCK = asyncio.Semaphore(1)

# Minimal network blocklist: prevent agent egress while leaving loopback open.
_NETWORK_BLOCKLIST = r"""
set -e
iptables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
iptables -A OUTPUT -d 172.17.0.0/16 -j ACCEPT 2>/dev/null || true
iptables -A OUTPUT -j REJECT 2>/dev/null || true
""".strip()

# Strip git remotes and rewrite history sentinels so the agent can't
# discover the upstream patch.
_SANITIZE_GIT = r"""
set -e
git remote remove origin 2>/dev/null || true
git config --unset-all remote.origin.url 2>/dev/null || true
git gc --quiet --prune=now 2>/dev/null || true
""".strip()


class DockerSandbox:
    """Single per-rollout docker container.

    Implements :class:`affine_opeg.domain.ports.env.SandboxExec` so the
    injected Evaluator / PatchExtractor can call into it without knowing
    about docker.
    """

    def __init__(
        self,
        task: Task,
        *,
        docker_image: str,
        evaluator: Evaluator,
        patch_extractor: PatchExtractor,
        workspace_path: str = "/app",
        memory: str = _DEFAULT_MEMORY,
        max_lifetime_s: int = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.task = task
        self.workspace_path = workspace_path
        self._image = docker_image
        self._memory = memory
        self._max_lifetime = max_lifetime_s
        self._evaluator = evaluator
        self._patch_extractor = patch_extractor
        self.container_id: str | None = None

    # ---- lifecycle ---- #

    async def _docker(self, *args: str, timeout: int = 30, input_: bytes | None = None) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if input_ is not None else None,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_), timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise SandboxTimeout(f"docker {args[0]} timed out after {timeout}s") from e
        return proc.returncode or 0, stdout, stderr

    async def exec(self, script: str, *, timeout: int = 60) -> tuple[int, str, str]:
        """SandboxExec.exec — run ``bash -c script`` inside the container."""
        if self.container_id is None:
            raise SandboxError("container not started")
        rc, out, err = await self._docker(
            "exec", self.container_id, "bash", "-c", script, timeout=timeout,
        )
        return rc, out.decode(errors="replace"), err.decode(errors="replace")

    async def _pull(self) -> None:
        # Fast path: image already fully present locally → skip the
        # pull entirely. ``docker image inspect`` is a couple of ms and
        # bypasses the serialisation lock below for already-cached
        # tasks (the common case once warm).
        rc_inspect, _, _ = await self._docker(
            "image", "inspect", self._image, timeout=10,
        )
        if rc_inspect == 0:
            return

        # Cold path: serialise the actual pull via _PULL_LOCK so
        # concurrent sandboxes don't fight containerd's snapshotter.
        last_err = b""
        async with _PULL_LOCK:
            # Re-check inside the lock — a peer may have just pulled
            # the same image while we waited.
            rc_inspect2, _, _ = await self._docker(
                "image", "inspect", self._image, timeout=10,
            )
            if rc_inspect2 == 0:
                return
            for attempt in range(1, _PULL_RETRIES + 1):
                rc, _out, err = await self._docker(
                    "pull", self._image, timeout=_DOCKER_PULL_TIMEOUT,
                )
                if rc == 0:
                    return
                last_err = err
                log.warning(
                    "sandbox.pull_retry",
                    image=self._image, attempt=attempt,
                    error=err.decode(errors="replace")[:200],
                )
                await asyncio.sleep(_PULL_BACKOFF_S * attempt)
        # All retries exhausted — final fallback to local inspect (image
        # may have landed partially) before surrendering.
        rc_final, _, _ = await self._docker(
            "image", "inspect", self._image, timeout=10,
        )
        if rc_final != 0:
            raise SandboxError(
                f"failed to pull {self._image} after {_PULL_RETRIES} attempts: "
                f"{last_err.decode(errors='replace')[:300]}"
            )

    async def _start(self) -> None:
        name = f"afr-rb-{os.urandom(4).hex()}"
        # ``docker run -d`` is mostly instant *if* the image is fully
        # present locally. When the daemon has to extract a multi-GB
        # image (swerebench/sweb.eval.x86_64.*) this can take a minute.
        # 120s leaves headroom without masking real hangs.
        rc, out, err = await self._docker(
            "run", "-d",
            "--name", name,
            "--memory", self._memory,
            "--entrypoint", "",
            self._image,
            "sleep", str(self._max_lifetime + 300),
            timeout=120,
        )
        if rc != 0:
            raise SandboxError(f"failed to start container: {err.decode(errors='replace')[:300]}")
        self.container_id = out.decode().strip()

    async def _prepare(self) -> None:
        # Block egress + sanitize git. Failure here is informational, not fatal:
        # we want the sandbox up even on iptables-less hosts (developer laptops).
        try:
            await self.exec(_NETWORK_BLOCKLIST, timeout=10)
        except SandboxError:
            log.debug("docker.sandbox.netblock_failed")
        try:
            await self.exec(
                f"cd {shlex.quote(self.workspace_path)} && " + _SANITIZE_GIT,
                timeout=30,
            )
        except SandboxError:
            log.warning("docker.sandbox.sanitize_git_failed")

    async def setup(self) -> None:
        await self._pull()
        await self._start()
        await self._prepare_workspace()
        await self._prepare()

    async def _prepare_workspace(self) -> None:
        """Ensure ``workspace_path`` exists, write any ``setup_files``,
        ensure ``git`` is installed and the workspace is a repo.

        Real SWE-rebench images already have the repo and git baked in, so
        ``setup_files`` is empty and ``git init`` is a no-op for them. For
        from-scratch smoke tasks (python:3.11-slim, etc.) this is what
        bootstraps the workspace.
        """
        rc, _out, err = await self.exec(
            f"mkdir -p {shlex.quote(self.workspace_path)}", timeout=10,
        )
        if rc != 0:
            raise SandboxError(f"failed to create workspace: {err[:300]}")

        setup_files = (self.task.hidden_tests or {}).get("setup_files") or {}
        for rel, content in setup_files.items():
            full = (self.workspace_path.rstrip("/") + "/" + rel.lstrip("/"))
            await self.write_file(full, content)

        # ``git`` is required so extract_patch produces a diff. On slim
        # base images it is missing; install once via apt/yum/apk best-effort.
        await self.exec(
            "command -v git >/dev/null 2>&1 || ("
            "apt-get update -qq >/dev/null 2>&1 && "
            "apt-get install -y -qq --no-install-recommends git >/dev/null 2>&1 || "
            "apk add --no-cache git >/dev/null 2>&1 || "
            "yum install -y -q git >/dev/null 2>&1 || true)",
            timeout=120,
        )
        await self.exec(
            f"cd {shlex.quote(self.workspace_path)} && "
            "(git rev-parse --git-dir >/dev/null 2>&1 || ("
            "git init -q && "
            "git -c user.email=a@f -c user.name=afr add -A && "
            "git -c user.email=a@f -c user.name=afr commit -q --allow-empty -m init"
            "))",
            timeout=30,
        )

        # Task-supplied setup commands (install pytest, fetch fixtures, ...).
        # Each runs inside ``workspace_path``; non-zero is logged but not fatal
        # — the test command will surface the real failure later if it matters.
        for cmd in (self.task.hidden_tests or {}).get("setup_commands") or []:
            rc, _out, err = await self.exec(
                f"cd {shlex.quote(self.workspace_path)} && {cmd}",
                timeout=300,
            )
            if rc != 0:
                log.warning("docker.sandbox.setup_command_failed",
                            cmd=cmd, stderr=err[:300])

    async def write_file(self, container_path: str, content: str) -> None:
        """SandboxExec.write_file — materialise ``content`` at ``container_path``."""
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        rc, _out, err = await self.exec(
            f"mkdir -p $(dirname {shlex.quote(container_path)}) && "
            f"echo {b64} | base64 -d > {shlex.quote(container_path)}",
            timeout=15,
        )
        if rc != 0:
            raise SandboxError(f"failed to write {container_path}: {err[:200]}")

    async def apply_patch(self, container_path: str, patch_text: str) -> None:
        """SandboxExec.apply_patch — pipe a unified diff through ``patch -p1``."""
        if self.container_id is None:
            raise SandboxError("container not started")
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", self.container_id, "bash", "-c",
            f"cd {shlex.quote(self.workspace_path)} && (patch -p1 || git apply --whitespace=fix)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(
                proc.communicate(input=patch_text.encode()), timeout=30,
            )
        except asyncio.TimeoutError as e:
            proc.kill(); await proc.wait()
            raise SandboxTimeout("apply_patch timed out") from e

    async def teardown(self) -> None:
        if self.container_id:
            try:
                await self._docker("rm", "-f", self.container_id, timeout=30)
            except Exception:  # noqa: BLE001
                pass
            self.container_id = None

    # ---- evaluation surface (called by application layer) ---- #

    async def extract_patch(self) -> str:
        return await self._patch_extractor.extract(self)

    async def run_hidden_tests(self) -> dict[str, Any]:
        return await self._evaluator.evaluate(self)


class DockerSandboxFactory:
    """Acquires DockerSandbox instances under a concurrency cap.

    Per-task wiring lives entirely on the bound Env: the factory itself
    knows nothing about pytest, conda envs, or SWE-rebench image quirks.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 32,
        env_resolver: Any = None,
    ) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        # Indirection lets tests inject a stub registry.
        self._resolve: Any = env_resolver or get_env

    @asynccontextmanager
    async def acquire(self, task: Task) -> AsyncIterator[DockerSandbox]:
        async with self._sem:
            env: Env = self._resolve(task.env_name)
            sb = DockerSandbox(
                task,
                docker_image=env.image(task),
                evaluator=env.evaluator(task),
                patch_extractor=env.patch_extractor(task),
                workspace_path=env.workspace_path(task),
            )
            await sb.setup()
            try:
                yield sb
            finally:
                await sb.teardown()
