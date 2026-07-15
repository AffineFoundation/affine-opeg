"""DockerSandbox.exec: separate script failure from container death.

A dead container (OOM-killed, evicted, reaped) must surface as SandboxError,
not as the script's own rc/stderr — otherwise the pytest grader records
reward=0 for an infra failure and the rollout is published as status=ok.
"""
import asyncio

import pytest

from affine_opeg.adapters.sandboxes.docker_sandbox import DockerSandbox
from affine_opeg.domain.errors import SandboxError
from affine_opeg.domain.models import Task


DAEMON_ERR = b"Error response from daemon: container abc123 is not running\n"


def _sandbox(responses):
    """Build a DockerSandbox whose _docker is stubbed.

    ``responses`` maps the docker subcommand ("exec" / "inspect") to a
    canned (rc, stdout, stderr) tuple. Calls are recorded for assertions.
    """
    task = Task(
        env_name="swe-rebench", task_id=1, repo="r", base_commit="c",
        problem="p", hidden_tests={},
    )
    sb = DockerSandbox(
        task, docker_image="img", evaluator=None, patch_extractor=None,
    )
    sb.container_id = "abc123"
    calls = []

    async def fake_docker(*args, timeout=30, input_=None):
        calls.append(args)
        return responses[args[0]]

    sb._docker = fake_docker
    return sb, calls


def test_exec_success_skips_liveness_probe():
    sb, calls = _sandbox({"exec": (0, b"out", b"")})
    rc, out, err = asyncio.run(sb.exec("true"))
    assert (rc, out, err) == (0, "out", "")
    assert [a[0] for a in calls] == ["exec"]


def test_exec_script_failure_with_live_container_passes_through():
    sb, calls = _sandbox({
        "exec": (1, b"1 failed", b"boom"),
        "inspect": (0, b"true\n", b""),
    })
    rc, out, err = asyncio.run(sb.exec("pytest"))
    assert rc == 1 and out == "1 failed" and err == "boom"
    assert [a[0] for a in calls] == ["exec", "inspect"]


def test_exec_raises_when_container_died():
    sb, _ = _sandbox({
        "exec": (1, b"", DAEMON_ERR),
        "inspect": (0, b"false\n", b""),
    })
    with pytest.raises(SandboxError, match="container died mid-run"):
        asyncio.run(sb.exec("pytest"))


def test_exec_raises_when_container_removed():
    # ``docker inspect`` on a removed container fails outright.
    sb, _ = _sandbox({
        "exec": (1, b"", DAEMON_ERR),
        "inspect": (1, b"", b"Error: No such object: abc123"),
    })
    with pytest.raises(SandboxError, match="container died mid-run"):
        asyncio.run(sb.exec("pytest"))
