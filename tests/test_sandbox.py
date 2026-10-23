"""Fast checks for the sandbox plumbing that need no Docker daemon or Modal.

Docker calls are replaced by recording fakes of `subprocess.run`, and sandboxes
by fake deployments, so these run offline in the default `pytest` run.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from assignment import cli
from assignment import env as env_module
from assignment.env import AssignmentDockerDeployment, Environment
from assignment.utils import image as image_module


@pytest.fixture
def docker_runs(monkeypatch):
    """Record every `subprocess.run` call made by `assignment.env`, running none."""
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(env_module.subprocess, "run", run)
    return calls


def test_start_command_kills_the_server_at_the_deployment_timeout(docker_runs):
    deployment = AssignmentDockerDeployment(image="img:1", deployment_timeout=90.7)

    command = deployment._get_swerex_start_cmd("secret")

    assert command[:4] == ["timeout", "-s", "KILL", "90"]
    assert command[4:6] == ["/bin/sh", "-c"]
    assert "--auth-token secret" in command[6]
    assert "pipx run swe-rex" in command[6]


def test_with_pipx_builds_and_returns_a_pipx_tag(docker_runs):
    deployment = AssignmentDockerDeployment(image="img:1", container_runtime="podman")

    assert deployment._config.image == "img:1-pipx"
    [(args, kwargs)] = docker_runs
    assert args == ["podman", "build", "-q", "-t", "img:1-pipx", "-"]
    assert kwargs["input"].startswith("FROM img:1\n")
    assert "apt-get install -y pipx" in kwargs["input"]


def test_without_pipx_the_image_is_used_as_is(docker_runs):
    deployment = AssignmentDockerDeployment(image="img:1", install_pipx=False)

    assert deployment._config.image == "img:1"
    assert docker_runs == []


def test_modal_sandbox_kwargs_are_rejected_on_docker():
    with pytest.raises(ValueError, match="modal_sandbox_kwargs"):
        Environment(sandbox="docker", modal_sandbox_kwargs={"cpu": 2})


def test_unknown_sandbox_is_rejected():
    with pytest.raises(ValueError, match="Unknown sandbox"):
        Environment(sandbox="kubernetes")


def test_tunnel_url_is_not_available_on_docker():
    environment = Environment.__new__(Environment)
    environment.sandbox = "docker"

    with pytest.raises(NotImplementedError):
        environment.tunnel_url(8000)


class FakeRuntime:
    """Answers every command with a failure, so `activate_conda_env` raises."""

    async def execute(self, command):
        return SimpleNamespace(stdout="", stderr="", exit_code=1)


class FakeDeployment:
    """Stands in for both deployments, recording whether it was stopped."""

    instances: ClassVar[list["FakeDeployment"]] = []

    def __init__(self, **kwargs):
        self.runtime = FakeRuntime()
        self._sandbox = None
        self.stopped = False
        FakeDeployment.instances.append(self)

    async def start(self):
        pass

    async def is_alive(self):
        pass

    async def stop(self):
        self.stopped = True


@pytest.mark.parametrize(
    ("sandbox", "deployment"),
    [("docker", "AssignmentDockerDeployment"), ("modal", "AssignmentModalDeployment")],
)
def test_init_failure_after_start_stops_the_sandbox(monkeypatch, sandbox, deployment):
    """A constructor that fails after the sandbox started must not leak it."""
    FakeDeployment.instances = []
    monkeypatch.setattr(env_module, deployment, FakeDeployment)

    with pytest.raises(FileNotFoundError):
        Environment(sandbox=sandbox, conda_env="missing")

    [started] = FakeDeployment.instances
    assert started.stopped


def test_docker_build_context_holds_exactly_what_modal_would_upload(monkeypatch, tmp_path):
    source = tmp_path / "source"
    for relative in [
        "app.py",
        "pkg/module.py",
        "pkg/module.pyc",
        "pkg/__pycache__/module.cpython-312.pyc",
        ".git",
        ".venv/bin/python",
        ".env",
    ]:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    contexts = []
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if "-f" in args:
            # The context is a temporary directory, gone once the build returns.
            context = Path(args[-1])
            contexts.append(
                {p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()}
            )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(image_module.subprocess, "run", run)
    task = SimpleNamespace(
        id="demo",
        base_commit="0123456789abcdef",
        source=source,
        dockerfile=tmp_path / "Dockerfile",
        pins=("b==2", "a==1"),
    )

    tag = image_module._build_docker_image(task, force_build=False)

    assert contexts == [{"app.py", "pkg/module.py"}]
    assert tag == "assignment-testbed/demo:0123456789ab"
    assert calls[-1] == ["docker", "build", "-t", tag, "-"]


class StopAtEnvironment(Exception):
    """Raised by the fake Environment to end the CLI run once its arguments are known."""


@pytest.mark.parametrize(("flags", "expected"), [([], "docker"), (["--sandbox", "modal"], "modal")])
def test_cli_passes_sandbox_to_the_image_build_and_environment(monkeypatch, flags, expected):
    seen = {}

    def build_testbed_image(task, sandbox):
        seen["image"] = sandbox
        return "image"

    def environment(**kwargs):
        seen["environment"] = kwargs["sandbox"]
        raise StopAtEnvironment

    monkeypatch.setattr(cli.Task, "load", lambda path: SimpleNamespace())
    monkeypatch.setattr(cli, "build_testbed_image", build_testbed_image)
    monkeypatch.setattr(cli, "Environment", environment)
    monkeypatch.setattr(
        sys,
        "argv",
        ["assignment-code-agent", "--model", "m", "--trajectory", "t.json", "--patch-output", "p"]
        + flags,
    )

    with pytest.raises(StopAtEnvironment):
        cli.run_code_agent()

    assert seen == {"image": expected, "environment": expected}
