"""Docker sandboxes start, stop, die, and clean up as `Environment` promises."""

from __future__ import annotations

import time

import pytest
from support import IMAGE, sandbox_containers

from assignment.env import Environment
from assignment.sandbox.dockercli import DockerError, docker

pytestmark = [pytest.mark.docker, pytest.mark.usefixtures("no_leaked_containers")]


def container_exists(name: str) -> bool:
    return name in sandbox_containers()


def test_sandbox_variable_selects_docker_and_stop_is_idempotent(monkeypatch):
    monkeypatch.setenv("SANDBOX", "docker")
    env = Environment(image=IMAGE)
    name = env.deployment.container
    assert env.backend.name == "docker"
    assert container_exists(name) and env.is_alive()

    env.stop()
    assert not container_exists(name)
    assert not env.is_alive()
    env.stop()

    with pytest.raises(RuntimeError, match="no longer running"):
        env.execute("true")


def test_context_manager_removes_the_container_on_error():
    with pytest.raises(KeyError), Environment(image=IMAGE, backend="docker") as env:
        name = env.deployment.container
        raise KeyError("boom")
    assert not container_exists(name)


def test_a_killed_sandbox_is_terminal():
    with Environment(image=IMAGE, backend="docker") as env:
        docker("kill", env.deployment.container)
        with pytest.raises(RuntimeError, match="no longer running"):
            env.execute("echo unreachable")


def test_the_container_is_reclaimed_after_deployment_timeout():
    # Long enough for startup to finish first even on a busy machine; a lifetime
    # shorter than startup kills the container while Environment() waits on it.
    lifetime = 15
    env = Environment(image=IMAGE, backend="docker", deployment_timeout=lifetime)
    name = env.deployment.container
    try:
        assert env.execute("echo alive")["returncode"] == 0
        deadline = time.monotonic() + lifetime + 30
        while container_exists(name) and time.monotonic() < deadline:
            time.sleep(0.5)
        # Gone without stop(): what saves a crashed caller from leaking it.
        assert not container_exists(name)
        assert not env.is_alive()
    finally:
        env.stop()


def test_a_failed_startup_leaves_no_container():
    with pytest.raises(RuntimeError, match="did not become reachable"):
        Environment(image=IMAGE, backend="docker", startup_timeout=0)


def test_sandbox_hardening():
    with Environment(image=IMAGE, backend="docker") as env:
        assert env.execute("command -v swerex-remote")["returncode"] != 0
        assert env.execute("command -v python")["output"] == "/usr/local/bin/python\n"
        written = env.execute("touch /opt/swerex/tampered")
        assert written["returncode"] != 0 and "Read-only" in written["output"]
        status = env.execute("grep NoNewPrivs /proc/self/status")["output"]
        assert status.split() == ["NoNewPrivs:", "1"]

        published = docker("port", env.deployment.container).splitlines()
        assert published and all("-> 127.0.0.1:" in line for line in published)


def test_a_remote_docker_daemon_is_refused(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://someone@build-box.example")
    with pytest.raises(DockerError, match="this machine's Docker daemon only"):
        Environment(image=IMAGE, backend="docker")
