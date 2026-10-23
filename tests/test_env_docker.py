"""Tests for the local Docker sandbox environment, mirroring `test_env.py`.

These start real containers on the local Docker daemon, so they are slow.
Run with `make test-docker` (`pytest -m docker`).
"""

import subprocess
import time

import pytest

from assignment.env import Environment

pytestmark = pytest.mark.docker

IMAGE = "python:3.12-slim"
# Environment runs containers from the pipx layer it builds on top of IMAGE.
CONTAINER_IMAGE = f"{IMAGE}-pipx"


def containers() -> list[str]:
    """Containers from the test image in any state, including ones being removed."""
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", f"ancestor={CONTAINER_IMAGE}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()


def wait_until_no_containers(timeout: float = 30) -> list[str]:
    """Wait for --rm to finish removing killed containers; return any that remain."""
    deadline = time.monotonic() + timeout
    while (left := containers()) and time.monotonic() < deadline:
        time.sleep(1)
    return left


@pytest.fixture(scope="module", autouse=True)
def no_leftover_containers():
    """Every test starts and ends with no container from the test image."""
    assert not containers(), (
        f"Containers from {CONTAINER_IMAGE} are already present; remove them first."
    )
    yield
    assert not wait_until_no_containers(), "A test left a container behind."


@pytest.fixture(scope="module")
def env():
    """One container shared by the command tests, torn down cleanly afterwards."""
    env = Environment(image=IMAGE, sandbox="docker")
    assert env.is_alive()
    assert len(containers()) == 1, "Launched one container but found more than one."

    yield env

    env.stop()
    assert not env.is_alive()
    assert not wait_until_no_containers(), "The stopped container was not removed."


def test_shell_command(env):
    """A bare command string runs through the shell, via the shell=True default."""
    output = env.execute("echo 'hello, world'")
    assert output["returncode"] == 0
    assert output["output"] == "hello, world\n"


def test_argv_command(env):
    """An argv list runs unshelled when shell=False is passed explicitly."""
    output = env.execute(["echo", "hello, world"], shell=False)
    assert output["returncode"] == 0
    assert output["output"] == "hello, world\n"


def test_failing_command(env):
    """A failing command reports its exit code, with the traceback in the output."""
    output = env.execute("python -c 'print(\"hello, world\"); raise Exception(\"test exception\")'")
    assert output["returncode"] != 0
    assert "hello, world\n" in output["output"]
    assert "test exception" in output["output"]


def test_platform_is_read_from_the_container(env):
    assert env.system == "Linux"


def test_deployment_timeout_expiry_then_stop(env):
    """A container is killed at deployment_timeout, and stopping it afterwards is clean.

    Once the container has died, closing the SWE-ReX runtime fails because
    nothing listens any more; stop() must still not raise and must reap it.
    """
    expiring = Environment(image=IMAGE, sandbox="docker", deployment_timeout=60)
    try:
        deadline = time.monotonic() + 120
        while expiring.is_alive() and time.monotonic() < deadline:
            time.sleep(1)
        assert not expiring.is_alive(), "The container outlived its deployment_timeout."

        with pytest.raises(RuntimeError, match="no longer running"):
            expiring.execute("true")
    finally:
        expiring.stop()

    assert not expiring.is_alive()
    assert expiring.deployment._container_name is None


def test_init_failure_after_start_leaves_no_container(env):
    """A constructor that fails once the container is up must not leak it."""
    before = set(containers())

    with pytest.raises(FileNotFoundError):
        Environment(image=IMAGE, sandbox="docker", conda_env="missing")

    deadline = time.monotonic() + 30
    while set(containers()) - before and time.monotonic() < deadline:
        time.sleep(1)
    assert not set(containers()) - before, "The failed Environment left its container running."
