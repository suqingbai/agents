"""Shared fixtures for the sandbox backend tests.

Tests marked `docker` start real containers on this machine's Docker and are
skipped, not failed, when Docker is not usable. Tests marked `modal` start real
Modal sandboxes (billable). Neither runs under a plain `make test`.
"""

from __future__ import annotations

import pytest
from support import sandbox_containers

from assignment.sandbox.dockercli import Daemon, DockerError, local_daemon


@pytest.fixture(scope="session")
def docker_daemon() -> Daemon:
    """The local daemon, or a skip explaining why there is none."""
    try:
        return local_daemon()
    except DockerError as exc:
        pytest.skip(f"Docker is not usable for sandboxes: {exc}")


@pytest.fixture
def no_leaked_containers(docker_daemon):
    """Fail the test if it leaves a sandbox container behind."""
    before = sandbox_containers()
    yield
    leaked = sandbox_containers() - before
    assert not leaked, f"sandbox containers left behind: {sorted(leaked)}"
