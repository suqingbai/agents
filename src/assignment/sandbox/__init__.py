"""Sandbox backends: where `Environment` runs commands.

Every backend runs the same SWE-ReX server next to the code under test, so
command output, exit codes, and timeouts mean the same thing everywhere. They
differ only in how the image gets built, how the sandbox starts and stops, and
how this process reaches ports inside it.

Pick one with `SANDBOX=modal|docker` (environment or `.env`; default `modal`) or
`backend=` on `Environment`, `ChessSandbox`, and `evaluate`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from swerex.deployment.abstract import AbstractDeployment
from swerex.runtime.abstract import AbstractRuntime

from assignment.sandbox.config import (
    DockerSandboxConfig,
    ModalSandboxConfig,
    SandboxConfig,
    resolve_sandbox_config,
)
from assignment.sandbox.docker import DockerBackend
from assignment.sandbox.modal import ModalBackend

__all__ = [
    "DockerSandboxConfig",
    "ModalSandboxConfig",
    "SandboxBackend",
    "SandboxConfig",
    "create_backend",
    "resolve_sandbox_config",
]


class SandboxBackend(Protocol):
    """One sandbox: somewhere a SWE-ReX server runs next to the code under test."""

    name: str
    # The SWE-ReX deployment underneath, for callers that need its details.
    deployment: AbstractDeployment

    def start(self) -> None:
        """Block until the SWE-ReX runtime answers. On failure, clean up, then raise."""

    @property
    def runtime(self) -> AbstractRuntime:
        """The SWE-ReX runtime that `Environment.execute` sends commands to."""

    def is_alive(self) -> bool:
        """Ask the sandbox itself, not a cached handle, whether it still runs."""

    def port_url(self, port: int) -> str:
        """A URL this machine reaches `port` inside the sandbox at."""

    def stop(self, timeout: float = 10) -> None:
        """Terminate the sandbox."""


def create_backend(
    config: ModalSandboxConfig | DockerSandboxConfig,
    *,
    image: Any,
    exposed_ports: Sequence[int],
    startup_timeout: float,
    runtime_timeout: float,
    deployment_timeout: float,
    install_pipx: bool,
    modal_sandbox_kwargs: dict[str, Any] | None,
) -> SandboxBackend:
    """Make an unstarted backend of the configured type."""
    options = {
        "image": image,
        "exposed_ports": exposed_ports,
        "startup_timeout": startup_timeout,
        "runtime_timeout": runtime_timeout,
        "deployment_timeout": deployment_timeout,
        "install_pipx": install_pipx,
        "modal_sandbox_kwargs": modal_sandbox_kwargs,
    }
    if isinstance(config, DockerSandboxConfig):
        return DockerBackend(config, **options)
    return ModalBackend(**options)
