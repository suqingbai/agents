"""The Docker CLI, as the Docker backend uses it, and checks on its daemon.

Everything goes through the `docker` command rather than an SDK, so Docker's own
settings (`DOCKER_HOST`, `DOCKER_CONTEXT`, `docker context use`, credential
helpers) apply exactly as they do in a terminal.
"""

from __future__ import annotations

import ipaddress
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

# Image mounts (`docker create --mount type=image`), which put the SWE-ReX
# server into every container read-only, arrived in Engine 28.
MIN_ENGINE_MAJOR = 28
# Queries that should answer instantly. A daemon that does not answer in this
# time is as good as absent, and Docker would otherwise wait on it for minutes.
QUERY_TIMEOUT = 20


class DockerError(RuntimeError):
    """A docker command failed, or Docker is not usable for sandboxes."""


def docker(
    *args: str,
    input: bytes | None = None,
    timeout: float | None = QUERY_TIMEOUT,
    check: bool = True,
    merge_stderr: bool = False,
    detached: bool = False,
) -> str:
    """Run `docker <args>` and return its stripped stdout (and stderr, if merged).

    Args:
        detached: For cleanup that must finish. The command runs in its own
            session, out of reach of a Ctrl-C at the terminal, and is left to
            complete when this process is interrupted or stops waiting for it.

    Raises:
        DockerError: The CLI is missing, timed out, or (with `check`) exited
            non-zero. The message carries docker's own stderr.
    """
    try:
        process = subprocess.Popen(
            ["docker", *args],
            stdin=subprocess.DEVNULL if input is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            start_new_session=detached,
        )
    except FileNotFoundError as exc:
        raise DockerError(
            "The docker CLI is not installed or not on PATH; SANDBOX=docker needs Docker "
            f"Engine {MIN_ENGINE_MAJOR} or newer."
        ) from exc
    try:
        stdout, stderr = process.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if not detached:
            process.kill()
            process.wait()
        raise DockerError(
            f"`docker {' '.join(args[:2])}` did not finish within {timeout:g}s."
        ) from exc
    except BaseException:
        if not detached:
            process.kill()
            process.wait()
        raise
    if check and process.returncode != 0:
        message = (stderr or b"").decode(errors="replace").strip()
        raise DockerError(
            f"`docker {' '.join(args[:2])}` failed (exit {process.returncode}): {message}"
        )
    return stdout.decode(errors="replace").strip()


@dataclass(frozen=True)
class Daemon:
    """The Docker daemon sandboxes run on."""

    endpoint: str
    version: str
    os: str
    arch: str

    @property
    def platform(self) -> str:
        return f"{self.os}/{self.arch}"


def is_local_endpoint(endpoint: str) -> bool:
    """Whether a Docker endpoint is a daemon on this machine.

    Only then is a port the daemon publishes on 127.0.0.1 reachable from here.
    """
    parsed = urlparse(endpoint)
    if parsed.scheme in ("unix", "npipe"):
        return True
    if parsed.scheme == "tcp" and parsed.hostname:
        if parsed.hostname == "localhost":
            return True
        try:
            return ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            return False
    return False


def engine_major(version: str) -> int | None:
    """The major number of an Engine version like `29.8.1`, or None if unreadable."""
    head = version.strip().lstrip("v").split(".", 1)[0]
    return int(head) if head.isdigit() else None


def local_daemon() -> Daemon:
    """Find the daemon Docker is pointed at, and check that sandboxes can run on it.

    Raises:
        DockerError: Docker is missing or unreachable, points at a daemon on
            another machine, or the Engine is too old or not a Linux engine.
    """
    try:
        endpoint = docker(
            "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"
        )
    except DockerError as exc:
        raise DockerError(
            f"Could not resolve which Docker daemon to use: {exc}"
        ) from exc
    if not is_local_endpoint(endpoint):
        raise DockerError(
            f"SANDBOX=docker runs containers on this machine's Docker daemon only, but Docker "
            f"points at {endpoint} (set by DOCKER_HOST, DOCKER_CONTEXT, or `docker context use`). "
            "Point Docker at the local daemon, for example with `unset DOCKER_HOST DOCKER_CONTEXT` "
            "and `docker context use default`, or use SANDBOX=modal."
        )

    try:
        version, os_name, arch = docker(
            "version", "--format", "{{.Server.Version}}|{{.Server.Os}}|{{.Server.Arch}}"
        ).split("|")
    except (DockerError, ValueError) as exc:
        raise DockerError(
            f"The Docker daemon at {endpoint} is not reachable ({exc}). Start Docker "
            f"(Engine {MIN_ENGINE_MAJOR} or newer), or use SANDBOX=modal."
        ) from exc

    major = engine_major(version)
    if major is None or major < MIN_ENGINE_MAJOR:
        raise DockerError(
            f"Docker Engine {version} at {endpoint} is too old: SANDBOX=docker needs Engine "
            f"{MIN_ENGINE_MAJOR} or newer, for the image mounts that put the SWE-ReX server into "
            "each container read-only. Upgrade Docker, or use SANDBOX=modal."
        )
    if os_name != "linux":
        raise DockerError(
            f"The Docker daemon at {endpoint} runs {os_name} containers; SANDBOX=docker needs "
            "Linux containers."
        )
    return Daemon(endpoint=endpoint, version=version, os=os_name, arch=arch)
