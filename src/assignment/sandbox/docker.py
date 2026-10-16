"""The Docker sandbox backend: a SWE-ReX server in a container on this machine.

SWE-ReX's own `DockerDeployment` is not used. Its control server takes port
8000, which the chess server needs; it publishes on every interface; it
installs SWE-ReX with pipx on every start; and it can hang forever when startup
times out. This deployment drives the Docker CLI directly and reuses the rest
of SWE-ReX unchanged: the `swerex-remote` server, `RemoteRuntime`, and the
startup wait.
"""

from __future__ import annotations

import asyncio
import io
import logging
import secrets
import tarfile
import time
from collections.abc import Sequence
from pathlib import Path

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import AbstractRuntime, IsAliveResponse
from swerex.runtime.remote import RemoteRuntime
from swerex.utils.log import get_logger
from swerex.utils.wait import _wait_until_alive

from assignment.sandbox.config import DockerSandboxConfig
from assignment.sandbox.dockercli import DockerError, docker, local_daemon
from assignment.sandbox.images import LABEL, SERVER_ROOT, prepare_images
from assignment.utils.image import ImageSpec

# SWE-ReX's port on Modal too. Port 8000 stays free for the chess server.
SWEREX_PORT = 8880
# The first container from a large image just pulled waits for the image to be
# unpacked, which takes minutes for a SWE-bench image.
CONTAINER_TIMEOUT = 900
# Flags every sandbox container gets. Model-written commands run as root in
# here, so take away what they never need: privilege escalation through setuid
# binaries, raw sockets on the bridge network, device nodes, and fork bombs.
HARDENING = (
    "--security-opt", "no-new-privileges",
    "--cap-drop", "NET_RAW",
    "--cap-drop", "MKNOD",
    "--cap-drop", "AUDIT_WRITE",
    "--pids-limit", "4096",
)  # fmt: skip


def _files_tar(files: Sequence[tuple[Path, str]]) -> bytes:
    """A tar archive placing each local file at its absolute sandbox path."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for source, target in files:
            data = Path(source).read_bytes()
            info = tarfile.TarInfo(target.lstrip("/"))
            info.size = len(data)
            info.mode = 0o644
            info.mtime = int(time.time())
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def as_image_spec(image: object) -> ImageSpec:
    """Read an `Environment` image argument as a recipe the Docker backend can build."""
    if isinstance(image, ImageSpec):
        return image
    if isinstance(image, (str, Path)) and Path(image).is_file():
        # A Dockerfile, built from its own directory, as SWE-ReX does on Modal.
        dockerfile = Path(image).resolve()
        return ImageSpec(dockerfile=dockerfile, context=dockerfile.parent)
    if isinstance(image, str):
        return ImageSpec(base=image)
    raise TypeError(
        f"The Docker backend cannot run a {type(image).__name__} image. Pass an ImageSpec "
        "(see assignment.utils.image.build_testbed_image), an image name, or a Dockerfile "
        "path; a modal.Image only works with SANDBOX=modal."
    )


class AssignmentDockerDeployment(AbstractDeployment):
    """A SWE-ReX deployment in a container on the local Docker daemon."""

    def __init__(
        self,
        *,
        image: ImageSpec,
        exposed_ports: Sequence[int] = (),
        startup_timeout: float = 600,
        runtime_timeout: float = 600,
        deployment_timeout: float = 600,
        run_args: Sequence[str] = (),
        builder: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or get_logger("rex-deploy")
        self._image = image
        self._exposed_ports = tuple(exposed_ports)
        self._startup_timeout = startup_timeout
        self._runtime_timeout = runtime_timeout
        self._deployment_timeout = deployment_timeout
        self._run_args = tuple(run_args)
        self._builder = builder
        self._hooks = CombinedDeploymentHook()
        self._runtime: RemoteRuntime | None = None
        # Set from `docker create` until the container is removed.
        self.container: str | None = None

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @property
    def runtime(self) -> AbstractRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def start(self) -> None:
        """Create and start the container, then wait until its server answers.

        On any failure the container is removed before the error propagates.
        """
        if self.container is not None:
            self.logger.warning(
                "Deployment is already started; ignoring duplicate start()."
            )
            return

        self._hooks.on_custom_step("Preparing Docker images")
        images = prepare_images(
            self._image, daemon=local_daemon(), builder=self._builder
        )

        token = secrets.token_hex(32)
        name = f"assignment-sandbox-{secrets.token_hex(6)}"
        publish = [
            arg
            for port in (SWEREX_PORT, *self._exposed_ports)
            for arg in ("-p", f"127.0.0.1::{port}")
        ]
        self._hooks.on_custom_step("Starting Docker container")
        started = time.monotonic()
        # Claimed before `docker create`: a create that fails or times out may
        # still have made the container, and the cleanup below removes it.
        self.container = name
        try:
            docker(
                "create", "--name", name, "--label", f"{LABEL}=1",
                # --rm with a killing `timeout` as the main process is the
                # lifetime limit: if this process dies without stop(), the
                # container still goes away once deployment_timeout elapses.
                "--rm", "--init", "--entrypoint", "",
                *HARDENING,
                # The server comes from its own image, read-only and off PATH.
                "--mount",
                f"type=image,source={images.server},target={SERVER_ROOT},"
                f"image-subpath={SERVER_ROOT.lstrip('/')},readonly=true",
                *self._run_args,
                *publish,
                images.testbed,
                "timeout", "-s", "KILL", str(max(1, int(self._deployment_timeout))),
                f"{SERVER_ROOT}/venv/bin/swerex-remote",
                "--host", "0.0.0.0", "--port", str(SWEREX_PORT), "--auth-token", token,
                timeout=CONTAINER_TIMEOUT,
            )  # fmt: skip
            if self._image.files:
                files = _files_tar(self._image.files)
                docker("cp", "-", f"{name}:/", input=files, timeout=CONTAINER_TIMEOUT)
            docker("start", name, timeout=CONTAINER_TIMEOUT)
            self.logger.info("Started container %s from %s", name, images.testbed)

            url = self.port_url(SWEREX_PORT)
            host, port = url.rsplit(":", 1)
            self._runtime = RemoteRuntime(
                host=host,
                port=int(port),
                timeout=self._runtime_timeout,
                auth_token=token,
                logger=self.logger,
            )
            self._hooks.on_custom_step("Starting runtime")

            async def probe(timeout: float | None) -> IsAliveResponse:
                # Fail at once on a dead container rather than at the timeout.
                if not self.container_running():
                    raise RuntimeError(
                        f"the container exited while its SWE-ReX server was starting. The "
                        f"image {images.testbed} needs glibc and `timeout` (coreutils) for the "
                        "server to run"
                    )
                assert self._runtime is not None
                return await self._runtime.is_alive(timeout=timeout)

            await _wait_until_alive(
                probe, timeout=self._startup_timeout, function_timeout=5
            )
            self.logger.info(
                "Sandbox %s ready in %.1fs", name, time.monotonic() - started
            )
        except DockerError:
            await self.stop()
            raise
        except Exception as exc:
            logs = self._log_tail()
            await self.stop()
            raise RuntimeError(
                f"Docker sandbox {name} did not become reachable: "
                f"{type(exc).__name__}: {exc}{logs}"
            ) from exc
        except BaseException:
            # Interrupted (Ctrl-C): still never leave the container behind.
            await self.stop()
            raise

    def _log_tail(self) -> str:
        if self.container is None:
            return ""
        try:
            logs = docker("logs", "--tail", "40", self.container, merge_stderr=True)
        except DockerError:
            return ""
        return f"\nContainer logs:\n{logs}" if logs else ""

    def container_running(self) -> bool:
        """Ask Docker whether the container still runs."""
        if self.container is None:
            return False
        try:
            return (
                docker("inspect", "--format", "{{.State.Running}}", self.container)
                == "true"
            )
        except DockerError:
            # Removed: by stop(), or by --rm once its lifetime ran out.
            return False

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if not self.container_running():
            return IsAliveResponse(
                is_alive=False, message=f"Container {self.container} is not running"
            )
        return await self.runtime.is_alive(timeout=timeout)

    def port_url(self, port: int) -> str:
        """The http://127.0.0.1 URL a port published at creation is reachable at."""
        if self.container is None:
            raise DeploymentNotStartedError()
        published = docker("port", self.container, check=False)
        for line in published.splitlines():
            # "8000/tcp -> 127.0.0.1:49153"
            container_port, _, address = line.partition(" -> ")
            if container_port == f"{port}/tcp" and address.startswith("127.0.0.1:"):
                return f"http://{address}"
        forwarded = sorted(
            {
                int(line.split("/", 1)[0])
                for line in published.splitlines()
                if "/" in line
            }
            - {SWEREX_PORT}
        )
        raise ValueError(
            f"Port {port} was not forwarded. Available forwarded ports: "
            f"{', '.join(map(str, forwarded)) or 'none'}."
        )

    async def stop(self, timeout: float = 30) -> None:
        """Remove the container. Safe to call more than once."""
        self.remove(timeout)

    def remove(self, timeout: float = 30) -> None:
        """`stop()` without an event loop."""
        # The server needs no shutdown request: the container goes with it, and
        # RemoteRuntime keeps no connection open between requests.
        self._runtime = None
        if self.container is None:
            return
        name, self.container = self.container, None
        deadline = time.monotonic() + timeout
        try:
            # Detached: a second Ctrl-C while stopping must not leave it behind.
            docker("rm", "--force", name, timeout=timeout, check=False, detached=True)
            # A container that just exited may still be going through --rm's own
            # removal, which `rm` does not wait for. Stopped should mean gone.
            while time.monotonic() < deadline:
                docker("container", "inspect", "--format", "{{.Id}}", name)
                time.sleep(0.1)
        except DockerError:
            return
        self.logger.warning(
            "Container %s still exists %ss after removing it", name, timeout
        )

    def __del__(self):
        # AbstractDeployment.__del__ needs an event loop, which is usually gone
        # by then; removing the container needs none.
        if getattr(self, "container", None) is not None:
            self.remove(timeout=10)


class DockerBackend:
    """A sandbox in a container on this machine's Docker daemon."""

    name = "docker"

    def __init__(
        self,
        config: DockerSandboxConfig,
        *,
        image: object,
        exposed_ports: Sequence[int],
        startup_timeout: float,
        runtime_timeout: float,
        deployment_timeout: float,
        install_pipx: bool,
        modal_sandbox_kwargs: dict | None,
    ):
        if modal_sandbox_kwargs:
            raise ValueError(
                "modal_sandbox_kwargs only applies to SANDBOX=modal. Pass exposed_ports= for "
                "ports, and set SANDBOX_DOCKER_RUN_ARGS for container resources."
            )
        # install_pipx is unused: the server is mounted, never installed.
        del install_pipx
        self.deployment = AssignmentDockerDeployment(
            image=as_image_spec(image),
            exposed_ports=exposed_ports,
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
            deployment_timeout=deployment_timeout,
            run_args=config.run_args,
            builder=config.build_context,
        )

    def start(self) -> None:
        asyncio.run(self.deployment.start())

    @property
    def runtime(self) -> AbstractRuntime:
        return self.deployment.runtime

    def is_alive(self) -> bool:
        return self.deployment.container_running()

    def stop(self, timeout: float = 10) -> None:
        self.deployment.remove(timeout)

    def port_url(self, port: int) -> str:
        return self.deployment.port_url(port)
