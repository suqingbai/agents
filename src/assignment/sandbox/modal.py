"""The Modal sandbox backend: SWE-ReX's Modal deployment, with fixes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

import modal
from swerex.deployment.modal import ModalDeployment, _ImageBuilder
from swerex.runtime.abstract import AbstractRuntime
from swerex.runtime.remote import RemoteRuntime

from assignment.utils.image import ImageSpec, is_ignored

logger = logging.getLogger(__name__)


def _tls_port_configuration(
    modal_sandbox_kwargs: dict[str, Any], runtime_port: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Put the SWE-ReX control port on the TLS tunnel exactly once."""

    kwargs = dict(modal_sandbox_kwargs)
    encrypted = list(kwargs.pop("encrypted_ports", []))
    unencrypted = [
        port for port in kwargs.pop("unencrypted_ports", []) if port != runtime_port
    ]
    encrypted = list(dict.fromkeys([runtime_port, *encrypted]))
    return encrypted, unencrypted, kwargs


class AssignmentModalDeployment(ModalDeployment):
    """Modal deployment with a correctly matched TLS tunnel and URL.

    SWE-ReX 1.4.0 opens its control port as unencrypted TCP but initializes the
    HTTP client with ``Tunnel.url``, which is Modal's HTTPS endpoint. Keep the
    upstream runtime and authentication protocol, but expose that control port
    through ``encrypted_ports`` so its URL and transport agree.
    """

    async def start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning(
                "Deployment is already started; ignoring duplicate start()."
            )
            return

        self.logger.info("Starting Modal sandbox")
        self._hooks.on_custom_step("Starting Modal sandbox")
        started = time.monotonic()
        token = self._get_token()
        encrypted, unencrypted, extra_kwargs = _tls_port_configuration(
            self._modal_kwargs, self._port
        )
        try:
            self._sandbox = await modal.Sandbox.create.aio(
                "/usr/bin/env",
                "bash",
                "-c",
                self._start_swerex_cmd(token),
                image=self._image,
                timeout=int(self._deployment_timeout),
                encrypted_ports=encrypted,
                unencrypted_ports=unencrypted,
                app=self._app,
                **extra_kwargs,
            )
            tunnels = await self._sandbox.tunnels.aio()
            tunnel = tunnels[self._port]
            elapsed = time.monotonic() - started
            log_url = await self.get_modal_log_url()
            self.logger.info(
                "Sandbox %s created in %.2fs", self._sandbox.object_id, elapsed
            )
            self.logger.info("Modal logs: %s", log_url)
            await asyncio.sleep(1)
            self.logger.info("Starting SWE-ReX runtime at %s", tunnel.url)
            self._hooks.on_custom_step("Starting runtime")
            self._runtime = RemoteRuntime(
                host=tunnel.url,
                timeout=self._runtime_timeout,
                auth_token=token,
                logger=self.logger,
            )
            remaining = max(0, self._startup_timeout - elapsed)
            await self._wait_until_alive(timeout=remaining)
        except Exception as exc:
            details = [f"{type(exc).__name__}: {exc}"]
            if self._sandbox is not None:
                details.append(f"sandbox_id={self._sandbox.object_id}")
                with contextlib.suppress(Exception):
                    details.append(f"logs={await self.get_modal_log_url()}")
                with contextlib.suppress(Exception):
                    await self._sandbox.terminate.aio()
            self._runtime = None
            self._sandbox = None
            raise RuntimeError(
                "Modal sandbox started, but the SWE-ReX control runtime did not "
                "become reachable. " + " | ".join(details)
            ) from exc


def to_modal_image(spec: ImageSpec) -> modal.Image:
    """Build an `ImageSpec` as a Modal image."""
    if spec.dockerfile is not None:
        logger.info("Building %s on Modal from %s", spec.dockerfile, spec.context)
        image = modal.Image.from_dockerfile(
            str(spec.dockerfile),
            context_dir=str(spec.context),
            force_build=spec.force_build,
            ignore=is_ignored,
        )
    else:
        # SWE-ReX's builder handles registry credentials and ECR for a name.
        image = _ImageBuilder(install_pipx=False).auto(spec.base)

    # Pin last, deliberately. Modal appends its own dependency install after the
    # Dockerfile's commands on image builder versions <= 2024.10, and the 2023.12
    # requirements file pins fastapi==0.88.0, which pulls starlette down to 0.22
    # and breaks every test using TestClient. Chaining here puts this layer on
    # top of that one, so the task's versions are the ones that survive.
    if spec.pins:
        logger.info("Pinning %d packages on top of the built image", len(spec.pins))
        image = image.pip_install(*spec.pins)

    for source, target in spec.files:
        # copy=True: SWE-ReX adds its runtime build layer afterwards.
        image = image.add_local_file(str(source), target, copy=True)
    return image


class ModalBackend:
    """A sandbox on Modal, reached through Modal's encrypted tunnels."""

    name = "modal"

    def __init__(
        self,
        *,
        image: ImageSpec | str | PurePath | modal.Image,
        exposed_ports: Sequence[int],
        startup_timeout: float,
        runtime_timeout: float,
        deployment_timeout: float,
        install_pipx: bool,
        modal_sandbox_kwargs: dict[str, Any] | None,
    ):
        kwargs = dict(modal_sandbox_kwargs or {})
        if exposed_ports:
            kwargs["encrypted_ports"] = [
                *kwargs.get("encrypted_ports", []),
                *exposed_ports,
            ]
        self.deployment = AssignmentModalDeployment(
            # Anything but a recipe goes to SWE-ReX as is, as it always has.
            image=to_modal_image(image) if isinstance(image, ImageSpec) else image,
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
            deployment_timeout=deployment_timeout,
            install_pipx=install_pipx,
            modal_sandbox_kwargs=kwargs,
        )

    def start(self) -> None:
        async def _start():
            await self.deployment.start()
            await self.deployment.is_alive()

        asyncio.run(_start())

    @property
    def runtime(self) -> AbstractRuntime:
        return self.deployment.runtime

    def is_alive(self) -> bool:
        # Modal reclaims a sandbox once `deployment_timeout` elapses, and the
        # deployment keeps its handle afterwards, so this asks the sandbox itself
        # rather than trusting the handle's existence.
        sandbox = self.deployment._sandbox
        return sandbox is not None and sandbox.poll() is None

    def stop(self, timeout: float = 10) -> None:
        async def _stop():
            # ModalDeployment.stop() has an inverted poll() check and only
            # terminates sandboxes that have *already* exited, so a live sandbox
            # leaks until deployment_timeout. Terminate it explicitly. Grab the
            # reference first: stop() clears _sandbox.
            sandbox = self.deployment._sandbox
            try:
                await asyncio.wait_for(self.deployment.stop(), timeout=timeout)
            finally:
                if sandbox is not None:
                    await asyncio.wait_for(sandbox.terminate.aio(), timeout=timeout)

        asyncio.run(_stop())

    def port_url(self, port: int) -> str:
        async def _tunnel_url() -> str:
            tunnels = await self.deployment.sandbox.tunnels.aio()
            try:
                return tunnels[port].url
            except KeyError as exc:
                forwarded = ", ".join(str(item) for item in sorted(tunnels)) or "none"
                raise ValueError(
                    f"Port {port} was not forwarded. Available forwarded ports: {forwarded}."
                ) from exc

        return asyncio.run(_tunnel_url())
