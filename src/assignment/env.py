import asyncio
import logging
import posixpath
import subprocess
import time
from pathlib import PurePath
from typing import Any, Literal

import modal
from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.deployment.docker import DockerDeployment
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command
from swerex.runtime.remote import RemoteRuntime


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
            self.logger.warning("Deployment is already started; ignoring duplicate start().")
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
            self.logger.info("Sandbox %s created in %.2fs", self._sandbox.object_id, elapsed)
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
                try:
                    details.append(f"logs={await self.get_modal_log_url()}")
                except Exception:
                    pass
                try:
                    await self._sandbox.terminate.aio()
                except Exception:
                    pass
            self._runtime = None
            self._sandbox = None
            raise RuntimeError(
                "Modal sandbox started, but the SWE-ReX control runtime did not "
                "become reachable. " + " | ".join(details)
            ) from exc


class AssignmentDockerDeployment(DockerDeployment):
    """Local Docker deployment that bootstraps and expires like the Modal one.

    Upstream DockerDeployment installs pipx with the image's ``python3 -m pip``,
    which adds packages to the testbed's own interpreter, and its containers
    live until ``stop()``. Instead, add the same apt pipx layers that
    ModalDeployment appends, start the server with Modal's command, and cap the
    container's lifetime at ``deployment_timeout`` as Modal does.
    """

    def __init__(
        self,
        *,
        image: str,
        install_pipx: bool = True,
        deployment_timeout: float = 3600.0,
        runtime_timeout: float | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            image: Tag of a local or registry Docker image.
            install_pipx: Build ``<image>-pipx`` with pipx installed through
                apt, needed to bootstrap the SWE-ReX server.
            deployment_timeout: Seconds the container may stay alive before it
                is killed and removed.
            runtime_timeout: Accepted for parity with AssignmentModalDeployment
                and ignored; DockerDeploymentConfig rejects unknown keys.
            **kwargs: Passed to DockerDeploymentConfig (e.g. startup_timeout).
        """
        self._deployment_timeout = deployment_timeout
        if install_pipx:
            image = self._with_pipx(image, kwargs.get("container_runtime", "docker"))
        super().__init__(image=image, logger=logger, **kwargs)

    @staticmethod
    def _with_pipx(image: str, container_runtime: str) -> str:
        # The same commands ModalDeployment's _ImageBuilder.ensure_pipx_installed adds.
        dockerfile = (
            f"FROM {image}\n"
            "RUN apt-get update\n"
            "RUN apt-get install -y pipx\n"
            "RUN pipx ensurepath\n"
        )
        tag = f"{image}-pipx"
        subprocess.run(
            [container_runtime, "build", "-q", "-t", tag, "-"],
            input=dockerfile,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return tag

    def _get_swerex_start_cmd(self, token: str) -> list[str]:
        rex_args = f"--auth-token {token}"
        server = f"{REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args}"
        # Make the container's main process expire on time; --rm then removes it.
        return [
            "timeout",
            "-s",
            "KILL",
            str(int(self._deployment_timeout)),
            "/bin/sh",
            "-c",
            server,
        ]

    async def stop(self) -> None:
        """Close the runtime, then always kill and reap the container.

        Upstream stop() awaits ``runtime.close()`` first, so when that raises
        (the container already died) or is cancelled, the kill, the reap and the
        state reset never run. Close it here, tolerating failure, and leave the
        rest to upstream with no runtime left to close.
        """
        runtime, self._runtime = self._runtime, None
        try:
            if runtime is not None:
                await runtime.close()
        except Exception as exc:
            self.logger.warning(
                "Closing the SWE-ReX runtime failed; killing the container anyway: %s", exc
            )
        finally:
            await super().stop()

    def is_running(self) -> bool:
        """Whether the container process is still running."""
        return self._container_process is not None and self._container_process.poll() is None


class Environment:
    """
    Executes bash commands in a Modal or local Docker sandbox via SWE-ReX.
    """

    # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_modal.py

    def __init__(
        self,
        image: "str | PurePath | modal.Image" = "python:3.12",
        cwd: str = "/",
        startup_timeout: float = 600,
        runtime_timeout: float = 600,
        deployment_timeout: float = 600,
        install_pipx: bool = True,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        conda_env: str | None = None,
        sandbox: Literal["modal", "docker"] = "docker",
    ):
        """Launch a sandbox and block until its runtime answers.

        Args:
            image: A prebuilt `modal.Image`, a Dockerhub or ECR image name, or a
                path to a Dockerfile. Build a task testbed with
                `assignment.utils.image.build_testbed_image`, which is the only
                way to pass a credential to a private clone.
            cwd: Working directory for commands that do not specify one.
            startup_timeout: Seconds to wait for the SWE-ReX runtime to come up.
            runtime_timeout: Seconds a single command may run before timing out.
            deployment_timeout: Seconds the sandbox may stay alive before it is
                reclaimed.
            install_pipx: Install pipx in the image, needed to bootstrap the
                SWE-ReX server when it is not already present.
            modal_sandbox_kwargs: Additional keyword arguments forwarded to
                ``modal.Sandbox.create``. This is used for capabilities such as
                encrypted port forwarding. Modal only.
            conda_env: Name of a conda environment to put on PATH for every
                command. SWE-bench images install the repository under test into
                an environment named ``testbed`` but never activate it, so
                without this ``python`` is conda's base environment, where the
                repository and its dependencies are not installed.
            sandbox: ``"modal"`` for a Modal sandbox, or ``"docker"`` for a
                container on the local Docker daemon, where `image` must be a
                Docker image tag.
        """
        self.cwd = cwd
        # Merged into every command's environment; a per-call `env` wins.
        self.env_defaults: dict[str, str] = {}
        self.sandbox = sandbox
        if sandbox == "modal":
            self.deployment = AssignmentModalDeployment(
                image=image,
                startup_timeout=startup_timeout,
                runtime_timeout=runtime_timeout,
                deployment_timeout=deployment_timeout,
                install_pipx=install_pipx,
                modal_sandbox_kwargs=modal_sandbox_kwargs,
            )
        elif sandbox == "docker":
            if modal_sandbox_kwargs:
                raise ValueError(
                    "modal_sandbox_kwargs only applies to sandbox='modal', "
                    "not sandbox='docker'."
                )
            self.deployment = AssignmentDockerDeployment(
                image=image,
                startup_timeout=startup_timeout,
                runtime_timeout=runtime_timeout,
                deployment_timeout=deployment_timeout,
                install_pipx=install_pipx,
            )
        else:
            raise ValueError(f"Unknown sandbox {sandbox!r}; expected 'modal' or 'docker'.")

        async def _start():
            await self.deployment.start()
            await self.deployment.is_alive()

        asyncio.run(_start())

        # The sandbox is running now, and a failed constructor never reaches
        # __exit__, so stop it here rather than leak it until deployment_timeout.
        try:
            if conda_env:
                self.activate_conda_env(conda_env)

            # Read the platform from inside the container, not from platform.uname(),
            # which would describe the machine running this code instead.
            self.system, self.release, self.version, self.machine = self.execute(
                "uname -s; uname -r; uname -v; uname -m"
            )["output"].splitlines()
        except BaseException:
            self.stop()
            raise

    def is_alive(self) -> bool:
        """Whether the sandbox is still running.

        Modal reclaims a sandbox once `deployment_timeout` elapses, and the
        deployment keeps its handle afterwards, so this asks the sandbox itself
        rather than trusting the handle's existence. A Docker container is
        killed at the same deadline, so this asks whether its process still runs.
        """
        if self.sandbox == "docker":
            return self.deployment.is_running()
        sandbox = self.deployment._sandbox
        return sandbox is not None and sandbox.poll() is None

    def activate_conda_env(self, name: str, root: str = "/opt/miniconda3") -> str:
        """Put a conda environment's bin directory first on PATH for all commands.

        Activating properly needs a login shell, which these commands do not
        get. Prepending the environment's bin directory has the same effect for
        ``python``, ``pip``, and anything else installed there.

        Args:
            name: The environment name, for example ``testbed``.
            root: Where conda is installed in the image.

        Returns:
            The PATH now used for every command.

        Raises:
            FileNotFoundError: If the environment is not in the image, rather
                than silently leaving the wrong interpreter on PATH.
        """
        binary_dir = posixpath.join(root, "envs", name, "bin")
        if self.execute(f"test -d {binary_dir}", cwd="/")["returncode"] != 0:
            raise FileNotFoundError(f"No conda environment at {binary_dir}")

        current = self.execute("printenv PATH", cwd="/")["output"].strip()
        self.env_defaults["PATH"] = f"{binary_dir}:{current}"
        return self.env_defaults["PATH"]

    def execute(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        # Command defaults to shell=False, which requires `command` to be an argv
        # list; default to shell=True so a plain command string works.
        shell: bool | None = True,
        # check=False so a failing command returns its exit code instead of
        # raising and discarding the output the agent needs to see.
        check: bool = False,
    ) -> dict:
        """Run a command in the sandbox and return its result.

        Args:
            command: A shell string, or an argv list if `shell` is False.
            timeout: Seconds before the command is killed. None means no timeout.
            cwd: Working directory to run the command in. Defaults to the
                environment's `cwd`.
            env: Environment variables to set for the command.
            shell: Run the command through a shell, as with `subprocess.run`.
                None is treated as True, since a model calling this as a tool may
                send null for an omitted argument.
            check: Raise on a non-zero exit code instead of reporting it.

        Returns:
            A dict with `output`, `returncode`, and `exception_info`. `output`
            holds stdout and stderr interleaved, as they would appear in a
            terminal. `exception_info` is empty when the command ran, whatever
            its exit code. If the call itself fails (the sandbox died, the
            command timed out), the exception is caught rather than raised:
            `returncode` is -1, `exception_info` describes it, and an `extra` key
            carries the exception type.
        """
        # Command.shell is a strict bool, so None would fail validation.
        shell = True if shell is None else shell

        arguments = {
            "command": command,
            "timeout": timeout,
            "env": {**self.env_defaults, **(env or {})} or None,
            "shell": shell,
            "check": check,
            "merge_output_streams": False,
        }
        if cwd is not None:
            arguments["cwd"] = cwd
        # A cwd from the caller wins; otherwise the environment's default applies.
        arguments.setdefault("cwd", self.cwd)

        try:
            result = asyncio.run(self.deployment.runtime.execute(Command(**arguments)))
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                # The coding agent consumes one observation string, while the
                # task harness needs the streams separately for diagnostics.
                "output": result.stdout,
                "returncode": result.exit_code,
                "exception_info": "",
            }
            if result.stderr:
                output["output"] += result.stderr
        except Exception as e:
            # A command can fail for reasons worth reporting to the model, but a
            # sandbox that is gone is terminal: every later command fails the
            # same way, so raise rather than let the caller keep going.
            if not self.is_alive():
                raise RuntimeError(
                    "The sandbox is no longer running, so no further commands can "
                    f"be executed. Last error: {e}"
                ) from e

            # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/extra/swerex_modal.py#L82-L87
            # Same keys as the success path, so callers never have to branch on
            # which one they got.
            output = {
                "stdout": "",
                "stderr": str(e) if str(e) else "",
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        return output

    def stop(self, timeout: float = 10):
        """Shut down the runtime and terminate the sandbox.

        Args:
            timeout: Seconds allowed for each of the shutdown and the
                termination steps.
        """

        if self.sandbox == "docker":
            # DockerDeployment.stop() kills the container, and --rm removes it.
            asyncio.run(asyncio.wait_for(self.deployment.stop(), timeout=timeout))
            return

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

    def tunnel_url(self, port: int) -> str:
        """Return the public URL for a port forwarded when the sandbox started.

        Raises:
            NotImplementedError: On a Docker sandbox, which has no Modal tunnels.
        """
        if self.sandbox == "docker":
            raise NotImplementedError(
                "tunnel_url is only available for sandbox='modal'; Docker sandboxes "
                "have no Modal tunnels."
            )

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

    def __enter__(self):
        """Enter a `with` block. The sandbox is already running by this point."""
        return self

    def __exit__(self, *exc):
        """Terminate the sandbox on leaving a `with` block, including on error."""
        self.stop()
