import asyncio
import posixpath
from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

import modal
from swerex.runtime.abstract import Command

from assignment.sandbox import (
    DockerSandboxConfig,
    ModalSandboxConfig,
    SandboxBackend,
    create_backend,
    resolve_sandbox_config,
)

# Re-exported: these lived here before the backends moved to assignment.sandbox.
from assignment.sandbox.modal import AssignmentModalDeployment, _tls_port_configuration
from assignment.utils.image import ImageSpec

__all__ = ["AssignmentModalDeployment", "Environment", "_tls_port_configuration"]


class Environment:
    """
    Executes bash commands in a sandbox (Modal or Docker) via SWE-ReX.
    """

    # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_modal.py

    def __init__(
        self,
        image: "ImageSpec | str | PurePath | modal.Image" = "python:3.12",
        cwd: str = "/",
        startup_timeout: float = 600,
        runtime_timeout: float = 600,
        deployment_timeout: float = 600,
        install_pipx: bool = True,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        conda_env: str | None = None,
        *,
        backend: "str | ModalSandboxConfig | DockerSandboxConfig | None" = None,
        exposed_ports: Sequence[int] = (),
    ):
        """Launch a sandbox and block until its runtime answers.

        Args:
            image: An `ImageSpec`, a Dockerhub or ECR image name, a path to a
                Dockerfile, or (Modal only) a prebuilt `modal.Image`. Describe a
                task testbed with `assignment.utils.image.build_testbed_image`,
                which is the only way to pass a credential to a private clone.
            cwd: Working directory for commands that do not specify one.
            startup_timeout: Seconds to wait for the SWE-ReX runtime to come up.
            runtime_timeout: Seconds a single command may run before timing out.
            deployment_timeout: Seconds the sandbox may stay alive before it is
                reclaimed.
            install_pipx: Install pipx in the image, needed to bootstrap the
                SWE-ReX server on Modal when it is not already present. Docker
                mounts its own server and ignores this.
            modal_sandbox_kwargs: Additional keyword arguments forwarded to
                ``modal.Sandbox.create``. Modal only.
            conda_env: Name of a conda environment to put on PATH for every
                command. SWE-bench images install the repository under test into
                an environment named ``testbed`` but never activate it, so
                without this ``python`` is conda's base environment, where the
                repository and its dependencies are not installed.
            backend: `"modal"`, `"docker"`, or a sandbox config. None reads the
                `SANDBOX` variable (environment or `.env`), defaulting to Modal.
            exposed_ports: Ports inside the sandbox to make reachable from this
                machine through `port_url`.
        """
        self.cwd = cwd
        # Merged into every command's environment; a per-call `env` wins.
        self.env_defaults: dict[str, str] = {}
        self.backend: SandboxBackend = create_backend(
            resolve_sandbox_config(backend),
            image=image,
            exposed_ports=exposed_ports,
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
            deployment_timeout=deployment_timeout,
            install_pipx=install_pipx,
            modal_sandbox_kwargs=modal_sandbox_kwargs,
        )
        self.backend.start()

        if conda_env:
            self.activate_conda_env(conda_env)

        # Read the platform from inside the container, not from platform.uname(),
        # which would describe the machine running this code instead.
        self.system, self.release, self.version, self.machine = self.execute(
            "uname -s; uname -r; uname -v; uname -m"
        )["output"].splitlines()

    @property
    def deployment(self):
        """The SWE-ReX deployment behind the backend."""
        return self.backend.deployment

    def is_alive(self) -> bool:
        """Whether the sandbox is still running.

        A sandbox is reclaimed once `deployment_timeout` elapses, and the
        deployment keeps its handle afterwards, so this asks the sandbox itself
        rather than trusting the handle's existence.
        """
        return self.backend.is_alive()

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
            result = asyncio.run(self.backend.runtime.execute(Command(**arguments)))
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
        self.backend.stop(timeout)

    def port_url(self, port: int) -> str:
        """Return the URL a port listed in `exposed_ports` is reachable at.

        On Modal this is a public HTTPS tunnel; on Docker it is
        http://127.0.0.1 on this machine.
        """
        return self.backend.port_url(port)

    # The historical name, from when every sandbox was on Modal.
    tunnel_url = port_url

    def __enter__(self):
        """Enter a `with` block. The sandbox is already running by this point."""
        return self

    def __exit__(self, *exc):
        """Terminate the sandbox on leaving a `with` block, including on error."""
        self.stop()
