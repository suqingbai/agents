"""The `Environment` contract, run against each sandbox backend.

The Docker parameter runs under `make test-docker`. The Modal parameter is
billable and runs only when asked for: `uv run pytest -m modal tests_sandbox`.
"""

from __future__ import annotations

import base64
import time
from urllib.request import urlopen

import pytest
from support import IMAGE

from assignment.env import Environment
from assignment.eval.harness import _write_remote_file

PORT = 9000


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("docker", marks=pytest.mark.docker),
        pytest.param("modal", marks=pytest.mark.modal),
    ],
)
def env(request):
    """One sandbox per backend, shared by every test in this module."""
    if request.param == "docker":
        request.getfixturevalue("docker_daemon")
    sandbox = Environment(image=IMAGE, backend=request.param, exposed_ports=[PORT])
    yield sandbox
    sandbox.stop()


def test_platform_is_read_from_inside_the_sandbox(env):
    assert env.system == "Linux"
    assert env.release and env.version and env.machine


def test_shell_command(env):
    output = env.execute("echo 'hello, world'")
    assert output["returncode"] == 0
    assert output["output"] == "hello, world\n"
    assert output["exception_info"] == ""


def test_argv_command(env):
    output = env.execute(["echo", "hello, world"], shell=False)
    assert output["returncode"] == 0
    assert output["output"] == "hello, world\n"


def test_failing_command(env):
    output = env.execute(
        'python -c \'print("hello, world"); raise Exception("test exception")\''
    )
    assert output["returncode"] != 0
    assert "hello, world\n" in output["output"]
    assert "test exception" in output["output"]
    assert "test exception" in output["stderr"] and output["stdout"] == "hello, world\n"


def test_a_timed_out_command_is_reported_and_the_sandbox_lives_on(env):
    started = time.monotonic()
    output = env.execute("sleep 30", timeout=1)
    assert time.monotonic() - started < 15
    assert output["returncode"] == -1
    assert "Timeout" in output["output"]
    assert output["extra"]["exception_type"] == "CommandTimeoutError"
    assert env.is_alive()
    assert env.execute("echo still here")["output"] == "still here\n"


def test_cwd_and_env_overrides(env):
    assert env.execute("pwd")["output"] == "/\n"
    assert env.execute("pwd", cwd="/tmp")["output"] == "/tmp\n"
    env.env_defaults["GREETING"] = "default"
    try:
        assert env.execute("echo $GREETING")["output"] == "default\n"
        assert (
            env.execute("echo $GREETING", env={"GREETING": "override"})["output"]
            == "override\n"
        )
    finally:
        env.env_defaults.pop("GREETING")


def test_file_round_trip(env):
    content = "line one\n  indented 'quoted' \"double\" $HOME `tick`\nunicode: é✓\n"
    _write_remote_file(env, content, "/tmp/round-trip.txt")
    assert env.execute("cat /tmp/round-trip.txt")["output"] == content

    blob = bytes(range(256)) * 64
    encoded = base64.b64encode(blob).decode()
    assert env.execute(f"echo {encoded} | base64 -d > /tmp/blob.bin")["returncode"] == 0
    read = env.execute("base64 -w0 /tmp/blob.bin")["output"]
    assert base64.b64decode(read) == blob


def test_activate_conda_env_puts_its_bin_first(env):
    root = "/tmp/fake-conda"
    env.execute(
        f"mkdir -p {root}/envs/demo/bin && printf '#!/bin/sh\\necho from-demo\\n' "
        f"> {root}/envs/demo/bin/demo-tool && chmod +x {root}/envs/demo/bin/demo-tool"
    )
    original = env.env_defaults.copy()
    try:
        path = env.activate_conda_env("demo", root=root)
        assert path.startswith(f"{root}/envs/demo/bin:")
        assert env.execute("demo-tool")["output"] == "from-demo\n"
        with pytest.raises(FileNotFoundError):
            env.activate_conda_env("missing", root=root)
    finally:
        env.env_defaults = original


def test_exposed_port_is_reachable_from_this_machine(env):
    env.execute(
        f"mkdir -p /tmp/www && echo served > /tmp/www/index.txt && cd /tmp/www && "
        f"nohup python -m http.server {PORT} > /tmp/http.log 2>&1 < /dev/null &"
    )
    url = env.port_url(PORT)
    assert url == env.tunnel_url(PORT)
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(f"{url.rstrip('/')}/index.txt", timeout=5) as response:
                assert response.read() == b"served\n"
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.25)


def test_an_unexposed_port_is_an_error(env):
    with pytest.raises(ValueError, match="Port 9999 was not forwarded"):
        env.port_url(9999)
