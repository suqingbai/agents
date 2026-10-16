# 11-768 Assignment 1: Build an Agent Harness

This repository contains the starter code and instructions for Assignment 1 for 11-768, the Carnegie Mellon University course on AI Agents. You can find the problem statement, along with instructions for setup and notes on grading in [ASSIGNMENT.md](ASSIGNMENT.md)

Authors: Weiwei Sun and Saujas Vaduguru, with feedback from 11-768 course staff (instructors Daniel Fried and Graham Neubig, and TAs Aditya Soni, Andy Liu, Apurva Gandhi, Demi Wang, Jiarui Liu, and Yueqi Song).

## Running sandboxes on local Docker instead of Modal

Modal is the default. To run every sandbox (the agents, the chess server, and the graders) as a container on this machine's Docker instead, set `SANDBOX=docker`, in `.env` or per command:

```bash
make doctor SANDBOX=docker       # checks Docker: a local daemon, Engine 28 or newer
make run-code-agent SANDBOX=docker
make check-part1 SANDBOX=docker
make test-docker                 # the sandbox contract, chess, and grading on Docker
```

Code can choose per call with `backend="docker"` on `Environment`, `ChessSandbox`, and `evaluate`.

- **Requirements.** Docker Engine 28 or newer on this machine. Docker is refused when `DOCKER_HOST`, `DOCKER_CONTEXT`, or `docker context use` point at a daemon on another machine.
- **Images.** They are built here and reused while their inputs are unchanged: `assignment-testbed:<hash>` (the task Dockerfile plus pins) and `assignment-swerex:<hash>` (the SWE-ReX server, mounted read-only at `/opt/swerex` in every container). SWE-bench images are pulled as published. The first build takes a few minutes. Remove old ones with `docker image prune --all --filter label=assignment.sandbox`.
- **Containers.** Named `assignment-sandbox-*` and labelled `assignment.sandbox=1`, with ports published on `127.0.0.1` only. Each one removes itself at `deployment_timeout`, even when its caller crashed. List any with `docker ps --filter label=assignment.sandbox`.
- **Isolation.** Containers use runc with `no-new-privileges`, a few dropped capabilities, and a process limit. They are weaker than Modal's gVisor sandboxes. For gVisor where it is installed, or for resource limits, add `docker create` flags: `SANDBOX_DOCKER_RUN_ARGS="--runtime=runsc --cpus=2 --memory=4g"`.
- **Builds elsewhere.** If this machine cannot build the images, set `SANDBOX_DOCKER_BUILD_CONTEXT=<buildx builder>` to use another BuildKit builder whose output loads into the local daemon (see `docker buildx ls`). Nothing falls back on its own.
- **Artifacts differ slightly.** Trajectories from Docker show this machine's kernel in `<system_information>`, and the chess server at `http://127.0.0.1:<port>` instead of a Modal tunnel. Patches are unaffected, because `/testbed` is built from the same sources.
