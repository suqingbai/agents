"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from jinja2 import Template

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    TOOL_USE_RULE,
    Agent,
    format_system_information,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

CODE_AGENT_SYSTEM_PROMPT_TEMPLATE = Template(
    """You are an autonomous software engineer. You resolve an issue in a code repository by running shell commands in a Linux sandbox through tool calls. No human will answer questions or approve steps during the run, so keep working on your own until the issue is resolved.

{{ system_information }}

# Environment
- The repository is at `{{ cwd }}`, the default working directory for every command.
- The repository is installed in editable mode, with its dependencies, in the Python environment of the first `python` on PATH. Run code and tests with that `python`; your source edits take effect without reinstalling. Do not install, upgrade, or remove packages unless the issue requires it.
- A command that never exits (a server, a watcher) stalls the run. Never start a server in the foreground, and pass `timeout` for anything that might hang.
- One tool call observation holds 10000 characters. Edit files with targeted replacements or short scripts instead of rewriting whole files.

# How to work
1. Investigate: read the issue, then find the relevant code with `grep -rn`, `find`, and narrow file reads.
2. Reproduce: write a small script under `/tmp` that shows the failure, and run it before you change anything.
3. Fix: make the smallest change to the project's source code that fixes the root cause, in the style of the surrounding code.
4. Verify: rerun the reproduction, then run the existing tests nearest the code you changed and confirm nothing else broke.
5. Clean up: remove debugging edits, and keep scratch files outside the repository.

# Rules
- Do not add, edit, or delete tests in the repository. Hidden tests are applied on top of your change.
- Leave your changes uncommitted in the working tree. Do not run `git commit`, `git stash`, `git reset`, or `git checkout`.
- {{ tool_use_rule }}
- You may make several independent tool calls in one response; they run in the order given.
- {{ step_budget_rule }} Prefer commands that answer several questions at once.
- If the issue is ambiguous, choose the most reasonable reading and say so in your final message; do not stop to ask.
{% if compaction_rule %}- {{ compaction_rule }}
{% endif %}- Calling `send_message` ends the run, and nothing can be done after it. Call it once, when the fix is verified, with a short summary of the root cause, the change, and how you verified it.
{%- if skills_prompt %}

{{ skills_prompt }}
{%- endif %}"""
)
"""Standing instructions for the coding agent; the skills section is optional."""

CODE_AGENT_TASK_PROMPT_TEMPLATE = Template(
    """Resolve the issue below in the repository at `{{ cwd }}`.

<issue>
{{ task }}
</issue>

Start by reproducing the problem, and verify your fix before you finish."""
)
"""The user message that starts a run; carries the task unmodified."""

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.

        self.system_prompt = CODE_AGENT_SYSTEM_PROMPT_TEMPLATE.render(
            system_information=format_system_information(environment),
            cwd=environment.cwd,
            tool_use_rule=TOOL_USE_RULE,
            step_budget_rule=self.step_budget_rule(),
            compaction_rule=self.compaction_rule(
                "re-read a file when you need its exact current contents"
            ),
            skills_prompt=self.skills_prompt(),
        )
        self.task_prompt = CODE_AGENT_TASK_PROMPT_TEMPLATE.render(
            cwd=environment.cwd, task=self.task
        )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.

        observations: list[dict[str, str]] = []
        for call in tool_calls:
            call_id = call.get("id", "")
            name = call.get("function", {}).get("name")

            if name not in ("execute", "send_message"):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            f"Error: unrecognized tool {name!r}. "
                            "Available tools: execute, send_message."
                        ),
                    }
                )
                continue

            try:
                arguments = json.loads(call["function"].get("arguments") or "")
            except json.JSONDecodeError as exc:
                error = f"arguments are not valid JSON ({exc})"
            else:
                error = self._argument_error(name, arguments)

            if error:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Error: invalid arguments for {name!r}: {error}.",
                    }
                )
                continue

            if name == "send_message":
                # Submitting ends the run; the patch is read from patch.txt afterwards.
                self.finished = True
                content = "Message sent."
            else:
                options = {
                    key: arguments[key]
                    for key in ("shell", "cwd", "timeout", "env")
                    if key in arguments
                }
                result = self.env.execute(arguments["command"], **options)
                shown = {"output": result["output"], "returncode": result["returncode"]}
                if result.get("exception_info"):
                    shown["exception_info"] = result["exception_info"]
                content = format_tool_output(shown)

            observations.append(
                {"role": "tool", "tool_call_id": call_id, "content": content}
            )

        return observations

    @staticmethod
    def _argument_error(name: str, arguments: Any) -> str | None:
        """Describe what is wrong with a call's parsed arguments, or None if valid."""

        if not isinstance(arguments, dict):
            return "arguments must be a JSON object"

        if name == "send_message":
            if not isinstance(arguments.get("summary"), str):
                return "'summary' is required and must be a string"
            return None

        command = arguments.get("command")
        is_argv = isinstance(command, list) and all(isinstance(a, str) for a in command)
        if not (isinstance(command, str) or is_argv):
            return "'command' is required and must be a string or a list of strings"
        return None
