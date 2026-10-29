"""Helpers for compacting the agent's context into working memory."""

from __future__ import annotations

import json
import math
from typing import Any


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


def build_compaction_old_prefix(
    old_prefix: list[dict], previous_summary: str | None = None
) -> str:
    """Serialize old agent messages as the text between the <old_prefix> tags.

    The text opens with "[previous summary]" and the summary when one is given,
    then lists each assistant message as "[assistant action]" and its text,
    followed by "[tool call] {name} {arguments}" and "[tool observation]" with
    that call's tool message content, paired by tool_call_id in call order.
    Lines are joined with newlines, with no leading or trailing newline.
    """

    observations = {
        message["tool_call_id"]: message.get("content") or ""
        for message in old_prefix
        if message.get("role") == "tool"
    }

    lines: list[str] = []
    if previous_summary:
        lines += ["[previous summary]", previous_summary]
    for message in old_prefix:
        if message.get("role") != "assistant":
            continue
        lines.append("[assistant action]")
        if message.get("content"):
            lines.append(message["content"])
        for call in message.get("tool_calls", []):
            function = call["function"]
            lines.append(f"[tool call] {function['name']} {function['arguments']}")
            lines += ["[tool observation]", observations.get(call["id"], "")]
    return "\n".join(lines)


def split_recent_steps(
    agent_state: list[dict[str, Any]], keep_recent_steps: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split agent state into the old prefix and the kept recent steps."""

    # A step starts at an assistant message and runs through its tool
    # observations, so the kept suffix starts at the Nth-from-last one.
    step_starts = [
        index
        for index, message in enumerate(agent_state)
        if message.get("role") == "assistant"
    ]
    keep_from = step_starts[-keep_recent_steps]
    return agent_state[:keep_from], agent_state[keep_from:]
