#!/usr/bin/env python3
"""PreToolUse hook: hard-denies champion-data tool calls past the ReAct
iteration cap, as a backstop to the prompt-level budget stated in
predictor.md.

NOT YET EMPIRICALLY VERIFIED — see the "ReAct Iteration Cap Enforcement"
section of the implementation plan. Specifically unconfirmed: whether a
PreToolUse hook registered at the project level fires on tool calls made
inside a subagent's own loop (invoked via Task from the orchestrating
skill), or only on top-level session tool calls. Smoke-test before relying
on this: set MAX_TOOL_CALLS low, give the predictor a tool it will want to
call more than that many times, and confirm it actually gets blocked.

Reads a PreToolUse hook payload from stdin (expects at least tool_name and
transcript_path), counts prior mcp__champion-data__* tool_use entries in the
transcript, and denies once the count reaches MAX_TOOL_CALLS.
"""

import json
import sys

MAX_TOOL_CALLS = 8
TOOL_PREFIX = "mcp__champion-data__"


def count_prior_calls(transcript_path: str) -> int:
    count = 0
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = entry.get("message", {}).get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and str(block.get("name", "")).startswith(TOOL_PREFIX)
                    ):
                        count += 1
    except FileNotFoundError:
        return 0
    return count


def main() -> int:
    payload = json.load(sys.stdin)
    tool_name = payload.get("tool_name", "")

    if not tool_name.startswith(TOOL_PREFIX):
        return 0  # not our concern, allow silently

    prior_calls = count_prior_calls(payload.get("transcript_path", ""))
    if prior_calls >= MAX_TOOL_CALLS:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"ReAct iteration cap ({MAX_TOOL_CALLS} tool calls) reached. "
                    "Produce your final answer with the evidence already gathered."
                ),
            }
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
