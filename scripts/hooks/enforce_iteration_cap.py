#!/usr/bin/env python3
"""PreToolUse hook: hard-denies champion-data tool calls past the ReAct
iteration cap, as a backstop to the prompt-level budget stated in
predictor.md.

EMPIRICALLY VERIFIED (2026-08-30), and fixed based on what that
verification found in the sibling hook (capture_predictor_trace.py — see
its docstring for the full story): `transcript_path` in the hook payload is
the *top-level session* transcript, shared across everything that happens
in that session — not scoped to the current subagent invocation. Counting
mcp__champion-data__* calls in it, as this script originally did, would
overcount across a regeneration (the second predictor invocation's count
would include the first invocation's tool calls too, since both land in
the same shared file) and in any session where the predictor is invoked
more than once (e.g. this file's own smoke tests). That would make the cap
trigger early on exactly the runs — regenerations — where the predictor
most needs its full budget.

The fix: read the current invocation's own isolated transcript instead —
`<session_dir>/subagents/agent-<agent_id>.jsonl`, where `<session_dir>` is
`transcript_path` with its `.jsonl` extension stripped and `agent_id` is a
field the hook payload already provides. That file contains exactly (and
only) this subagent invocation's own turns, confirmed by direct inspection.

Reads a PreToolUse hook payload from stdin (expects at least tool_name,
transcript_path, and agent_id), counts prior mcp__champion-data__*
tool_use entries in *this invocation's* isolated transcript, and denies
once the count reaches MAX_TOOL_CALLS.
"""

import json
import os
import sys

MAX_TOOL_CALLS = 8
TOOL_PREFIX = "mcp__champion-data__"


def _subagent_transcript_path(session_transcript_path: str, agent_id: str) -> str:
    session_dir, _ext = os.path.splitext(session_transcript_path)
    return os.path.join(session_dir, "subagents", f"agent-{agent_id}.jsonl")


def count_prior_calls(session_transcript_path: str, agent_id: str) -> int:
    if not agent_id:
        return 0
    transcript_path = _subagent_transcript_path(session_transcript_path, agent_id)
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

    agent_id = payload.get("agent_id", "") or payload.get("agentId", "")
    prior_calls = count_prior_calls(payload.get("transcript_path", ""), agent_id)
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
