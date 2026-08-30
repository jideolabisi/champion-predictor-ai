#!/usr/bin/env python3
"""SubagentStop hook: captures the predictor subagent's actual tool calls
and reasoning text into a JSON trace file, for the `did` guardrail subagent
to review during- and post-generation.

Why this has to be a hook, not a self-report: if the Predictor summarized
its own evidence trail, a compromised or hallucinating Predictor could just
vouch for itself in that summary — defeating the point of an independent
guardrail. This reads the real transcript instead.

EMPIRICALLY VERIFIED (2026-08-30) — two real bugs found in the process:

1. `transcript_path` in the SubagentStop payload is the *top-level session*
   transcript, not the subagent's own. Reading it whole from line 1, as
   this script originally did, captured the entire session's tool calls —
   confirmed empirically: a diagnostic predictor run instructed to make
   exactly one tool call produced a 168KB trace containing 25 unrelated
   Bash calls, 24 Edits, 19 Reads, etc. from the rest of the session. This
   silently defeated the guardrail's purpose (an independent check of what
   the predictor actually did) rather than erroring visibly.

   A first attempt tried to filter the shared transcript by
   `parent_tool_use_id`, on the assumption that on-disk transcript entries
   carry that field the way the CLI's `--output-format stream-json` wire
   events do. They don't — inspecting the real per-subagent file directly
   showed entries keyed by `agentId`/`parentUuid`/`isSidechain` instead,
   with no `parent_tool_use_id` at all. That approach is gone.

   The actual fix: Claude Code already writes each subagent's own
   fully-isolated transcript to a sibling file —
   `<session_dir>/subagents/agent-<agent_id>.jsonl`, where `<session_dir>`
   is `transcript_path` with its `.jsonl` extension stripped, and
   `agent_id` is a field the SubagentStop payload already provides. No
   filtering needed — that file already contains exactly (and only) this
   subagent's own turns.

2. Per Claude Code's own hooks docs: "the transcript file is written
   asynchronously and may lag the in-memory conversation, so it may not
   yet include the current turn's most recent messages when a hook fires."
   A short bounded retry compensates for that race without meaningfully
   slowing the pipeline down.

Reads a SubagentStop hook payload from stdin (expects transcript_path and
agent_id), reads that subagent's isolated transcript file, walks it for
tool_use / tool_result content blocks and assistant text blocks, and
writes a compact JSON trace to outputs/.trace/predictor_latest.json.
Always exits 0 — a capture failure must never block the pipeline; the
`did` subagent degrades to reviewing without a trace (and should treat
that as reduced confidence, not a crash) rather than being handed a
polluted trace it would wrongly trust.
"""

import json
import os
import sys
import time

TRACE_DIR = os.path.join("outputs", ".trace")
TRACE_PATH = os.path.join(TRACE_DIR, "predictor_latest.json")
MAX_TEXT_CHARS = 4000  # per block, so one huge tool result can't blow up the trace file
READ_RETRIES = 5
READ_RETRY_DELAY_SECONDS = 0.3


def _truncate(text: str) -> str:
    text = str(text)
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + "...[truncated]"


def _subagent_transcript_path(session_transcript_path: str, agent_id: str) -> str:
    session_dir, _ext = os.path.splitext(session_transcript_path)
    return os.path.join(session_dir, "subagents", f"agent-{agent_id}.jsonl")


def _read_transcript_lines(path: str) -> list[str]:
    """Read transcript lines, retrying briefly to absorb the documented
    async lag between a subagent stopping and its transcript being flushed.
    """
    lines: list[str] = []
    for attempt in range(READ_RETRIES):
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                return lines
        except FileNotFoundError:
            pass
        if attempt < READ_RETRIES - 1:
            time.sleep(READ_RETRY_DELAY_SECONDS)
    return lines


def build_trace(session_transcript_path: str, agent_id: str) -> dict:
    if not agent_id:
        return {"tool_calls": [], "reasoning_text": [], "note": "no agent_id in hook payload — cannot locate isolated subagent transcript"}

    subagent_path = _subagent_transcript_path(session_transcript_path, agent_id)
    lines = _read_transcript_lines(subagent_path)
    if not lines:
        return {"tool_calls": [], "reasoning_text": [], "note": f"subagent transcript not found or empty: {subagent_path}"}

    tool_calls = []
    reasoning_text = []

    for line in lines:
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
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "tool_use":
                tool_calls.append({
                    "tool": block.get("name", ""),
                    "input": block.get("input", {}),
                })
            elif block_type == "tool_result":
                raw = block.get("content", "")
                if isinstance(raw, list):
                    raw = " ".join(
                        b.get("text", "") for b in raw if isinstance(b, dict)
                    )
                tool_calls.append({"observation": _truncate(raw)})
            elif block_type == "text":
                text = block.get("text", "")
                if text.strip():
                    reasoning_text.append(_truncate(text))

    return {"tool_calls": tool_calls, "reasoning_text": reasoning_text}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        transcript_path = payload.get("transcript_path", "")
        agent_id = payload.get("agent_id", "") or payload.get("agentId", "")
        trace = build_trace(transcript_path, agent_id)

        os.makedirs(TRACE_DIR, exist_ok=True)
        with open(TRACE_PATH, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)
    except Exception as exc:  # never block the pipeline on a capture failure
        try:
            os.makedirs(TRACE_DIR, exist_ok=True)
            with open(TRACE_PATH, "w", encoding="utf-8") as f:
                json.dump({"tool_calls": [], "reasoning_text": [], "error": str(exc)}, f, indent=2)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
