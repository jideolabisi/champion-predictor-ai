#!/usr/bin/env python3
"""SubagentStop hook: captures the predictor subagent's actual tool calls
and reasoning text into a JSON trace file, for the `did` guardrail subagent
to review during- and post-generation.

Why this has to be a hook, not a self-report: if the Predictor summarized
its own evidence trail, a compromised or hallucinating Predictor could just
vouch for itself in that summary — defeating the point of an independent
guardrail. This reads the real transcript instead.

NOT YET EMPIRICALLY VERIFIED — same caveat as
scripts/hooks/enforce_iteration_cap.py: unconfirmed whether a project-level
SubagentStop hook actually fires for a Task-invoked subagent, vs. only
top-level session stops. Smoke-test before relying on it: run the
predictor subagent, then check that outputs/.trace/predictor_latest.json
was actually written with non-empty content.

Reads a SubagentStop hook payload from stdin (expects at least
transcript_path), walks the transcript for tool_use / tool_result content
blocks and assistant text blocks, and writes a compact JSON trace to
outputs/.trace/predictor_latest.json. Always exits 0 — a capture failure
must never block the pipeline; the `did` subagent degrades to reviewing
without a trace (and should treat that as reduced confidence, not a crash).
"""

import json
import os
import sys

TRACE_DIR = os.path.join("outputs", ".trace")
TRACE_PATH = os.path.join(TRACE_DIR, "predictor_latest.json")
MAX_TEXT_CHARS = 4000  # per block, so one huge tool result can't blow up the trace file


def _truncate(text: str) -> str:
    text = str(text)
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + "...[truncated]"


def build_trace(transcript_path: str) -> dict:
    tool_calls = []
    reasoning_text = []

    try:
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {"tool_calls": [], "reasoning_text": [], "note": "transcript not found"}

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
        trace = build_trace(transcript_path)

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
