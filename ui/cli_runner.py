"""Bridge to run the real Champion Predictor AI pipeline (the `predictor`/
`critic`/`did` subagents orchestrated by the `champion-predictor` skill,
with its MCP tools and PreToolUse/SubagentStop hooks) via the Claude Code
CLI in headless (--print) mode, streaming progress back to the UI.

This exists because Gradio is a plain Python process with no access to
Claude Code's subagents, MCP tool wiring, or hooks — the only way to run
the *real* pipeline (as opposed to the local heuristic in
prediction_service.py) is to actually launch a Claude Code session.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"

DEFAULT_MAX_BUDGET_USD = 5.0
DEFAULT_TIMEOUT_SECONDS = 900

# Tools the champion-predictor skill's control flow actually needs: Task (to
# invoke the predictor/critic/did subagents), Read/Write (steps 3, 7, 9 —
# writing the validation temp file and the final report), the exact two
# scripts the orchestrator shells out to (steps 0 and 7), the predictor's
# MCP data tools, and WebSearch (predictor.md's consensus/fallback tool).
# Deliberately not a blanket permission bypass beyond this: the predictor
# uses WebSearch (real internet access), and this whole project's design is
# about not trusting ungrounded output blindly.
ALLOWED_TOOLS = [
    "Task",
    "Read",
    "Write",
    "Bash(.venv/Scripts/python.exe scripts/verify_data_integrity.py --verify)",
    "Bash(.venv/Scripts/python.exe scripts/validate_predictions.py --input *)",
    "mcp__champion-data__*",
    "WebSearch",
]


def find_claude_cli() -> str:
    """Locate the Claude Code CLI binary.

    Tries, in order: the CLAUDE_CLI_PATH env var, `claude` on PATH, and (for
    setups where the CLI is only reachable as the binary bundled inside the
    VS Code extension, not on PATH) the extension's own install directory.
    """
    env_path = os.environ.get("CLAUDE_CLI_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    found = shutil.which("claude")
    if found:
        return found

    binary_name = "claude.exe" if os.name == "nt" else "claude"
    candidates = glob.glob(
        str(Path.home() / ".vscode*" / "extensions" / "anthropic.claude-code-*" / "resources" / "native-binary" / binary_name)
    )
    if candidates:
        return max(candidates, key=lambda p: Path(p).stat().st_mtime)

    raise RuntimeError(
        "Claude Code CLI not found. Install it with "
        "`npm install -g @anthropic-ai/claude-code` so `claude` is on PATH, "
        "or set the CLAUDE_CLI_PATH environment variable to its binary."
    )


def _build_prompt(mode: str, season: int, target_team: str | None, scenario: str | None) -> str:
    if mode == "what_if":
        return (
            f"/champion-predictor Evaluate this what-if scenario for {target_team} "
            f"in the {season} season context: {scenario}"
        )
    return f"/champion-predictor Predict the NFC championship for the {season} season."


def _existing_report_stems() -> set[str]:
    if not PREDICTIONS_DIR.exists():
        return set()
    return {f.stem for f in PREDICTIONS_DIR.iterdir() if f.is_file()}


def _newest_new_report_stem(before: set[str]) -> str | None:
    if not PREDICTIONS_DIR.exists():
        return None
    candidates = [f for f in PREDICTIONS_DIR.iterdir() if f.is_file() and f.stem not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_mtime).stem


def run_real_predictor_stream(
    mode: str,
    season: int,
    target_team: str | None = None,
    scenario: str | None = None,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Run the real predictor pipeline via headless Claude Code, yielding
    `{"type": "log", "text": ...}` progress events as they stream in.

    The final event has `"type": "done"` and carries:
    - `ok`: whether the run completed successfully
    - `error`: an error/explanation string when not ok (or when the skill
      stopped for a human-in-the-loop pause it couldn't resolve headlessly)
    - `report_stem`: the stem of the newly written file in
      outputs/predictions/, or None if no new report was written
    - `result_event`: the raw final `result` event from the CLI (cost,
      duration, turn count), or None if the process never produced one
    """
    claude_bin = find_claude_cli()
    prompt = _build_prompt(mode, season, target_team, scenario)
    before_stems = _existing_report_stems()

    cmd = [
        claude_bin,
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "bypassPermissions",
        "--allowedTools", *ALLOWED_TOOLS,
        "--forward-subagent-text",
        "--max-budget-usd", str(max_budget_usd),
    ]
    # Deliberately no --no-session-persistence: that flag stops Claude Code
    # from writing a transcript file at all, which breaks the SubagentStop
    # hook (capture_predictor_trace.py) that the `did` guardrail depends on
    # to check the predictor's claims — confirmed empirically (it degrades
    # to "transcript not found" with the flag set). A persisted session file
    # is a fine tradeoff for a UI-triggered run.

    yield {"type": "log", "text": f"Launching Claude Code:\n  {prompt}\n"}

    task_labels: dict[str, str] = {}
    result_event: dict[str, Any] | None = None

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as exc:
        yield {"type": "done", "ok": False, "error": f"Failed to launch Claude Code CLI: {exc}", "report_stem": None, "result_event": None}
        return

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield {"type": "log", "text": line}
                continue

            etype = obj.get("type")
            parent_id = obj.get("parent_tool_use_id")
            prefix = f"[{task_labels.get(parent_id, 'subagent')}] " if parent_id else ""

            if etype == "system" and obj.get("subtype") == "init":
                yield {"type": "log", "text": f"Session started (model: {obj.get('model')}, mcp: {obj.get('mcp_servers')})"}
            elif etype == "assistant":
                # Rendered as an explicit ReAct Thought/Action pair per block:
                # free text is the model's reasoning for what to do next
                # (Thought), a tool_use is the action it takes on that
                # reasoning (Action). The matching Observation arrives later
                # as a separate "user" event carrying the tool_result.
                for block in obj.get("message", {}).get("content", []):
                    btype = block.get("type")
                    if btype == "text" and block.get("text", "").strip():
                        yield {"type": "log", "text": f"{prefix}Thought: {block['text'].strip()}"}
                    elif btype == "tool_use":
                        name = block.get("name", "tool")
                        if name == "Task":
                            label = (
                                block.get("input", {}).get("subagent_type")
                                or block.get("input", {}).get("description")
                                or "subagent"
                            )
                            task_labels[block.get("id", "")] = label
                            yield {"type": "log", "text": f"{prefix}Action: Invoke {label} subagent..."}
                        else:
                            summary = json.dumps(block.get("input", {}))[:160]
                            yield {"type": "log", "text": f"{prefix}Action: {name}({summary})"}
            elif etype == "user":
                # Tool results come back as a "user" event's tool_result
                # content blocks — this is the Observation half of the
                # Thought/Action/Observation loop; the original code never
                # surfaced these at all.
                for block in obj.get("message", {}).get("content", []):
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                        )
                    content = str(content).strip()
                    if not content:
                        continue
                    summary = content if len(content) <= 300 else content[:300] + "...[truncated]"
                    yield {"type": "log", "text": f"{prefix}Observation: {summary}"}
            elif etype == "result":
                result_event = obj

        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            yield {
                "type": "done",
                "ok": False,
                "error": f"Timed out after {timeout_seconds}s and was killed.",
                "report_stem": None,
                "result_event": result_event,
            }
            return
    finally:
        if proc.stdout:
            proc.stdout.close()

    new_stem = _newest_new_report_stem(before_stems)
    ok = bool(result_event) and not result_event.get("is_error", True)
    if not ok:
        error = (
            (result_event or {}).get("result")
            or f"Claude Code exited with code {proc.returncode} and produced no result event."
        )
    elif new_stem is None:
        # The skill completed without an error but never wrote a report —
        # e.g. it paused for a human-in-the-loop confirmation it couldn't
        # get in headless mode. Surface Claude's own final text so the user
        # sees why, instead of silently reporting success with no output.
        error = result_event.get("result") if result_event else None
    else:
        error = None

    yield {
        "type": "done",
        "ok": ok,
        "error": error,
        "report_stem": new_stem,
        "result_event": result_event,
    }
