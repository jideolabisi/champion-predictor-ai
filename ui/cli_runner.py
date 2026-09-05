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
import uuid
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
    # Deliberately more specific than a bare "predict the championship" —
    # the DiD pre-generation guardrail scores request completeness/fit-to-
    # objective (SKILL.md step 2 / did.md's pre-generation mode), and naming
    # the exact deliverable and the tools-only constraint up front both
    # raises that score and reinforces (at the request level, on top of
    # predictor.md's own instructions) that the answer must come from
    # retrieved data, not the model's pretrained knowledge of team strength.
    grounding = (
        "grounded only in data retrieved via the provided tools (seasonal "
        "stats, roster, transactions, coaching/management changes, "
        "injuries) plus the WebSearch consensus check — not prior/"
        "pretrained knowledge of team strength or standings"
    )
    if mode == "what_if":
        return (
            f"/champion-predictor Evaluate this what-if scenario for {target_team} "
            f"in the {season} season context: {scenario}. Produce baseline and "
            f"scenario-adjusted win probabilities for all 16 NFC teams, {grounding}."
        )
    return (
        f"/champion-predictor Predict the {season} NFC championship: produce "
        f"win probabilities for all 16 NFC teams, {grounding}."
    )


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


# Maps the live event stream onto SKILL.md's fixed 9-step control flow so the
# UI's trace log can tag every line with which step produced it (e.g.
# "@did.DiD Pre-Gen"), not just which subagent. `current_phase` below is a
# single (agent, activity) pair rather than a per-subagent map because the
# steps run strictly sequentially in this ReAct architecture — never
# concurrently — so "whatever tool_use most recently declared a phase" is
# always the right phase for every line until the next one declares another.
_PHASE_MARKERS = (
    ("pre-generation", ("did", "DiD Pre-Gen")),
    ("during-generation", ("did", "DiD During-Gen")),
    ("post-generation", ("did", "DiD Post-Gen")),
)


def _detect_task_phase(label: str, tool_input: dict, prior_did_calls: int, mode_label: str) -> tuple[str, str]:
    if label == "predictor":
        return ("predictor", mode_label)
    if label == "critic":
        return ("critic", "Critique")
    if label == "did":
        blob = json.dumps(tool_input).lower()
        for needle, phase in _PHASE_MARKERS:
            if needle in blob:
                return phase
        # SKILL.md step 2/4/8 has the orchestrator put a literal "MODE:
        # pre-generation"/etc. line in the did subagent's prompt — if that
        # text isn't found verbatim (e.g. a skill-prompt wording change),
        # fall back to ordinal position: did is invoked pre/during/post-gen
        # in that fixed order on a normal pass.
        ordinal = prior_did_calls % 3
        return ("did", ("DiD Pre-Gen", "DiD During-Gen", "DiD Post-Gen")[ordinal])
    return (label, label)


def _tag_prefix(phase: tuple[str, str] | None) -> str:
    return f"@{phase[0]}.{phase[1]} " if phase else ""


def _bulletize(text: str, phase: tuple[str, str] | None) -> str:
    """Prefix a trace entry with a bullet and its activity-category tag, so
    the UI's Reasoning Trace box shows one distinguishable, labeled line per
    entry instead of an undifferentiated wall of text."""
    lines = text.split("\n")
    prefix = _tag_prefix(phase)
    out = [f"• {prefix}{lines[0]}"]
    out.extend(f"  {line}" if line else "" for line in lines[1:])
    return "\n".join(out)


def _stream_and_parse(cmd: list[str], timeout_seconds: int, mode_label: str) -> Iterator[dict[str, Any]]:
    """Launch a `claude ... --output-format stream-json` command and yield
    `{"type": "log", "text": ...}` events (tagged with the activity-category
    phase detected from the stream, via `_detect_task_phase`/`_bulletize`) as
    they arrive.

    The final yielded event is always an internal `{"type": "_result", ...}`
    marker — never forwarded to the UI — carrying whatever `run_real_predictor_stream`/
    `resume_real_predictor_stream` need to build their own public `"done"`
    event: `result_event` (the raw CLI result, if any), `returncode`, and
    (only on the failure paths) `launch_error`/`timed_out`.
    """
    task_labels: dict[str, str] = {}
    result_event: dict[str, Any] | None = None
    current_phase: tuple[str, str] | None = None
    did_call_count = 0

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
        yield {"type": "_result", "result_event": None, "returncode": None, "launch_error": str(exc)}
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

            if etype == "system" and obj.get("subtype") == "init":
                text = f"Session started (model: {obj.get('model')}, mcp: {obj.get('mcp_servers')})"
                yield {"type": "log", "text": _bulletize(text, current_phase)}
            elif etype == "assistant":
                # Rendered as an explicit ReAct Thought/Action pair per block:
                # free text is the model's reasoning for what to do next
                # (Thought), a tool_use is the action it takes on that
                # reasoning (Action). The matching Observation arrives later
                # as a separate "user" event carrying the tool_result.
                for block in obj.get("message", {}).get("content", []):
                    btype = block.get("type")
                    if btype == "text" and block.get("text", "").strip():
                        yield {"type": "log", "text": _bulletize(f"Thought: {block['text'].strip()}", current_phase)}
                    elif btype == "tool_use":
                        name = block.get("name", "tool")
                        tool_input = block.get("input", {})
                        # The subagent-launching tool has been named "Task"
                        # in some Claude Code versions and "Agent" in
                        # others (this session's own tool list calls it
                        # "Agent") — keying off the literal tool name broke
                        # phase detection entirely (every subsequent line
                        # silently inherited whatever phase Step 0 set, since
                        # this branch never matched). `subagent_type` is the
                        # stable signal: it's only ever present on a
                        # subagent-launch call, regardless of what that
                        # tool is named this version.
                        if "subagent_type" in tool_input or name in ("Task", "Agent"):
                            label = (
                                tool_input.get("subagent_type")
                                or tool_input.get("description")
                                or "subagent"
                            )
                            task_labels[block.get("id", "")] = label
                            current_phase = _detect_task_phase(label, tool_input, did_call_count, mode_label)
                            if label == "did":
                                did_call_count += 1
                            yield {"type": "log", "text": _bulletize(f"Action: Invoke {label} subagent...", current_phase)}
                        else:
                            if name == "Bash":
                                bash_cmd = str(tool_input.get("command", ""))
                                if "verify_data_integrity" in bash_cmd:
                                    current_phase = ("Process", "Integrity Check")
                                elif "validate_predictions" in bash_cmd:
                                    current_phase = ("Process", "Validate")
                            elif name == "Write" and not parent_id:
                                current_phase = ("Process", "Finalize")
                            summary = json.dumps(tool_input)[:160]
                            yield {"type": "log", "text": _bulletize(f"Action: {name}({summary})", current_phase)}
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
                    yield {"type": "log", "text": _bulletize(f"Observation: {summary}", current_phase)}
            elif etype == "result":
                result_event = obj

        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            yield {"type": "_result", "result_event": result_event, "returncode": None, "timed_out": True}
            return
    finally:
        if proc.stdout:
            proc.stdout.close()

    yield {"type": "_result", "result_event": result_event, "returncode": proc.returncode}


def _finalize_done_event(
    final: dict[str, Any],
    before_stems: set[str],
    session_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Build the public `"done"` event from `_stream_and_parse`'s internal
    `_result` marker, common to both a fresh run and a resumed one."""
    base = {"type": "done", "report_stem": None, "session_id": session_id, "awaiting_decision": False}

    if final.get("launch_error"):
        return {**base, "ok": False, "error": f"Failed to launch Claude Code CLI: {final['launch_error']}", "result_event": None}

    result_event = final.get("result_event")
    if final.get("timed_out"):
        return {**base, "ok": False, "error": f"Timed out after {timeout_seconds}s and was killed.", "result_event": result_event}

    new_stem = _newest_new_report_stem(before_stems)
    ok = bool(result_event) and not result_event.get("is_error", True)
    if not ok:
        error = (
            (result_event or {}).get("result")
            or f"Claude Code exited with code {final.get('returncode')} and produced no result event."
        )
        awaiting_decision = False
    elif new_stem is None:
        # The skill completed without an error but never wrote a report.
        # Per SKILL.md, a successful run always ends at step 9 (write the
        # report) — so "ok" with no new file always means it paused
        # somewhere upstream for a human-in-the-loop decision it couldn't
        # get headlessly (the step-0 integrity mismatch, a step-2 "clarify"
        # verdict, a step-4 suspicion escalation, a step-6/8 regeneration-cap
        # escalation, etc.). These pauses don't share fixed wording or a
        # fixed number of options — SKILL.md doesn't mandate any — so rather
        # than pattern-match specific phrasing (which silently missed
        # real pauses worded differently), treat every such pause as
        # awaiting a reply and let the user answer in their own words.
        error = result_event.get("result") if result_event else None
        awaiting_decision = True
    else:
        error = None
        awaiting_decision = False

    return {**base, "ok": ok, "error": error, "report_stem": new_stem, "result_event": result_event}


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
    - `session_id`: this run's Claude Code session ID (set explicitly via
      `--session-id` below, rather than left to the CLI to assign, so the
      caller always has it even on a run that never wrote a report) — pass
      it to `resume_real_predictor_stream` to continue this same session
    - `awaiting_decision`: True whenever the run finished without error but
      never wrote a report — per SKILL.md, that only happens when it
      paused somewhere for a human-in-the-loop decision it couldn't get
      headlessly. The UI uses this to show a free-form reply box (instead
      of just the message) so the user can answer in their own words,
      whatever that particular pause turned out to ask.
    """
    claude_bin = find_claude_cli()
    prompt = _build_prompt(mode, season, target_team, scenario)
    session_id = str(uuid.uuid4())
    before_stems = _existing_report_stems()

    cmd = [
        claude_bin,
        "-p", prompt,
        "--session-id", session_id,
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
    # is also required for --resume (see resume_real_predictor_stream) to
    # work at all.

    yield {"type": "log", "text": _bulletize(f"Launching Claude Code (session {session_id}):\n  {prompt}\n", None)}

    final: dict[str, Any] = {}
    for event in _stream_and_parse(cmd, timeout_seconds, _mode_label(mode)):
        if event["type"] == "_result":
            final = event
            break
        yield event

    yield _finalize_done_event(final, before_stems, session_id, timeout_seconds)


def _mode_label(mode: str) -> str:
    """Human-readable trace tag for the predictor's current run mode — used
    in place of the old hardcoded "Predict/What-If" phase name, which
    labeled every predictor call the same way regardless of which mode was
    actually running (confusingly implying a plain Predict run was somehow
    also doing What-If)."""
    return "What-If" if mode == "what_if" else "Predict"


def resume_real_predictor_stream(
    session_id: str,
    reply_text: str,
    mode: str = "predict",
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Continue a prior run that paused for a human-in-the-loop decision
    (its `done` event had `awaiting_decision: True` — see
    `run_real_predictor_stream`) by replying into that *same* Claude Code
    session via `--resume`, instead of starting a fresh predictor pipeline
    from scratch. Yields the same `{"type": "log"/"done", ...}` event shape.

    `max_budget_usd` should be the budget of the *original* run — this is a
    continuation of that same session/spend, not a new budget grant.
    `mode` should be the *original* run's mode ("predict" or "what_if") —
    used only for trace-tag labeling (see `_mode_label`), not sent to Claude.
    """
    claude_bin = find_claude_cli()
    before_stems = _existing_report_stems()

    cmd = [
        claude_bin,
        "--resume", session_id,
        "-p", reply_text,
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "bypassPermissions",
        "--allowedTools", *ALLOWED_TOOLS,
        "--forward-subagent-text",
        "--max-budget-usd", str(max_budget_usd),
    ]

    yield {"type": "log", "text": _bulletize(f"Resuming session {session_id}:\n  {reply_text}\n", None)}

    final: dict[str, Any] = {}
    for event in _stream_and_parse(cmd, timeout_seconds, _mode_label(mode)):
        if event["type"] == "_result":
            final = event
            break
        yield event

    yield _finalize_done_event(final, before_stems, session_id, timeout_seconds)
