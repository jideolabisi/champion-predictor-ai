#!/usr/bin/env python3
"""Deterministic check for fabricated named coaching/management claims.

Why this exists (see memory: project-predictor-fabrication-incident,
2026-09-05): the predictor has repeatedly asserted specific coaching/GM/
front-office facts — real people's names (Sean McVay, Les Snead, Raheem
Morris, Terry Fontenot, Kevin Patullo, ...) — that do not appear anywhere in
that run's actual tool observations, then explicitly mislabeled them as
"confirmed via get_transactions". This happened three times in one day,
including immediately after `predictor.md` was patched with an explicit
prose instruction never to do this. Prose-only guardrails (predictor.md's
own rules, and the `did` subagent's judgment) have hit a ceiling for this
specific failure — this script is the architectural backstop: it does not
ask an LLM whether a claim looks grounded, it mechanically checks whether
the named person the predictor cited actually appears, verbatim (modulo
punctuation/case), anywhere in this run's trace.

Method:
  1. Scan text = prediction['explanation'] + prediction['delta_explanation']
     (if present) + every entry in the trace's `reasoning_text` (this also
     covers the predictor's loose prose outside the graded JSON — the
     SubagentStop hook captures that as reasoning_text too, which is
     exactly where the second incident's fabrication lived).
  2. Split into sentences; keep only sentences that mention a coaching/
     management role cue (head coach, GM, offensive coordinator, president
     of football operations, front office, etc.) — this scopes the check
     to exactly the failure class observed, not general fact-checking.
  3. Within each flagged sentence, extract Title-Case 2-3 word spans as
     candidate person names (a plain regex, not real NER — good enough
     here because the failure mode is inventing full names, not subtle
     misspellings).
  4. Drop candidates that are known team/city names (not people).
  5. For every remaining candidate, verify it appears (case/punctuation-
     normalized) somewhere in the trace's tool_calls observations —
     ground truth for what was actually retrieved this run, including
     WebSearch results, which appear as ordinary tool_calls entries.
     Anything that doesn't appear is a confirmed fabrication: not "the
     `did` guardrail is suspicious of this," but "this specific string
     the predictor cited does not exist anywhere in what it actually
     retrieved."

Scope / limitations (deliberately narrow, complements — does not replace —
the `did` guardrail's judgment-based checks):
  - Only catches invented *names*. A fabricated claim about a real person
    already present in the trace for an unrelated reason (a wrong date, a
    misattributed quote) will not be caught here.
  - Regex name-extraction can miss single-word titles/nicknames or produce
    the occasional benign false positive (a real name written slightly
    differently than in the trace) — treat a "fail" verdict as a strong,
    specific signal to inspect, not an infallible oracle.

Usage:
  python scripts/verify_coaching_claims.py --prediction path/to/pred.json \
      [--trace outputs/.trace/predictor_latest.json]

Exits 0 and prints a JSON verdict on stdout, mirroring
scripts/validate_predictions.py's contract so the orchestrating skill can
parse it the same way.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = REPO_ROOT / "outputs" / ".trace" / "predictor_latest.json"

# Phrases matched case-insensitively as substrings (with word boundaries).
_ROLE_PHRASES = [
    "head coach", "general manager", "offensive coordinator",
    "defensive coordinator", "special teams coordinator",
    "president of football operations", "president of football",
    "front office", "front-office", "executive vice president",
    "director of player personnel", "quarterbacks coach",
    "interim head coach", "interim general manager",
]
# Abbreviations matched as case-sensitive whole words only — lowercase "gm"
# or "hc" inside ordinary prose is not a role cue, but the predictor writes
# these consistently capitalized ("HC Sean McVay", "GM Les Snead").
_ROLE_ABBR = ["HC", "GM", "OC", "DC", "EVP"]

_ROLE_PHRASE_RE = re.compile(
    "|".join(re.escape(p) for p in _ROLE_PHRASES), re.IGNORECASE
)
_ROLE_ABBR_RE = re.compile(r"\b(?:" + "|".join(_ROLE_ABBR) + r")\b")

# Title-Case 2-3 word spans, e.g. "Sean McVay", "Terry Fontenot", "Matt Ryan".
# Requires each word to start uppercase and continue lowercase, which
# naturally excludes ALL-CAPS team codes (ATL, PHI, ...) and role
# abbreviations (HC, GM) from being mistaken for candidate names.
_NAME_RE = re.compile(
    r"\b[A-Z][a-z'\.\-]+(?:\s+[A-Z][a-z'\.\-]+){1,2}\b"
)

# A trailing possessive ("Green Bay's", "Sean McVay's") is part of the
# same Title-Case span _NAME_RE matches, so it must be stripped before
# comparing against team stopwords or the grounded corpus — otherwise
# "Green Bay's" survives the "Green Bay" stopword check (it's a
# different string) and then fails corpus verification too, since
# _normalize only strips the apostrophe, not the trailing "s"
# ("Green Bay's" -> "green bays", which never appears in a corpus that
# only has "Green Bay"). This affects any possessive name, not just
# team names.
_POSSESSIVE_SUFFIX_RE = re.compile(r"['’]s\b")


def _strip_possessive(name: str) -> str:
    return _POSSESSIVE_SUFFIX_RE.sub("", name)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Team full names and their component city/nickname bigrams — these are
# Title-Case 2-3 word spans too, and would otherwise be mistaken for
# candidate person names.
_TEAM_STOPWORDS = {
    "Arizona Cardinals", "Atlanta Falcons", "Carolina Panthers",
    "Chicago Bears", "Dallas Cowboys", "Detroit Lions", "Green Bay",
    "Green Bay Packers", "Los Angeles", "Los Angeles Rams",
    "Minnesota Vikings", "New Orleans", "New Orleans Saints", "New York",
    "New York Giants", "Philadelphia Eagles", "Seattle Seahawks",
    "San Francisco", "San Francisco 49Ers", "Tampa Bay",
    "Tampa Bay Buccaneers", "Washington Commanders", "New England",
    "New England Patriots",
}
_EXTRA_STOPWORDS = {
    "Super Bowl", "Pro Bowl", "All Pro", "Week 1",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[.\-'’]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_scan_text(prediction: dict, trace: dict) -> str:
    parts = [
        prediction.get("explanation") or "",
        prediction.get("delta_explanation") or "",
    ]
    parts.extend(trace.get("reasoning_text") or [])
    return "\n".join(p for p in parts if p)


def _flagged_sentences(scan_text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(scan_text)
    return [
        s for s in sentences
        if _ROLE_PHRASE_RE.search(s) or _ROLE_ABBR_RE.search(s)
    ]


def _candidate_names(sentences: list[str]) -> list[str]:
    seen = set()
    candidates = []
    for sentence in sentences:
        for match in _NAME_RE.findall(sentence):
            name = _strip_possessive(match)
            if name in _TEAM_STOPWORDS or name in _EXTRA_STOPWORDS:
                continue
            if name not in seen:
                seen.add(name)
                candidates.append(name)
    return candidates


def _build_grounded_corpus(trace: dict) -> str:
    observations = [
        c.get("observation", "")
        for c in (trace.get("tool_calls") or [])
        if "observation" in c
    ]
    return _normalize(" ".join(observations))


def verify(prediction: dict, trace: dict) -> dict:
    scan_text = _extract_scan_text(prediction, trace)
    flagged_sentences = _flagged_sentences(scan_text)
    candidates = _candidate_names(flagged_sentences)
    corpus = _build_grounded_corpus(trace)

    trace_available = bool(trace.get("tool_calls"))
    unverified = []
    verified = []
    for name in candidates:
        if _normalize(name) in corpus:
            verified.append(name)
        else:
            unverified.append(name)

    passed = not unverified
    return {
        "checked": trace_available,
        "candidates_checked": candidates,
        "verified_names": verified,
        "unverified_claims": unverified,
        "passed": passed if trace_available else None,
        "verdict": (
            "fail" if (trace_available and unverified)
            else ("pass" if trace_available else "unchecked")
        ),
        "reason": (
            f"{len(unverified)} named coaching/management claim(s) appear in "
            f"role-cue sentences but do not appear anywhere in this run's "
            f"actual tool observations: {unverified}. Treat as confirmed "
            f"fabrication, not a suspicion signal — escalate to HITL "
            f"regardless of the regeneration counter."
            if unverified else
            "No unverified named coaching/management claims found."
            if trace_available else
            "Trace file missing or empty — this check could not run; "
            "treat coaching/management sourcing as unverified by this "
            "script (not the same as verified)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, help="Path to predictor JSON output.")
    parser.add_argument(
        "--trace", default=str(DEFAULT_TRACE_PATH),
        help="Path to the captured trace JSON (default: outputs/.trace/predictor_latest.json).",
    )
    args = parser.parse_args()

    prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))

    trace_path = Path(args.trace)
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    else:
        trace = {}

    result = verify(prediction, trace)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
