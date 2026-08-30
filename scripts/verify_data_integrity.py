#!/usr/bin/env python3
"""Checksum manifest for the local input data the predictor's tools read.

Per checkpoint 6.1: input files are already read-only to the predictor and
critic (neither has Write/Edit/Bash in its `tools:` frontmatter — this
script doesn't need to enforce that, only documents it). What this script
adds is detecting *at-rest* modification of those files between fetch and
use — it does NOT detect poisoning that happened upstream, before a fetch
script wrote the file in the first place; that would need source-level
trust verification, out of scope here.

Usage:
  python scripts/verify_data_integrity.py --generate   # (re)write the baseline manifest
  python scripts/verify_data_integrity.py              # verify against it (default)
  python scripts/verify_data_integrity.py --verify

Exits 0 either way and prints a JSON verdict on stdout — the orchestrating
skill decides what to do with a mismatch (pause for human confirmation; a
mismatch could be a legitimate re-fetch, not necessarily tampering).
"""

import argparse
import hashlib
import json
import os
import sys

MANIFEST_PATH = os.path.join("data", ".checksums.json")
WATCHED_DIRS = [
    os.path.join("data", "raw"),
    os.path.join("data", "validation"),
]


def _iter_files():
    for base in WATCHED_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for name in files:
                path = os.path.join(root, name)
                yield os.path.relpath(path, ".").replace(os.sep, "/")


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest() -> dict:
    return {rel: _hash_file(rel) for rel in sorted(_iter_files())}


def generate() -> dict:
    manifest = compute_manifest()
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return {"passed": True, "action": "generated", "file_count": len(manifest)}


def verify() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {
            "passed": False,
            "action": "verify",
            "reason": "No manifest found — run with --generate first.",
            "changed": [],
        }

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    current = compute_manifest()
    changed = []
    for rel, old_hash in baseline.items():
        new_hash = current.get(rel)
        if new_hash is None:
            changed.append({"file": rel, "status": "missing"})
        elif new_hash != old_hash:
            changed.append({"file": rel, "status": "modified"})
    for rel in current:
        if rel not in baseline:
            changed.append({"file": rel, "status": "new_untracked"})

    return {"passed": len(changed) == 0, "action": "verify", "changed": changed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    result = generate() if args.generate else verify()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
