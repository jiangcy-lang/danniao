#!/usr/bin/env python3
"""Cursor hook: after agent stop / session end, commit & push dirty work to origin.

Fail-open: never block the agent on sync failure. Reads hook JSON from stdin.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "origin"
BRANCH_FALLBACK = "main"
LOG = Path(__file__).resolve().parent / "sync_to_remote.log"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}\n"
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def run(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
        encoding="utf-8",
        errors="replace",
    )


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    raw = sys.stdin.read()
    try:
        json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        pass

    if not (ROOT / ".git").exists():
        log("skip: not a git repository")
        emit({})
        return 0

    remote = run(["git", "remote", "get-url", REMOTE])
    if remote.returncode != 0:
        log(f"skip: no remote '{REMOTE}' ({remote.stderr.strip()})")
        emit({})
        return 0

    status = run(["git", "status", "--porcelain"])
    if status.returncode != 0:
        log(f"error: git status failed: {status.stderr.strip()}")
        emit({})
        return 0

    dirty = bool(status.stdout.strip())
    branch = run(["git", "branch", "--show-current"])
    current = (branch.stdout or BRANCH_FALLBACK).strip() or BRANCH_FALLBACK

    if not dirty:
        # Still push if local commits are ahead of origin
        ahead = run(["git", "rev-list", "--count", f"{REMOTE}/{current}..HEAD"])
        if ahead.returncode != 0 or (ahead.stdout or "").strip() in ("", "0"):
            log("skip: clean working tree, nothing to push")
            emit({})
            return 0

    if dirty:
        add = run(["git", "add", "-A"])
        if add.returncode != 0:
            log(f"error: git add failed: {add.stderr.strip()}")
            emit({})
            return 0

        # Re-check after add (e.g. all ignored)
        status2 = run(["git", "status", "--porcelain"])
        if not (status2.stdout or "").strip():
            log("skip: nothing staged after add")
        else:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            msg = f"chore(sync): auto-sync after development ({ts})"
            commit = run(["git", "commit", "-m", msg])
            if commit.returncode != 0:
                # Possibly nothing to commit (hooks / empty)
                log(f"commit skipped/failed: {commit.stdout.strip()} {commit.stderr.strip()}")
            else:
                log(f"committed: {msg}")

    # Ensure upstream exists; first push uses -u
    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream.returncode != 0:
        push = run(["git", "push", "-u", REMOTE, current])
    else:
        push = run(["git", "push", REMOTE, current])

    if push.returncode != 0:
        log(f"push failed: {push.stderr.strip()}")
        emit(
            {
                "followup_message": (
                    "自动同步到 GitHub 失败，请检查 SSH 与远程仓库后手动 "
                    f"`git push -u {REMOTE} {current}`。详情见 .cursor/hooks/sync_to_remote.log"
                )
            }
        )
        return 0

    log(f"pushed to {REMOTE}/{current}")
    emit({})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # fail-open
        log(f"unhandled: {exc}")
        emit({})
        raise SystemExit(0)
