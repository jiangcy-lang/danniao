#!/usr/bin/env python3
"""Cursor hook: after agent stop / session end, commit & push dirty work to origin.

Fail-open: never block the agent on sync failure. Reads hook JSON from stdin.

Uses commit-tree (not `git commit`) to avoid Cursor wrappers injecting unsupported
`--trailer` flags into ancient Git, and routes SSH through ssh_wrapper.cmd for
32-bit msysgit + Windows OpenSSH compatibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "origin"
BRANCH_FALLBACK = "main"
LOG = Path(__file__).resolve().parent / "sync_to_remote.log"
SSH_WRAPPER = Path(__file__).resolve().parent / "ssh_wrapper.cmd"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}\n"
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def git_env() -> dict[str, str]:
    env = os.environ.copy()
    if SSH_WRAPPER.exists():
        env["GIT_SSH"] = str(SSH_WRAPPER)
    return env


def run(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
        encoding="utf-8",
        errors="replace",
        env=git_env(),
    )


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def current_branch() -> str:
    branch = run(["git", "branch", "--show-current"])
    name = (branch.stdout or "").strip()
    if name:
        return name
    # Git < 2.22
    ref = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    name = (ref.stdout or "").strip()
    return name if name and name != "HEAD" else BRANCH_FALLBACK


def commit_with_tree(message: str) -> str | None:
    """Create a commit via write-tree/commit-tree/update-ref (no `git commit`)."""
    tree = run(["git", "write-tree"])
    if tree.returncode != 0:
        log(f"write-tree failed: {tree.stderr.strip()}")
        return None
    tree_id = tree.stdout.strip()

    cmd = ["git", "commit-tree", tree_id, "-m", message]
    head = run(["git", "rev-parse", "HEAD"])
    if head.returncode == 0 and head.stdout.strip():
        cmd.extend(["-p", head.stdout.strip()])

    created = run(cmd)
    if created.returncode != 0:
        log(f"commit-tree failed: {created.stderr.strip()}")
        return None
    commit_id = created.stdout.strip()

    branch = current_branch()
    upd = run(["git", "update-ref", f"refs/heads/{branch}", commit_id])
    if upd.returncode != 0:
        log(f"update-ref failed: {upd.stderr.strip()}")
        return None
    return commit_id


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
        # Old git: git remote -v
        remotes = run(["git", "remote", "-v"])
        if REMOTE not in (remotes.stdout or ""):
            log(f"skip: no remote '{REMOTE}'")
            emit({})
            return 0

    status = run(["git", "status", "--porcelain"])
    if status.returncode != 0:
        log(f"error: git status failed: {status.stderr.strip()}")
        emit({})
        return 0

    dirty = bool(status.stdout.strip())
    current = current_branch()

    if not dirty:
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

        status2 = run(["git", "status", "--porcelain"])
        if not (status2.stdout or "").strip():
            log("skip: nothing staged after add")
        else:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            msg = f"chore(sync): auto-sync after development ({ts})"
            commit_id = commit_with_tree(msg)
            if commit_id:
                log(f"committed {commit_id}: {msg}")
            else:
                emit(
                    {
                        "followup_message": (
                            "自动提交失败，详见 .cursor/hooks/sync_to_remote.log"
                        )
                    }
                )
                return 0

    # First push: set upstream if missing
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
                    "自动同步到 GitHub 失败，请检查 SSH 密钥与 OpenSSH 后手动 "
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
