#!/usr/bin/env python3
"""Lint the docs/ wiki: broken links, orphans, stale claims, frontmatter.

Categories:
  - broken_links: relative `](path.md)` links that do not resolve on disk.
                  Hard error (exit 1).
  - orphans:      pages under docs/ that nothing else links to. Excludes
                  docs/raw/, README.md, CONVERSATION_CONTEXT.md, log.md,
                  SCHEMA.md, TODO_*.md (these are deliberate roots).
  - stale:        pages whose frontmatter `updated` is >90 days old AND that
                  mention a date / version string outside that window.
  - frontmatter:  pages outside docs/raw/ that lack a parseable YAML
                  frontmatter block.

Usage:
    python scripts/wiki_lint.py docs/
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\]\(([^)#]+?\.md)(?:#[^)]*)?\)")
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_RE = re.compile(r"`[^`\n]+`")


def strip_code(text: str) -> str:
    text = FENCE_RE.sub("", text)
    text = INLINE_RE.sub("", text)
    return text
STALE_DAYS = 90
ROOT_BASENAMES = {"README.md", "CONVERSATION_CONTEXT.md", "log.md", "SCHEMA.md"}


def has_frontmatter(text: str) -> bool:
    return text.lstrip().startswith("---")


def parse_frontmatter(text: str) -> dict:
    if not has_frontmatter(text):
        return {}
    body = text.lstrip()
    parts = body.split("---", 2)
    if len(parts) < 3:
        return {}
    block = parts[1]
    out = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def is_root_page(path: Path) -> bool:
    name = path.name
    if name in ROOT_BASENAMES:
        return True
    if name.startswith("TODO_") and name.endswith(".md"):
        return True
    if name.startswith("00_") and name.endswith(".md"):
        return True
    return False


def collect_md(docs_root: Path) -> list[Path]:
    return sorted(p for p in docs_root.rglob("*.md"))


def lint_broken_links(files: list[Path]) -> list[tuple[Path, str, str]]:
    broken = []
    for f in files:
        text = strip_code(f.read_text(encoding="utf-8"))
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://")):
                continue
            resolved = (f.parent / target).resolve()
            if not resolved.exists():
                broken.append((f, target, str(resolved)))
    return broken


def lint_orphans(files: list[Path], docs_root: Path) -> list[Path]:
    referenced: set[Path] = set()
    for f in files:
        text = strip_code(f.read_text(encoding="utf-8"))
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://")):
                continue
            try:
                resolved = (f.parent / target).resolve()
            except Exception:
                continue
            referenced.add(resolved)
    orphans = []
    for f in files:
        if "raw" in f.relative_to(docs_root).parts[:1]:
            continue
        if is_root_page(f):
            continue
        if f.resolve() not in referenced:
            orphans.append(f)
    return orphans


def lint_stale(files: list[Path]) -> list[tuple[Path, str, str]]:
    today = dt.date.today()
    stale = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        meta = parse_frontmatter(text)
        updated_str = meta.get("updated", "")
        if not updated_str:
            continue
        try:
            updated = dt.date.fromisoformat(updated_str)
        except ValueError:
            continue
        if (today - updated).days <= STALE_DAYS:
            continue
        body_dates = DATE_RE.findall(text)
        for d_str in body_dates:
            try:
                d = dt.date.fromisoformat(d_str)
            except ValueError:
                continue
            if (today - d).days > STALE_DAYS:
                stale.append((f, updated_str, d_str))
                break
    return stale


def lint_frontmatter(files: list[Path], docs_root: Path) -> list[Path]:
    missing = []
    for f in files:
        if "raw" in f.relative_to(docs_root).parts[:1]:
            continue
        text = f.read_text(encoding="utf-8")
        if not has_frontmatter(text):
            missing.append(f)
            continue
        meta = parse_frontmatter(text)
        if not all(k in meta for k in ("title", "tags", "source", "created", "updated")):
            missing.append(f)
    return missing


def fmt(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docs_root", type=Path, nargs="?", default=REPO_ROOT / "docs")
    args = parser.parse_args()

    docs_root = args.docs_root.resolve()
    if not docs_root.exists():
        print(f"error: {docs_root} does not exist", file=sys.stderr)
        return 2

    files = collect_md(docs_root)
    broken = lint_broken_links(files)
    orphans = lint_orphans(files, docs_root)
    stale = lint_stale(files)
    bad_fm = lint_frontmatter(files, docs_root)

    print(f"# wiki lint report ({len(files)} files under {fmt(docs_root)})\n")

    print(f"## broken links ({len(broken)})")
    for f, target, _ in broken:
        print(f"  {fmt(f)} -> {target}")
    print()

    print(f"## orphans ({len(orphans)})  [warning]")
    for f in orphans:
        print(f"  {fmt(f)}")
    print()

    print(f"## stale ({len(stale)})  [warning, updated >{STALE_DAYS}d ago + outdated dates in body]")
    for f, updated, body_date in stale:
        print(f"  {fmt(f)}  updated={updated}  body_date={body_date}")
    print()

    print(f"## frontmatter missing or incomplete ({len(bad_fm)})  [warning]")
    for f in bad_fm:
        print(f"  {fmt(f)}")
    print()

    if broken:
        print(f"FAIL: {len(broken)} broken link(s)")
        return 1
    print("OK (warnings only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
