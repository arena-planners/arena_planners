"""`python -m arena_planners`: planner weights operations (fetch / check)."""

from __future__ import annotations

import argparse
import sys

from arena_planners import weights
from arena_planners.resolver import ResolverError, planner_dir, planners_root


def _select(names: list[str], want_all: bool) -> list[str]:
    if want_all:
        if names:
            print("arena_planners: --all is mutually exclusive with planner names", file=sys.stderr)
            raise SystemExit(2)
        root = planners_root()
        return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "weights.yaml").is_file())
    if not names:
        print("arena_planners: specify planner name(s) or --all", file=sys.stderr)
        raise SystemExit(2)
    return names


def cmd_fetch(args: argparse.Namespace) -> int:
    rc = 0
    for name in _select(args.names, args.all):
        try:
            pdir = planner_dir(name)
        except ResolverError as exc:
            print(f"arena_planners: {exc}", file=sys.stderr)
            rc = 1
            continue
        if not weights.read(pdir):
            print(f"arena_planners: {name}: no weights declared")
            continue
        for dest in weights.fetch(pdir):
            print(f"arena_planners: {name}: {dest}")
    return rc


def cmd_check(args: argparse.Namespace) -> int:
    rc = 0
    for name in _select(args.names, args.all):
        try:
            pdir = planner_dir(name)
        except ResolverError as exc:
            print(f"arena_planners: {exc}", file=sys.stderr)
            rc = 1
            continue
        gaps = weights.missing(pdir)
        if gaps:
            rc = 1
            if not args.quiet:
                for dest in gaps:
                    print(f"[ ] {name}: missing {dest}")
        elif not args.quiet:
            print(f"[x] {name}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(prog="arena_planners", description="planner weights operations")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_fetch = sub.add_parser("fetch", help="download declared weights via huggingface_hub")
    p_fetch.add_argument("names", nargs="*")
    p_fetch.add_argument("--all", action="store_true", help="every planner with a weights.yaml")
    p_check = sub.add_parser("check", help="report declared weights missing on disk")
    p_check.add_argument("names", nargs="*")
    p_check.add_argument("--all", action="store_true")
    p_check.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    handlers = {"fetch": cmd_fetch, "check": cmd_check}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
