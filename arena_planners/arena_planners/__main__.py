"""`python -m arena_planners`: planner registry + weights operations (ls / fetch / check)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arena_planners import registry, weights


def cmd_ls(args: argparse.Namespace) -> int:
    root = registry.workspace_root()
    paths = registry.submodule_paths(root)
    status = registry.submodule_status(root)
    local = set(registry.local_planners(root))
    for name in registry.all_planners(root):
        if name in local:
            print(f"[x] {name} (local)")
            continue
        pending = any(status.get(p) == "uninit" for p in paths[name])
        print(f"{'[ ]' if pending else '[x]'} {name}")
    return 0


def _targets(args: argparse.Namespace) -> list[tuple[str, Path | None]]:
    root = registry.workspace_root()
    if args.all:
        if args.names:
            print("arena_planners: --all is mutually exclusive with planner names", file=sys.stderr)
            raise SystemExit(2)
        names = registry.all_planners(root)
    elif args.names:
        names = args.names
    else:
        print("arena_planners: specify planner name(s) or --all", file=sys.stderr)
        raise SystemExit(2)
    return [(name, registry.planner_dir(name, root)) for name in names]


def cmd_fetch(args: argparse.Namespace) -> int:
    rc = 0
    for name, pdir in _targets(args):
        if pdir is None:
            print(f"{name}: not registered", file=sys.stderr)
            rc = 1
        elif not weights.read(pdir):
            print(f"{name}: no weights")
        elif not weights.missing(pdir):
            print(f"{name}: already downloaded")
        else:
            print(f"{name}: downloading")
            try:
                weights.fetch(pdir)
            except Exception as exc:
                print(f"{name}: download failed: {exc}", file=sys.stderr)
                rc = 1
    return rc


def cmd_check(args: argparse.Namespace) -> int:
    rc = 0
    for name, pdir in _targets(args):
        if pdir is None:
            rc = 1
            if not args.quiet:
                print(f"[ ] {name}: not registered")
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
    ap = argparse.ArgumentParser(prog="arena_planners", description="planner registry + weights")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls", help="list planners and their checkout status")
    p_fetch = sub.add_parser("fetch", help="download declared weights via huggingface_hub")
    p_fetch.add_argument("names", nargs="*")
    p_fetch.add_argument("--all", action="store_true", help="every registered planner")
    p_check = sub.add_parser("check", help="report declared weights missing on disk")
    p_check.add_argument("names", nargs="*")
    p_check.add_argument("--all", action="store_true")
    p_check.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    return {"ls": cmd_ls, "fetch": cmd_fetch, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
