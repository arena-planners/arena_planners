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
    kinds = registry.kinds(root)
    local = set(registry.local_planners(root))
    for name in registry.all_planners(root):
        if name in local:
            print(f"[x] {name} (local)")
            continue
        pending = any(status.get(p) == "uninit" for p in paths[name])
        suffix = " (nav2)" if kinds.get(name) == "nav2" else ""
        print(f"{'[ ]' if pending else '[x]'} {name}{suffix}")
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


def _size(n: int) -> str:
    exp = min(3, max(0, (n.bit_length() - 1) // 10))
    return f"{n} B" if exp == 0 else f"{n / 1024**exp:.1f} {('KiB', 'MiB', 'GiB')[exp - 1]}"


def cmd_fetch(args: argparse.Namespace) -> int:
    rc = 0
    for name, pdir in _targets(args):
        if pdir is None:
            print(f"{name}: not registered", file=sys.stderr)
            rc = 1
            continue
        try:
            if not weights.read(pdir):
                print(f"{name}: no weights")
                continue
            for item in weights.fetch(pdir):
                print(f"{name}: {item.dest} {_size(item.size)} {item.status}", flush=True)
        except weights.GatedError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            rc = 1
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
        try:
            gaps = weights.missing(pdir)
        except weights.ManifestError as exc:
            rc = 1
            print(f"[ ] {name}: bad weights.yaml: {exc}", file=sys.stderr)
            continue
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
    p_fetch = sub.add_parser("fetch", help="download, verify (sha256) and link declared weights")
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
