from __future__ import annotations

import dataclasses
import re
import sys
import warnings
from pathlib import Path
from typing import Annotated

from arena_mcp.core import JOBS, Domain, Job, ToolError, cli
from pydantic import Field

from arena_planners import goal, registry, resolver, weights


def _unlocatable() -> str | None:
    try:
        registry.workspace_root()
    except resolver.ResolverError as exc:
        return str(exc)
    return None


domain = Domain(
    "planners",
    """Navigation planners of the arena_planners registry: list, manifests, checks, submodules and weights.
    Registry planners are tagged submodules of arena_planners (kind bridge or nav2) or local dirs with a planner.py.
    The listing also names rosnav_rl agents and nav2 controllers the resolver accepts as planner names.
    Submodule and weights operations run as jobs, poll them with jobs_status.""",
    check=_unlocatable,
)

_CHECK_LINE = re.compile(r"\[(?P<mark>[ x])\] (?P<planner>[^:]+): (?P<path>\S+)")


def _root() -> Path:
    try:
        return registry.workspace_root()
    except resolver.ResolverError as exc:
        raise ToolError(str(exc), hint="serve from an Arena checkout") from exc


def _known(root: Path, name: str) -> None:
    names = registry.all_planners(root)
    if name not in names:
        raise ToolError(
            f"no planner {name!r} in the registry, available: {', '.join(names)}", hint="planners_list shows them"
        )


def _manifest_summary(manifest: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {key: value for key, value in manifest.items() if not isinstance(value, dict)}
    out["sections"] = sorted(key for key, value in manifest.items() if isinstance(value, dict))
    return out


def _weights_presence(pdir: Path) -> dict[str, object] | None:
    try:
        entries = weights.read(pdir)
    except weights.ManifestError as exc:
        return {"error": str(exc)}
    if not entries:
        return None
    present = [entry.dest for entry in entries if (pdir / entry.dest).is_file()]
    return {
        "declared": len(entries),
        "present": len(present),
        "missing": [e.dest for e in entries if e.dest not in present],
    }


def _entry(
    root: Path, name: str, paths: dict[str, list[str]], status: dict[str, str], kinds: dict[str, str], local: set[str]
) -> dict[str, object]:
    submodules = {path: status.get(path, "unknown") for path in paths.get(name, [])}
    pdir = registry.planner_dir(name, root)
    entry: dict[str, object] = {
        "name": name,
        "kind": kinds.get(name, "bridge"),
        "local": name in local,
        "submodules": submodules,
        "dir": str(pdir) if pdir is not None else None,
    }
    if pdir is None or not (pdir / "planner.yaml").is_file():
        return entry
    try:
        manifest = resolver.load_manifest(name, workspace_root=root)
        entry["manifest"] = _manifest_summary(manifest)
        entry["vla"] = goal.declared(manifest) is not None
    except (resolver.ResolverError, OSError, ValueError) as exc:
        entry["manifest_error"] = str(exc)
    entry["weights"] = _weights_presence(pdir)
    return entry


def _status_result(name: str) -> dict[str, object]:
    root = _root()
    status = registry.submodule_status(root)
    return {
        "planner": name,
        "submodules": {path: status.get(path, "unknown") for path in registry.submodule_paths(root).get(name, [])},
    }


async def _verb(verb: str, name: str | None, *extra: str) -> dict[str, object]:
    root = _root()
    if name is not None:
        _known(root, name)
    args = [arg for arg in (*extra, name) if arg is not None]
    label = name or "all"

    def result(_job: Job) -> dict[str, object]:
        if name is None:
            status = registry.submodule_status(root)
            return {
                "submodules": {
                    path: status.get(path, "unknown")
                    for paths in registry.submodule_paths(root).values()
                    for path in paths
                }
            }
        return _status_result(name)

    job = await cli.job("planners", f"planners {verb} {label}", "feature", "planners", verb, *args, result=result)
    return job.as_dict()


@domain.tool(read_only=True)
def planners_list(
    kind: Annotated[str | None, Field(description="only registry planners of this kind, e.g. bridge or nav2")] = None,
) -> dict[str, object]:
    """Registry planners with kind, submodule status, manifest summary and weights, plus rosnav_rl and nav2 names."""
    root = _root()
    paths = registry.submodule_paths(root)
    status = registry.submodule_status(root)
    kinds = registry.kinds(root)
    local = set(registry.local_planners(root))
    entries = [_entry(root, name, paths, status, kinds, local) for name in registry.all_planners(root)]
    if kind is not None:
        present = sorted({str(entry["kind"]) for entry in entries})
        if kind not in present:
            raise ToolError(f"no registry planner of kind {kind!r}, kinds: {', '.join(present)}")
        return {"planners": [entry for entry in entries if entry["kind"] == kind]}
    available = resolver.list_available(workspace_root=root)
    return {"planners": entries, "rosnav_rl": available["rosnav_rl"], "nav2": available["nav2"]}


@domain.tool(read_only=True)
def planners_manifest(name: str) -> dict[str, object]:
    """A registry planner's planner.yaml, its resolution (source, adapter, selector) and its declared weights."""
    root = _root()
    _known(root, name)
    try:
        manifest = resolver.load_manifest(name, workspace_root=root)
    except resolver.ResolverError as exc:
        raise ToolError(str(exc), hint=f"planners_add {name} checks it out") from exc
    except FileNotFoundError as exc:
        kind = registry.kinds(root).get(name, "bridge")
        raise ToolError(
            f"planner {name!r} has no planner.yaml",
            hint="nav2 planners are configured through nav2 params"
            if kind == "nav2"
            else "planners_check shows whether it is checked out",
        ) from exc
    pdir = resolver.planner_dir(name, workspace_root=root)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            resolved: dict[str, object] | None = dict(resolver.resolve(name, workspace_root=root)._asdict())
            reason = None
        except resolver.ResolverError as exc:
            resolved, reason = None, str(exc)
    out: dict[str, object] = {
        "name": name,
        "dir": str(pdir),
        "kind": registry.kinds(root).get(name, "bridge"),
        "goal_inputs": sorted(goal.declared(manifest) or ()),
        "manifest": manifest,
        "resolved": resolved,
        "warnings": [str(warning.message) for warning in caught],
    }
    if reason is not None:
        out["unresolved"] = reason
    try:
        out["weights"] = [
            {key: value for key, value in dataclasses.asdict(entry).items() if value is not None}
            for entry in weights.read(pdir)
        ]
    except weights.ManifestError as exc:
        out["weights_error"] = str(exc)
    return out


@domain.tool(read_only=True)
def planners_check() -> dict[str, object]:
    """Submodule initialization per planner as `arena feature planners check` prints it, ok when it exits 0."""
    done = cli.run("feature", "planners", "check", check=False)
    planners: dict[str, list[dict[str, object]]] = {}
    for line in done["stdout"].splitlines():
        match = _CHECK_LINE.match(line)
        if match is not None:
            planners.setdefault(match["planner"], []).append(
                {"path": match["path"], "initialized": match["mark"] == "x"}
            )
    if done["exit_code"] not in (0, 1) or (done["exit_code"] == 1 and not planners):
        tail = (done["stderr"] or done["stdout"]).strip().splitlines()[-15:]
        raise ToolError(f"{done['command']} failed with exit code {done['exit_code']}:\n" + "\n".join(tail))
    return {
        "ok": done["exit_code"] == 0,
        "planners": [
            {"planner": name, "submodules": paths, "ok": all(path["initialized"] for path in paths)}
            for name, paths in sorted(planners.items())
        ],
    }


@domain.tool(destructive=True, open_world=True)
async def planners_add(name: str) -> dict[str, object]:
    """Job running `arena feature planners add <name>`: clone its submodules, fetch its weights, then deps and build."""
    return await _verb("add", name)


@domain.tool(destructive=True, open_world=True)
async def planners_update() -> dict[str, object]:
    """Job running `arena feature planners update`: refresh every initialized planner submodule, then deps and build."""
    return await _verb("update", None)


@domain.tool(destructive=True)
async def planners_rm(
    name: str,
    force: Annotated[
        bool, Field(description="deinit submodule paths other planners share too, they become pending")
    ] = False,
) -> dict[str, object]:
    """Job running `arena feature planners rm <name>`: deinit its submodules, local planner dirs stay."""
    return await _verb("rm", name, *(("--force",) if force else ()))


@domain.tool(destructive=True, open_world=True)
async def planners_fetch_weights(name: str) -> dict[str, object]:
    """Job running `python -m arena_planners fetch <name>`: download, sha256-verify and link its weights."""
    root = _root()
    _known(root, name)
    pdir = registry.planner_dir(name, root)

    def result(_job: Job) -> dict[str, object]:
        return {"planner": name, "weights": _weights_presence(pdir) if pdir is not None else None}

    job = await JOBS.start_process(
        "planners",
        f"fetch weights {name}",
        [sys.executable, "-m", "arena_planners", "fetch", name],
        cwd=root,
        result=result,
    )
    return job.as_dict()
