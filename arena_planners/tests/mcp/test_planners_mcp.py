from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError, cli

from arena_planners import goal, registry, resolver, weights
from arena_planners.mcp import (
    planners_add,
    planners_check,
    planners_fetch_weights,
    planners_list,
    planners_manifest,
    planners_rm,
)

ROOT = registry.workspace_root()


def _with_manifest() -> list[str]:
    names = []
    for name in registry.all_planners(ROOT):
        pdir = registry.planner_dir(name, ROOT)
        if pdir is not None and (pdir / "planner.yaml").is_file():
            names.append(name)
    return names


def test_list_covers_the_registry_with_kinds_and_status():
    out = planners_list()
    entries = {entry["name"]: entry for entry in out["planners"]}
    assert sorted(entries) == registry.all_planners(ROOT)
    kinds = registry.kinds(ROOT)
    status = registry.submodule_status(ROOT)
    local = set(registry.local_planners(ROOT))
    for name, entry in entries.items():
        assert entry["kind"] == kinds.get(name, "bridge")
        assert entry["local"] == (name in local)
        paths = registry.submodule_paths(ROOT).get(name, [])
        assert entry["submodules"] == {path: status.get(path, "unknown") for path in paths}
    available = resolver.list_available(workspace_root=ROOT)
    assert out["rosnav_rl"] == available["rosnav_rl"]
    assert out["nav2"] == available["nav2"]


def test_list_summarizes_manifests_and_weights_from_the_tree():
    names = _with_manifest()
    if not names:
        pytest.skip("no planner with a planner.yaml is checked out")
    entries = {entry["name"]: entry for entry in planners_list()["planners"]}
    for name in names:
        entry = entries[name]
        manifest = resolver.load_manifest(name, workspace_root=ROOT)
        assert entry["manifest"]["action_type"] == manifest["action_type"]
        assert "observations" in entry["manifest"]["sections"]
        assert entry["vla"] == (goal.declared(manifest) is not None)
        try:
            declared = weights.read(registry.planner_dir(name, ROOT))
        except weights.ManifestError as exc:
            assert entry["weights"] == {"error": str(exc)}
            continue
        if declared:
            assert entry["weights"]["declared"] == len(declared)
            assert entry["weights"]["present"] + len(entry["weights"]["missing"]) == len(declared)
        else:
            assert entry["weights"] is None


def test_list_filters_by_kind_and_rejects_an_unknown_kind():
    kinds = set(registry.kinds(ROOT).values())
    for kind in kinds:
        filtered = planners_list(kind=kind)["planners"]
        assert filtered and all(entry["kind"] == kind for entry in filtered)
    with pytest.raises(ToolError, match="kinds"):
        planners_list(kind="no_such_kind")


def test_manifest_matches_the_planner_yaml_and_weights_yaml():
    names = _with_manifest()
    if not names:
        pytest.skip("no planner with a planner.yaml is checked out")
    for name in names:
        out = planners_manifest(name)
        assert out["manifest"] == resolver.load_manifest(name, workspace_root=ROOT)
        try:
            declared = [entry.dest for entry in weights.read(registry.planner_dir(name, ROOT))]
        except weights.ManifestError as exc:
            assert out["weights_error"] == str(exc)
        else:
            assert [entry["dest"] for entry in out["weights"]] == declared
    if out["resolved"] is not None:
        assert out["resolved"]["source"] == "registry"
        assert out["resolved"]["selector_value"] == name


def test_manifest_and_jobs_refuse_an_unknown_planner():
    with pytest.raises(ToolError, match="planners_list"):
        planners_manifest("no_such_planner")
    for start in (planners_add, planners_rm, planners_fetch_weights):
        with pytest.raises(ToolError, match="no planner"):
            asyncio.run(start("no_such_planner"))


@pytest.mark.skipif(not cli.sourced(), reason="needs a shell that sourced arena")
def test_check_reports_what_the_cli_prints_per_planner():
    status = registry.submodule_status(ROOT)
    subs = registry.submodule_paths(ROOT)
    out = planners_check()
    assert [entry["planner"] for entry in out["planners"]] == sorted(name for name, paths in subs.items() if paths)
    for entry in out["planners"]:
        paths = subs[entry["planner"]]
        pending = [path for path in paths if status.get(path) != "init"]
        expected = (
            [{"path": path, "initialized": False} for path in pending]
            if pending
            else [{"path": path, "initialized": True} for path in paths]
        )
        assert entry["submodules"] == expected
        assert entry["ok"] == (not pending)
    assert out["ok"] == all(entry["ok"] for entry in out["planners"])


def test_domain_serves_every_tool_through_the_server():
    from mcp.server.mcpserver import MCPServer

    from arena_planners.mcp import domain

    assert domain.unavailable() is None
    server = MCPServer(name="planners_test")
    domain.install(server)

    async def scenario():
        tools = await server.list_tools()
        listed = await server.call_tool("planners_list", {"kind": "nav2"})
        return tools, listed

    tools, listed = asyncio.run(scenario())
    assert sorted(tool.name for tool in tools) == sorted(spec.name for spec in domain.tools)
    assert not listed.is_error
    nav2 = [entry["name"] for entry in listed.structured_content["planners"]]
    assert nav2 == sorted(name for name, kind in registry.kinds(ROOT).items() if kind == "nav2")
