"""Tests for arena_planners.resolver."""

from __future__ import annotations

import pytest

import arena_planners.resolver as _resolver_mod
from arena_planners.resolver import (
    ResolvedPlanner,
    ResolverError,
    installed_in_registry,
    list_available,
    nav2_diagnostics,
    planner_dir,
    resolve,
)


def _has_ament_index() -> bool:
    return _resolver_mod._ament_available


_GITMODULES = """\
[submodule "arena_planners/planners/drlvo"]
\tpath = arena_planners/planners/drlvo
\turl = https://example.com/drlvo.git
\tplanner = drlvo
"""

_GITMODULES_WITH_UNCLONED = """\
[submodule "arena_planners/planners/drlvo"]
\tpath = arena_planners/planners/drlvo
\turl = https://example.com/drlvo.git
\tplanner = drlvo
[submodule "arena_planners/planners/ghost_planner"]
\tpath = arena_planners/planners/ghost_planner
\turl = https://example.com/ghost_planner.git
\tplanner = ghost_planner
"""


def _make_workspace(tmp_path, *, install_drlvo: bool = True) -> None:
    (tmp_path / ".gitmodules").write_text(_GITMODULES)
    (tmp_path / "arena_planners").mkdir(parents=True)
    submod = tmp_path / "arena_planners" / "planners" / "drlvo"
    submod.mkdir(parents=True)
    if install_drlvo:
        (submod / "planner.py").write_text("")
    agents = tmp_path / "arena_training" / "agents" / "jackal_drl_v1"
    agents.mkdir(parents=True)
    (agents / "best_model.zip").write_bytes(b"")


@pytest.fixture()
def ws(tmp_path):
    _make_workspace(tmp_path)
    return tmp_path


@pytest.fixture()
def ws_uncloned(tmp_path):
    _make_workspace(tmp_path, install_drlvo=False)
    return tmp_path


@pytest.fixture()
def ws_collision(tmp_path):
    _make_workspace(tmp_path)
    collision_agent = tmp_path / "arena_training" / "agents" / "drlvo"
    collision_agent.mkdir(parents=True)
    (collision_agent / "best_model.zip").write_bytes(b"")
    return tmp_path


def test_resolve_registry(ws):
    result = resolve("drlvo", workspace_root=ws)
    assert result.source == "registry"
    assert result.adapter_kind == "drl"
    assert result.selector_key == "planner"
    assert result.selector_value == "drlvo"
    assert isinstance(result, ResolvedPlanner)


def test_resolve_rosnav_rl(ws):
    result = resolve("jackal_drl_v1", workspace_root=ws)
    assert result.source == "rosnav_rl"
    assert result.adapter_kind == "rosnav_rl"
    assert result.selector_key == "agent"
    assert result.selector_value == "jackal_drl_v1"


@pytest.mark.skipif(not _has_ament_index(), reason="ament_index_python not available")
def test_resolve_nav2_by_short_name(ws):
    result = resolve("mppi", workspace_root=ws)
    assert result.source == "nav2"
    assert result.adapter_kind == "nav2"
    assert result.selector_value == "nav2_mppi_controller::MPPIController"
    assert result.selector_key == "controller"


@pytest.mark.skipif(not _has_ament_index(), reason="ament_index_python not available")
def test_resolve_nav2_full_name(ws):
    result = resolve("nav2_mppi_controller::MPPIController", workspace_root=ws)
    assert result.source == "nav2"
    assert result.selector_value == "nav2_mppi_controller::MPPIController"
    assert result.selector_key == "controller"


def test_collision_warns_and_returns_registry(ws_collision):
    with pytest.warns(UserWarning, match="resolves in multiple sources"):
        result = resolve("drlvo", workspace_root=ws_collision)
    assert result.source == "registry"
    assert result.adapter_kind == "drl"


def test_not_installed_raises(ws_uncloned):
    with pytest.raises(ResolverError, match="registered but not installed"):
        resolve("drlvo", workspace_root=ws_uncloned)


def test_not_installed_hint(ws_uncloned):
    with pytest.raises(ResolverError, match="arena feature planners add drlvo"):
        resolve("drlvo", workspace_root=ws_uncloned)


def test_unknown_name_raises(ws):
    with pytest.raises(ResolverError, match="not found"):
        resolve("nonsense", workspace_root=ws)


def test_unknown_name_message_lists_available(ws):
    with pytest.raises(ResolverError, match="registry="):
        resolve("nonsense", workspace_root=ws)
    with pytest.raises(ResolverError, match="arena feature planners add"):
        resolve("nonsense", workspace_root=ws)


def test_list_available(ws):
    result = list_available(workspace_root=ws)
    assert "registry" in result
    assert "rosnav_rl" in result
    assert "nav2" in result
    assert "drlvo" in result["registry"]
    assert "jackal_drl_v1" in result["rosnav_rl"]
    assert isinstance(result["nav2"], list)


def test_installed_in_registry_true(ws):
    assert installed_in_registry("drlvo", workspace_root=ws) is True


def test_installed_in_registry_false_uncloned(ws_uncloned):
    assert installed_in_registry("drlvo", workspace_root=ws_uncloned) is False


def test_installed_in_registry_unknown(ws):
    assert installed_in_registry("nonexistent", workspace_root=ws) is False


def _make_local_only_workspace(tmp_path, *, with_planner_py: bool = True) -> None:
    """Workspace with a local-only planner (no .gitmodules entry) and an RL agent."""
    (tmp_path / ".gitmodules").write_text("")
    planners = tmp_path / "arena_planners" / "planners" / "local_only"
    planners.mkdir(parents=True)
    if with_planner_py:
        (planners / "planner.py").write_text("")
    agents = tmp_path / "arena_training" / "agents" / "jackal_drl_v1"
    agents.mkdir(parents=True)
    (agents / "best_model.zip").write_bytes(b"")


def test_local_only_planner_listed_and_resolvable(tmp_path):
    """A local dir with planner.py but no .gitmodules entry is discovered and resolves."""
    _make_local_only_workspace(tmp_path, with_planner_py=True)
    available = list_available(workspace_root=tmp_path)
    assert "local_only" in available["registry"]
    result = resolve("local_only", workspace_root=tmp_path)
    assert result.source == "registry"
    assert result.adapter_kind == "drl"
    assert planner_dir("local_only", workspace_root=tmp_path).name == "local_only"


def test_submodule_without_planner_py_listed_but_raises_on_resolve(tmp_path):
    """A .gitmodules entry without planner.py on disk is listed but raises ResolverError."""
    (tmp_path / ".gitmodules").write_text(_GITMODULES_WITH_UNCLONED)
    (tmp_path / "arena_planners" / "planners" / "drlvo").mkdir(parents=True)
    (tmp_path / "arena_planners" / "planners" / "drlvo" / "planner.py").write_text("")
    (tmp_path / "arena_training" / "agents").mkdir(parents=True)
    available = list_available(workspace_root=tmp_path)
    assert "ghost_planner" in available["registry"]
    with pytest.raises(ResolverError, match="arena feature planners add"):
        resolve("ghost_planner", workspace_root=tmp_path)


def test_nav2_controllers_empty_when_ament_unavailable(monkeypatch, ws):
    """_nav2_controllers() returns [] and warns once when ament_index_python is absent."""
    monkeypatch.setattr(_resolver_mod, "_ament_available", False)
    monkeypatch.setattr(_resolver_mod, "_WARN_ONCE_SENTINEL", [])
    with pytest.warns(UserWarning, match="ament_index_python not available"):
        result = _resolver_mod._nav2_controllers()
    assert result == []


def test_nav2_controllers_empty_no_double_warn(monkeypatch, ws):
    """Second call does not emit another warning when sentinel is already set."""
    monkeypatch.setattr(_resolver_mod, "_ament_available", False)
    monkeypatch.setattr(_resolver_mod, "_WARN_ONCE_SENTINEL", [True])
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        result = _resolver_mod._nav2_controllers()
    assert result == []


def test_nav2_plugins_debug_env(monkeypatch, capsys):
    """Setting ARENA_PLANNERS_RESOLVER_DEBUG=1 with no ament emits at least one debug line."""
    monkeypatch.setenv("ARENA_PLANNERS_RESOLVER_DEBUG", "1")
    monkeypatch.setattr(_resolver_mod, "_ament_available", False)
    monkeypatch.setattr(_resolver_mod, "_WARN_ONCE_SENTINEL", [True])
    _resolver_mod._nav2_plugins()
    captured = capsys.readouterr()
    assert "[arena_planners.resolver]" in captured.err


def test_nav2_diagnostics_structure(monkeypatch):
    """nav2_diagnostics() returns the expected keys and types."""
    monkeypatch.setattr(_resolver_mod, "_ament_available", False)
    monkeypatch.setattr(_resolver_mod, "_WARN_ONCE_SENTINEL", [True])
    result = nav2_diagnostics()
    assert "ament_available" in result
    assert "scanned_packages" in result
    assert "controllers" in result
    assert result["ament_available"] is False
    assert result["scanned_packages"] == 0
    assert isinstance(result["controllers"], list)


def _make_fake_nav2_plugin_xml(tmp_path, class_name: str, base_class: str) -> str:
    """Write a minimal pluginlib XML and return its path string."""
    content = f"""<library path="fake_lib">
  <class type="{class_name}" base_class_type="{base_class}">
    <description>fake</description>
  </class>
</library>"""
    p = tmp_path / "plugins.xml"
    p.write_text(content)
    return str(p)


def _make_fake_package_xml(tmp_path, pkg_name: str, plugin_xml_path: str) -> None:
    """Write a minimal package.xml with nav2_core export."""
    content = f"""<?xml version="1.0"?>
<package format="3">
  <name>{pkg_name}</name>
  <export>
    <nav2_core plugin="{plugin_xml_path}"/>
  </export>
</package>"""
    share = tmp_path / "share" / pkg_name
    share.mkdir(parents=True)
    (share / "package.xml").write_text(content)


def test_nav2_plugins_controller_kind(monkeypatch, tmp_path):
    """A Controller plugin is discovered with kind='controller'."""
    plugin_xml = _make_fake_nav2_plugin_xml(tmp_path, "my_pkg::MyController", "nav2_core::Controller")
    _make_fake_package_xml(tmp_path, "my_pkg", plugin_xml)

    monkeypatch.setattr(_resolver_mod, "_ament_available", True)

    def fake_packages():
        return {"my_pkg": str(tmp_path)}

    monkeypatch.setattr(_resolver_mod, "_get_packages_with_prefixes", fake_packages)

    plugins = _resolver_mod._nav2_plugins()
    assert any(p.class_name == "my_pkg::MyController" and p.kind == "controller" for p in plugins)


def test_nav2_plugins_base_class_fallback(monkeypatch, tmp_path):
    """Plugin XML using 'base_class' (older manifests) is still recognized."""
    content = """<library path="fake_lib">
  <class type="old_pkg::OldController" base_class="nav2_core::Controller">
    <description>old manifest</description>
  </class>
</library>"""
    plugin_xml = tmp_path / "old_plugins.xml"
    plugin_xml.write_text(content)
    _make_fake_package_xml(tmp_path, "old_pkg", str(plugin_xml))

    monkeypatch.setattr(_resolver_mod, "_ament_available", True)

    def fake_packages():
        return {"old_pkg": str(tmp_path)}

    monkeypatch.setattr(_resolver_mod, "_get_packages_with_prefixes", fake_packages)

    plugins = _resolver_mod._nav2_plugins()
    assert any(p.class_name == "old_pkg::OldController" and p.kind == "controller" for p in plugins)


def test_nav2_plugins_prefix_substitution(monkeypatch, tmp_path):
    """${prefix} in package.xml resolves to <prefix>/share/<pkg_name>, not just <prefix>."""
    pkg_name = "fake_ctrl_pkg"
    share = tmp_path / "share" / pkg_name
    share.mkdir(parents=True)
    plugin_xml_content = """<library path="fake_lib">
  <class type="fake_ctrl_pkg::FakeController" base_class_type="nav2_core::Controller">
    <description>fake</description>
  </class>
</library>"""
    (share / "plugins.xml").write_text(plugin_xml_content)
    pkg_xml_content = f"""<?xml version="1.0"?>
<package format="3">
  <name>{pkg_name}</name>
  <export>
    <nav2_core plugin="${{prefix}}/plugins.xml"/>
  </export>
</package>"""
    (share / "package.xml").write_text(pkg_xml_content)

    monkeypatch.setattr(_resolver_mod, "_ament_available", True)

    def fake_packages():
        return {pkg_name: str(tmp_path)}

    monkeypatch.setattr(_resolver_mod, "_get_packages_with_prefixes", fake_packages)

    plugins = _resolver_mod._nav2_plugins()
    assert any(p.class_name == "fake_ctrl_pkg::FakeController" for p in plugins)
