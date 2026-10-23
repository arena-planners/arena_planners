"""Tests for the install status in arena_planners.registry."""

from __future__ import annotations

from arena_planners import registry

_GITMODULES = """\
[submodule "planners/drlvo"]
\tpath = planners/drlvo
\turl = https://example.com/drlvo.git
\tplanner = drlvo
[submodule "planners/navid"]
\tpath = planners/navid
\turl = https://example.com/navid.git
\tplanner = navid
[submodule "planners/cma"]
\tpath = planners/cma
\turl = https://example.com/cma.git
\tplanner = cma
"""


def _workspace(tmp_path):
    sdk = tmp_path / "arena_planners"
    (sdk / "planners" / "drlvo").mkdir(parents=True)
    (sdk / "planners" / "navid").mkdir()
    (sdk / ".gitmodules").write_text(_GITMODULES)
    (sdk / "planners" / "drlvo" / ".git").write_text("gitdir: elsewhere\n")
    return tmp_path


def test_status_follows_the_checkout_without_a_git_repository(tmp_path):
    assert registry.submodule_status(_workspace(tmp_path)) == {
        "arena_planners/planners/drlvo": "init",
        "arena_planners/planners/navid": "uninit",
        "arena_planners/planners/cma": "uninit",
    }


def test_status_counts_a_git_directory_as_checked_out(tmp_path):
    ws = _workspace(tmp_path)
    (ws / "arena_planners" / "planners" / "cma" / ".git").mkdir(parents=True)
    assert registry.submodule_status(ws)["arena_planners/planners/cma"] == "init"


def test_status_is_empty_without_the_sdk(tmp_path):
    assert registry.submodule_status(tmp_path) == {}
