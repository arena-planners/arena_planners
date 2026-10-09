"""Tests for arena_planners.weights."""

from __future__ import annotations

import hashlib
from pathlib import Path

from arena_planners import weights

_REPO = "arena-rosnav/toy"


def _manifest(planner_dir: Path, entries: list[dict]) -> None:
    lines = ["files:"]
    for entry in entries:
        lines.append(f"  - repo: {entry['repo']}")
        for key in ("filename", "dest", "sha256"):
            if key in entry:
                lines.append(f"    {key}: {entry[key]}")
    planner_dir.mkdir(parents=True, exist_ok=True)
    (planner_dir / "weights.yaml").write_text("\n".join(lines) + "\n")


def _snapshot_file(cache: Path, filename: str, content: bytes) -> Path:
    path = cache / f"models--{_REPO.replace('/', '--')}" / "snapshots" / "0123abcd" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _link(planner_dir: Path, dest: str, target: Path) -> None:
    path = planner_dir / dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


def test_absent_dest_is_missing(tmp_path):
    pdir = tmp_path / "toy"
    _manifest(pdir, [{"repo": _REPO, "filename": "policy.pt", "dest": "model/policy.pt"}])
    assert weights.missing(pdir) == ["model/policy.pt"]


def test_link_with_declared_sha256_is_current(tmp_path):
    pdir = tmp_path / "toy"
    blob = _snapshot_file(tmp_path / "cache", "gst_state.pt", b"new")
    _manifest(
        pdir,
        [
            {
                "repo": _REPO,
                "filename": "gst_state.pt",
                "dest": "model/gst.pt",
                "sha256": hashlib.sha256(b"new").hexdigest(),
            }
        ],
    )
    _link(pdir, "model/gst.pt", blob)
    assert weights.missing(pdir) == []


def test_stale_link_after_manifest_rename_is_missing(tmp_path):
    pdir = tmp_path / "toy"
    old = _snapshot_file(tmp_path / "cache", "gst.pt", b"old pickle")
    _manifest(
        pdir,
        [
            {
                "repo": _REPO,
                "filename": "gst_state.pt",
                "dest": "model/gst.pt",
                "sha256": hashlib.sha256(b"new").hexdigest(),
            }
        ],
    )
    _link(pdir, "model/gst.pt", old)
    assert weights.missing(pdir) == ["model/gst.pt"]


def test_link_to_declared_filename_without_sha256_is_current(tmp_path):
    pdir = tmp_path / "toy"
    blob = _snapshot_file(tmp_path / "cache", "sub/policy.pt", b"w")
    _manifest(pdir, [{"repo": _REPO, "filename": "sub/policy.pt", "dest": "model/policy.pt"}])
    _link(pdir, "model/policy.pt", blob)
    assert weights.missing(pdir) == []


def test_link_to_renamed_filename_without_sha256_is_missing(tmp_path):
    pdir = tmp_path / "toy"
    old = _snapshot_file(tmp_path / "cache", "policy_old.pt", b"w")
    _manifest(pdir, [{"repo": _REPO, "filename": "policy.pt", "dest": "model/policy.pt"}])
    _link(pdir, "model/policy.pt", old)
    assert weights.missing(pdir) == ["model/policy.pt"]


def test_link_into_other_repo_without_sha256_is_missing(tmp_path):
    pdir = tmp_path / "toy"
    other = tmp_path / "cache" / "models--arena-rosnav--other" / "snapshots" / "0123abcd" / "policy.pt"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"w")
    _manifest(pdir, [{"repo": _REPO, "filename": "policy.pt", "dest": "model/policy.pt"}])
    _link(pdir, "model/policy.pt", other)
    assert weights.missing(pdir) == ["model/policy.pt"]


def test_regular_file_without_sha256_is_current(tmp_path):
    pdir = tmp_path / "toy"
    _manifest(pdir, [{"repo": _REPO, "filename": "policy.pt", "dest": "model/policy.pt"}])
    (pdir / "model").mkdir()
    (pdir / "model" / "policy.pt").write_bytes(b"w")
    assert weights.missing(pdir) == []


def test_fetch_skips_current_entries_without_download(tmp_path):
    pdir = tmp_path / "toy"
    blob = _snapshot_file(tmp_path / "cache", "policy.pt", b"w")
    _manifest(
        pdir,
        [
            {
                "repo": _REPO,
                "filename": "policy.pt",
                "dest": "model/policy.pt",
                "sha256": hashlib.sha256(b"w").hexdigest(),
            }
        ],
    )
    _link(pdir, "model/policy.pt", blob)
    assert weights.fetch(pdir) == []
