"""Tests for arena_planners.weights."""

from __future__ import annotations

import functools
import hashlib
import os
import threading
import urllib.error
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from arena_planners import weights

_SHA_A = f"    sha256: {'a' * 64}\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(pdir: Path, body: str) -> Path:
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "weights.yaml").write_text(body)
    return pdir


@pytest.fixture
def server(tmp_path: Path):
    root = tmp_path / "srv"
    root.mkdir()
    requests: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(root)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", root, requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.fixture
def served(tmp_path: Path, server):
    base, root, requests = server
    data = os.urandom(3 * 1024 * 1024 + 17)
    (root / "ckpt.pt").write_bytes(data)
    pdir = _manifest(
        tmp_path / "planner",
        f"files:\n  - url: {base}/ckpt.pt\n    dest: model/sub/ckpt.pt\n    sha256: {_sha(data)}\n",
    )
    return pdir, tmp_path / "cache", data, requests


def test_gs_url_maps_to_public_storage_endpoint():
    assert weights.http_url("gs://bucket/dir/model.ckpt") == "https://storage.googleapis.com/bucket/dir/model.ckpt"


def test_https_url_is_unchanged():
    assert weights.http_url("https://example.com/a.pt") == "https://example.com/a.pt"


def test_url_entry_downloads_verifies_and_links(served):
    pdir, cache, data, requests = served
    result = list(weights.fetch(pdir, cache))
    sha = _sha(data)
    assert result == [weights.Fetched("model/sub/ckpt.pt", len(data), "downloaded")]
    dest = pdir / "model/sub/ckpt.pt"
    assert dest.is_symlink() and dest.resolve() == (cache / sha).resolve()
    assert dest.read_bytes() == data
    assert (cache / f"{sha}.sha256-ok").read_text().strip() == sha
    assert requests == ["/ckpt.pt"]
    assert sorted(p.name for p in cache.iterdir()) == [sha, f"{sha}.sha256-ok"]


def test_cache_hit_trusts_marker_without_rehash_or_request(served):
    pdir, cache, data, requests = served
    list(weights.fetch(pdir, cache))
    blob = cache / _sha(data)
    blob.write_bytes(b"\0" * len(data))
    assert list(weights.fetch(pdir, cache)) == [weights.Fetched("model/sub/ckpt.pt", len(data), "cached")]
    assert requests == ["/ckpt.pt"]


def test_missing_marker_rehashes_cached_blob(served):
    pdir, cache, data, requests = served
    list(weights.fetch(pdir, cache))
    marker = cache / f"{_sha(data)}.sha256-ok"
    marker.unlink()
    assert [f.status for f in weights.fetch(pdir, cache)] == ["verified"]
    assert marker.is_file()
    assert requests == ["/ckpt.pt"]


def test_corrupt_unmarked_blob_is_downloaded_again(served):
    pdir, cache, data, requests = served
    cache.mkdir()
    (cache / _sha(data)).write_bytes(b"garbage")
    assert [f.status for f in weights.fetch(pdir, cache)] == ["downloaded"]
    assert (cache / _sha(data)).read_bytes() == data
    assert requests == ["/ckpt.pt"]


def test_sha_mismatch_raises_with_both_hashes_and_removes_download(tmp_path: Path, server):
    base, root, _ = server
    data = b"not the declared checkpoint"
    (root / "bad.pt").write_bytes(data)
    wrong = "e" * 64
    pdir = _manifest(
        tmp_path / "planner", f"files:\n  - url: {base}/bad.pt\n    dest: model/bad.pt\n    sha256: {wrong}\n"
    )
    cache = tmp_path / "cache"
    with pytest.raises(weights.ChecksumError) as err:
        list(weights.fetch(pdir, cache))
    assert wrong in str(err.value) and _sha(data) in str(err.value)
    assert list(cache.iterdir()) == []
    assert not (pdir / "model/bad.pt").exists()


def test_http_error_leaves_no_partial_file(tmp_path: Path, server):
    base, _, _ = server
    pdir = _manifest(tmp_path / "planner", f"files:\n  - url: {base}/absent.pt\n    dest: a.pt\n{_SHA_A}")
    cache = tmp_path / "cache"
    with pytest.raises(urllib.error.HTTPError):
        list(weights.fetch(pdir, cache))
    assert list(cache.iterdir()) == []


def test_correct_symlink_is_left_alone(served):
    pdir, cache, _, _ = served
    list(weights.fetch(pdir, cache))
    dest = pdir / "model/sub/ckpt.pt"
    before = os.lstat(dest)
    list(weights.fetch(pdir, cache))
    after = os.lstat(dest)
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_wrong_symlink_is_replaced(served, tmp_path: Path):
    pdir, cache, data, _ = served
    stray = tmp_path / "stray.pt"
    stray.write_bytes(b"stray")
    dest = pdir / "model/sub/ckpt.pt"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(stray)
    list(weights.fetch(pdir, cache))
    assert Path(os.readlink(dest)) == cache / _sha(data)
    assert stray.read_bytes() == b"stray"


def test_hf_entry_with_verified_dest_needs_no_network(tmp_path: Path):
    data = b"hf checkpoint bytes"
    pdir = _manifest(
        tmp_path / "planner",
        f"files:\n  - repo: org/model\n    filename: m.pt\n    revision: {'b' * 40}\n"
        f"    dest: model/m.pt\n    sha256: {_sha(data)}\n",
    )
    (pdir / "model").mkdir()
    (pdir / "model/m.pt").write_bytes(data)
    cache = tmp_path / "cache"
    assert [f.status for f in weights.fetch(pdir, cache)] == ["verified"]
    markers = list((cache / "hf").iterdir())
    assert len(markers) == 1 and markers[0].read_text().strip() == _sha(data)
    assert [f.status for f in weights.fetch(pdir, cache)] == ["cached"]


def test_gated_error_names_the_terms_page_and_the_login():
    message = str(weights.GatedError("nvidia/X-Mobility"))
    assert "https://huggingface.co/nvidia/X-Mobility" in message
    assert "hf auth login" in message and "HF_TOKEN" in message


def test_hf_kwargs_forward_revision():
    entry = weights.Entry(dest="m.pt", sha256="c" * 64, repo="org/model", filename="m.pt", revision="d" * 40)
    assert weights.hf_kwargs(entry) == {"repo_id": "org/model", "filename": "m.pt", "revision": "d" * 40}


def test_hf_kwargs_omit_absent_revision():
    entry = weights.Entry(dest="m.pt", sha256="c" * 64, repo="org/model", filename="m.pt")
    assert weights.hf_kwargs(entry) == {"repo_id": "org/model", "filename": "m.pt"}


def test_read_parses_hf_and_url_entries(tmp_path: Path):
    pdir = _manifest(
        tmp_path,
        f"files:\n"
        f"  - repo: org/model\n    filename: dir/m.pt\n    revision: {'e' * 40}\n    dest: model/m.pt\n"
        f"    sha256: {'A' * 64}\n"
        f"  - url: gs://bucket/x.ckpt\n    dest: model/x.ckpt\n    sha256: {'f' * 64}\n",
    )
    assert weights.read(pdir) == [
        weights.Entry(dest="model/m.pt", sha256="a" * 64, repo="org/model", filename="dir/m.pt", revision="e" * 40),
        weights.Entry(dest="model/x.ckpt", sha256="f" * 64, url="gs://bucket/x.ckpt"),
    ]


def test_read_without_manifest_is_empty(tmp_path: Path):
    assert weights.read(tmp_path) == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("  - repo: o/m\n    filename: m.pt\n    dest: m.pt\n", "`sha256`"),
        (f"  - repo: o/m\n    url: https://x/m.pt\n    filename: m.pt\n    dest: m.pt\n{_SHA_A}", "exactly one"),
        (f"  - dest: m.pt\n{_SHA_A}", "exactly one"),
        (f"  - repo: o/m\n    filename: m.pt\n    dest: m.pt\n{_SHA_A}    md5: x\n", "unknown keys ['md5']"),
        (f"  - url: https://x/m.pt\n    revision: main\n    dest: m.pt\n{_SHA_A}", "unknown keys ['revision']"),
        ("  - url: https://x/m.pt\n    dest: m.pt\n    sha256: abc\n", "64 hex"),
        (f"  - url: ftp://x/m.pt\n    dest: m.pt\n{_SHA_A}", "`url` must start"),
        (f"  - url: https://x/m.pt\n    dest: ../m.pt\n{_SHA_A}", "relative path"),
        (f"  - url: https://x/m.pt\n    dest: /m.pt\n{_SHA_A}", "relative path"),
        (f"  - repo: o/m\n    dest: m.pt\n{_SHA_A}", "`filename`"),
        (
            f"  - url: https://x/a.pt\n    dest: m.pt\n    sha256: {'a' * 64}\n"
            f"  - url: https://x/b.pt\n    dest: m.pt\n    sha256: {'b' * 64}\n",
            "duplicate `dest`",
        ),
        ("  - just-a-string\n", "expected a mapping"),
    ],
)
def test_invalid_entries_are_rejected(tmp_path: Path, body: str, message: str):
    _manifest(tmp_path, "files:\n" + body)
    with pytest.raises(weights.ManifestError) as err:
        weights.read(tmp_path)
    assert message in str(err.value)


def test_unknown_top_level_key_is_rejected(tmp_path: Path):
    _manifest(tmp_path, "files: []\nlicense: mit\n")
    with pytest.raises(weights.ManifestError, match="unknown keys"):
        weights.read(tmp_path)


def test_missing_lists_absent_dests(tmp_path: Path):
    pdir = _manifest(
        tmp_path / "planner",
        f"files:\n  - url: https://x/a.pt\n    dest: a.pt\n    sha256: {_sha(b'a')}\n"
        f"  - url: https://x/b.pt\n    dest: b.pt\n    sha256: {'b' * 64}\n",
    )
    (pdir / "a.pt").write_bytes(b"a")
    assert weights.missing(pdir, tmp_path / "cache") == ["b.pt"]


def test_link_to_another_file_than_declared_is_missing(tmp_path: Path):
    old = tmp_path / "hub" / "gst.pt"
    old.parent.mkdir()
    old.write_bytes(b"old pickle")
    pdir = _manifest(
        tmp_path / "planner",
        f"files:\n  - repo: org/model\n    filename: gst_state.pt\n    dest: model/gst.pt\n"
        f"    sha256: {_sha(b'new')}\n",
    )
    (pdir / "model").mkdir()
    (pdir / "model/gst.pt").symlink_to(old)
    assert weights.missing(pdir, tmp_path / "cache") == ["model/gst.pt"]


def test_link_to_the_declared_file_is_current(tmp_path: Path):
    blob = tmp_path / "hub" / "gst_state.pt"
    blob.parent.mkdir()
    blob.write_bytes(b"new")
    pdir = _manifest(
        tmp_path / "planner",
        f"files:\n  - repo: org/model\n    filename: gst_state.pt\n    dest: model/gst.pt\n"
        f"    sha256: {_sha(b'new')}\n",
    )
    (pdir / "model").mkdir()
    (pdir / "model/gst.pt").symlink_to(blob)
    assert weights.missing(pdir, tmp_path / "cache") == []


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"ARENA_WEIGHTS_CACHE": "/w", "HF_HOME": "/h"}, "/w"),
        ({"HF_HUB_CACHE": "/c/hub-x", "HF_HOME": "/h"}, "/c/arena-planners-weights"),
        ({"HF_HOME": "/opt/data/hf"}, "/opt/data/hf/arena-planners-weights"),
        ({"XDG_CACHE_HOME": "/x"}, "/x/huggingface/arena-planners-weights"),
    ],
)
def test_cache_root_sits_next_to_hf_hub_cache(env: dict[str, str], expected: str):
    assert weights.cache_root(env) == Path(expected)


def test_sha256_file_streams_across_blocks(tmp_path: Path):
    data = os.urandom(weights._BLOCK + 5)
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert weights.sha256_file(path) == _sha(data)
