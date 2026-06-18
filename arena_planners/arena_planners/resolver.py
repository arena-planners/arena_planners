"""Planner name resolver: maps a bare name to (adapter_kind, selector_key, selector_value).

Priority: registry -> rosnav_rl -> nav2. First match wins; warn on collision.
"""

from __future__ import annotations

import os
import sys
import typing
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from arena_planners import registry

try:
    from ament_index_python.packages import get_packages_with_prefixes as _get_packages_with_prefixes

    _ament_available: bool = True
except ImportError:

    def _get_packages_with_prefixes() -> dict[str, str]:  # type: ignore[misc]
        return {}

    _ament_available = False

_WARN_ONCE_SENTINEL: list[bool] = []

_ARENA_PLANNERS_SUBDIR = "arena_planners/planners"

_NAV2_BASE_CLASSES: dict[str, str] = {
    "nav2_core::Controller": "controller",
}


class Nav2Plugin(typing.NamedTuple):
    class_name: str
    kind: str


class ResolvedPlanner(typing.NamedTuple):
    name: str
    source: typing.Literal["registry", "rosnav_rl", "nav2"]
    adapter_kind: str
    selector_key: str
    selector_value: str
    package_name: str | None = None


class ResolverError(Exception): ...


def split_global_planner(value: str) -> tuple[str, str] | None:
    """Parse a `mobile.global_planner:=<family>/<kind>` value.

    Returns None for the sentinel "none". Raises ValueError for malformed input.
    Multi-slash values split on the first slash only; everything after is `kind`.
    """
    if value == "none":
        return None
    if "/" not in value:
        raise ValueError(f"global_planner must be 'none' or '<family>/<kind>'; got {value!r}")
    family, _, kind = value.partition("/")
    if not family or not kind:
        raise ValueError(f"global_planner '<family>/<kind>' segments cannot be empty; got {value!r}")
    return family, kind


def resolve_global_planner(family: str) -> tuple[Path, dict]:
    """Find `<family>.launch.py` (+ sidecar `<family>.yaml`) in any package's `share/<pkg>/launch/global_planner/`."""
    if not _ament_available:
        raise ResolverError("ament_index_python not available; source the ROS overlay")

    try:
        packages = _get_packages_with_prefixes()
    except Exception as exc:
        raise ResolverError(f"get_packages_with_prefixes() failed: {exc}") from exc

    available: set[str] = set()
    for pkg_name, prefix in packages.items():
        gp_dir = Path(prefix) / "share" / pkg_name / "launch" / "global_planner"
        if not gp_dir.is_dir():
            continue
        for entry in gp_dir.iterdir():
            if entry.suffix == ".py" and entry.name.endswith(".launch.py"):
                available.add(entry.name[: -len(".launch.py")])
        candidate = gp_dir / f"{family}.launch.py"
        if candidate.is_file():
            metadata: dict = {}
            sidecar = gp_dir / f"{family}.yaml"
            if sidecar.is_file():
                import yaml

                with open(sidecar) as f:
                    loaded = yaml.safe_load(f) or {}
                if not isinstance(loaded, dict):
                    raise ResolverError(f"{sidecar}: expected a mapping at the top level, got {type(loaded).__name__}")
                metadata = loaded
            return candidate, metadata

    raise ResolverError(f"no global_planner handler for family {family!r}; available: {sorted(available)}")


def _find_workspace_root(hint: Path | None) -> Path:
    if hint is not None:
        return hint
    match: Path | None = None
    for p in Path(__file__).resolve().parents:
        if (p / ".gitmodules").is_file() and (p / "arena_planners").is_dir():
            match = p
    if match is not None:
        return match
    raise ResolverError("cannot locate workspace root; set workspace_root= or run inside an Arena checkout")


def _registry_names(workspace_root: Path) -> list[str]:
    """Planner names from the registry SSOT: submodules unioned with local dirs."""
    return registry.all_planners(workspace_root)


def _registry_paths(workspace_root: Path) -> dict[str, list[str]]:
    """Planner name -> submodule paths (relative to root), from the registry SSOT."""
    return registry.submodule_paths(workspace_root)


def _rosnav_rl_names(workspace_root: Path) -> list[str]:
    agents_dir = workspace_root / "arena_training" / "agents"
    if not agents_dir.is_dir():
        return []
    return sorted(d.name for d in agents_dir.iterdir() if d.is_dir() and (d / "best_model.zip").is_file())


def _debug_enabled() -> bool:
    val = os.environ.get("ARENA_PLANNERS_RESOLVER_DEBUG", "")
    return val.lower() in ("1", "true", "yes")


def _dbg(msg: str) -> None:
    print(f"[arena_planners.resolver] {msg}", file=sys.stderr)


def _nav2_plugins() -> list[Nav2Plugin]:
    """Return nav2 controller and global-planner plugin records via ament_index introspection.

    Returns an empty list if ament_index_python is not importable.
    """
    debug = _debug_enabled()

    if not _ament_available:
        if debug:
            _dbg("ament_index_python not importable; nav2 plugin discovery disabled")
        if not _WARN_ONCE_SENTINEL:
            _WARN_ONCE_SENTINEL.append(True)
            warnings.warn(
                "ament_index_python not available; nav2 plugin discovery is disabled. "
                "Source your ROS install to enable it.",
                UserWarning,
                stacklevel=3,
            )
        return []

    plugins: list[Nav2Plugin] = []
    try:
        packages = _get_packages_with_prefixes()
    except Exception:
        if debug:
            _dbg("get_packages_with_prefixes() raised; returning []")
        return []

    if debug:
        _dbg(f"scanning {len(packages)} ament packages")

    for pkg_name, prefix in packages.items():
        pkg_xml = Path(prefix) / "share" / pkg_name / "package.xml"
        pkg_xml_found = pkg_xml.is_file()
        if debug:
            _dbg(f"pkg={pkg_name} prefix={prefix} package.xml={'found' if pkg_xml_found else 'MISSING'}")
        if not pkg_xml_found:
            continue
        try:
            root = ET.parse(str(pkg_xml)).getroot()
        except ET.ParseError:
            if debug:
                _dbg(f"  pkg={pkg_name}: package.xml parse error, skipping")
            continue
        export_el = root.find("export")
        has_nav2_export = export_el is not None and export_el.find("nav2_core") is not None
        if debug:
            _dbg(f"  pkg={pkg_name}: export/nav2_core={'found' if has_nav2_export else 'absent'}")
        if export_el is None:
            continue
        for nav2_el in export_el.iter("nav2_core"):
            plugin_attr = nav2_el.get("plugin", "")
            if not plugin_attr:
                continue
            pkg_share = str(Path(prefix) / "share" / pkg_name)
            plugin_path = plugin_attr.replace("${prefix}", pkg_share)
            plugin_xml = Path(plugin_path)
            plugin_xml_exists = plugin_xml.is_file()
            if debug:
                _dbg(f"  pkg={pkg_name}: plugin_xml={plugin_xml} exists={plugin_xml_exists}")
            if not plugin_xml_exists:
                continue
            try:
                proot = ET.parse(str(plugin_xml)).getroot()
            except ET.ParseError:
                if debug:
                    _dbg(f"  pkg={pkg_name}: plugin xml parse error for {plugin_xml}")
                continue
            class_els = list(proot.iter("class"))
            matched = 0
            for cls_el in class_els:
                base = cls_el.get("base_class_type") or cls_el.get("base_class", "")
                kind = _NAV2_BASE_CLASSES.get(base)
                if kind is None:
                    continue
                cls_type = cls_el.get("type") or cls_el.get("name", "")
                if not cls_type:
                    continue
                if not any(p.class_name == cls_type for p in plugins):
                    plugins.append(Nav2Plugin(class_name=cls_type, kind=kind))
                matched += 1
            if debug:
                _dbg(f"  pkg={pkg_name}: {len(class_els)} <class> elements, {matched} matched known base classes")

    return plugins


# keep the old name callable so existing tests and call-sites don't break
def _nav2_controllers() -> list[str]:
    return [p.class_name for p in _nav2_plugins()]


def _nav2_names() -> list[str]:
    return [p.class_name for p in _nav2_plugins()]


def _nav2_short_names() -> list[str]:
    return [p.class_name.split("::")[-1] for p in _nav2_plugins()]


def _nav2_match(name: str) -> Nav2Plugin | None:
    """Return Nav2Plugin if `name` matches exactly, by suffix, or by case-insensitive suffix prefix."""
    name_lower = name.lower()
    for plugin in _nav2_plugins():
        cls = plugin.class_name
        if cls == name:
            return plugin
        suffix = cls.split("::")[-1]
        if suffix.lower() == name_lower:
            return plugin
        if suffix.lower().startswith(name_lower):
            return plugin
    return None


def planners_root(workspace_root: Path | None = None) -> Path:
    """Return the planners directory (`<workspace>/arena_planners/planners`)."""
    return _find_workspace_root(workspace_root) / _ARENA_PLANNERS_SUBDIR


def planner_dir(name: str, *, workspace_root: Path | None = None) -> Path:
    """Return the directory for a registered planner (submodule or local working-tree).

    Raises ResolverError if the planner is not known or not present on disk.
    """
    root = _find_workspace_root(workspace_root)
    local = root / _ARENA_PLANNERS_SUBDIR / name
    if (local / "planner.py").is_file():
        return local
    sub_paths = _registry_paths(root)
    planner_paths = sub_paths.get(name)
    if planner_paths:
        rel = planner_paths[0]
        sub_path = root / rel
        if not sub_path.is_dir():
            raise ResolverError(
                f"planner '{name}' is registered at '{rel}' but the submodule is not checked out. "
                f"Run: arena feature planners add {name}"
            )
        return sub_path
    reg_names = _registry_names(root)
    raise ResolverError(
        f"planner '{name}' not found in registry. Available: {reg_names}. To install: arena feature planners add <name>"
    )


def load_manifest(planner_name: str, *, workspace_root: Path | None = None) -> dict:
    """Load and parse <planners>/<name>/planner.yaml. Returns the parsed dict.
    Raises ResolverError if the planner is unknown (delegates to existing planner_dir()).
    Raises FileNotFoundError if the directory exists but planner.yaml is missing.
    """
    import yaml  # noqa: PLC0415

    manifest_path = planner_dir(planner_name, workspace_root=workspace_root) / "planner.yaml"
    with open(manifest_path) as fh:
        return yaml.safe_load(fh) or {}


def installed_in_registry(name: str, *, workspace_root: Path | None = None) -> bool:
    """True if the planner's source is present (planner.py exists in the planners dir).

    The .venv / build artifacts are a runtime concern; the planner subprocess will
    surface missing deps loudly via its own ImportError.
    """
    root = _find_workspace_root(workspace_root)
    return (root / _ARENA_PLANNERS_SUBDIR / name / "planner.py").is_file()


def _registry_package_name(name: str, *, workspace_root: Path | None = None) -> str | None:
    """Read `<name>` from a registry planner's package.xml, or None if absent."""
    root = _find_workspace_root(workspace_root)
    pkg_xml = root / _ARENA_PLANNERS_SUBDIR / name / "package.xml"
    if not pkg_xml.is_file():
        return None
    try:
        tree = ET.parse(pkg_xml)
    except ET.ParseError:
        return None
    elem = tree.getroot().find("name")
    return elem.text.strip() if elem is not None and elem.text else None


_KIND_TO_SELECTOR_KEY: dict[str, str] = {
    "controller": "controller",
}


def resolve(name: str, *, workspace_root: Path | None = None) -> ResolvedPlanner:
    """Look up `name` across all sources in priority order; warn on collisions."""
    root = _find_workspace_root(workspace_root)

    reg_names = _registry_names(root)
    rl_names = _rosnav_rl_names(root)
    nav2_plugin = _nav2_match(name)

    hits: list[str] = []
    if name in reg_names:
        hits.append("registry")
    if name in rl_names:
        hits.append("rosnav_rl")
    if nav2_plugin is not None:
        hits.append("nav2")

    if not hits:
        raise ResolverError(
            f"planner '{name}' not found. "
            f"Available: registry={reg_names}, rosnav_rl={rl_names}, "
            f"nav2={_nav2_short_names()}. "
            f"To install a registry planner: arena feature planners add <name>"
        )

    chosen = hits[0]

    if len(hits) > 1:
        warnings.warn(
            f"planner name '{name}' resolves in multiple sources ({hits}); using {chosen}",
            UserWarning,
            stacklevel=2,
        )

    if chosen == "registry":
        if not installed_in_registry(name, workspace_root=root):
            raise ResolverError(
                f"planner '{name}' is registered but not installed. Run: arena feature planners add {name}"
            )
        return ResolvedPlanner(
            name=name,
            source="registry",
            adapter_kind="drl",
            selector_key="planner",
            selector_value=name,
            package_name=_registry_package_name(name, workspace_root=root),
        )

    if chosen == "rosnav_rl":
        return ResolvedPlanner(
            name=name,
            source="rosnav_rl",
            adapter_kind="rosnav_rl",
            selector_key="agent",
            selector_value=name,
        )

    assert nav2_plugin is not None
    selector_key = _KIND_TO_SELECTOR_KEY.get(nav2_plugin.kind, "controller")
    return ResolvedPlanner(
        name=name,
        source="nav2",
        adapter_kind="nav2",
        selector_key=selector_key,
        selector_value=nav2_plugin.class_name,
    )


def list_available(*, workspace_root: Path | None = None) -> dict[str, list[str]]:
    """Return {source: [names, ...]} for all sources, used by `arena feature planners ls`."""
    root = _find_workspace_root(workspace_root)
    return {
        "registry": _registry_names(root),
        "rosnav_rl": _rosnav_rl_names(root),
        "nav2": _nav2_short_names(),
    }


def nav2_diagnostics() -> dict[str, object]:
    """Return a diagnostic snapshot for programmatic debugging of nav2 plugin discovery."""
    return {
        "ament_available": _ament_available,
        "scanned_packages": _count_ament_packages(),
        "controllers": [p.class_name for p in _nav2_plugins() if p.kind == "controller"],
    }


def _count_ament_packages() -> int:
    if not _ament_available:
        return 0
    try:
        return len(_get_packages_with_prefixes())
    except Exception:
        return 0
