"""Guard: parkour's MDP package no longer star-imports, and collisions stay visible.

The previous ``tasks/parkour/mdp/__init__.py`` was seven star imports. Star imports bind
eagerly and the later one silently wins; ``joint_torques_l2`` is defined in both
``isaaclab.envs.mdp`` and ``instinctlab.envs.mdp``, and a reader of the config cannot
tell which one ran. This file pins three things that would otherwise regress without
a signal:

* no module under the package star-imports
* the cross-layer name-collision set stays the measured singleton
* every ``mdp.<name>`` the parkour configs reference is in the package ``__all__``

Isaac Lab is located by path and read as text. Nothing here imports ``isaaclab``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
import sys

import instinctlab
import instinctlab.tasks.parkour.mdp as parkour_mdp

_PARKOUR_MDP = pathlib.Path(parkour_mdp.__file__).resolve().parent
_INSTINCT_MDP = pathlib.Path(instinctlab.__file__).resolve().parent / "envs" / "mdp"
_PARKOUR_CONFIG = pathlib.Path(instinctlab.__file__).resolve().parent / "tasks" / "parkour" / "config"

# Cross-layer collisions measured by AST of public FunctionDef/ClassDef names.
# A new collision must fail here rather than silently rebind through __getattr__.
_KNOWN_COLLISIONS = frozenset({"joint_torques_l2"})


def _isaaclab_mdp_root() -> pathlib.Path:
    """Isaac Lab's ``envs/mdp`` directory, found from importlib metadata without importing it.

    Editable installs do not put the package on ``sys.path`` as ``isaaclab/``, so walking
    path entries misses it. ``find_spec`` reads the finder without executing the module.
    """
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        raise AssertionError("importlib cannot locate isaaclab; the collision statistic needs its source.")
    root = pathlib.Path(spec.origin).resolve().parent / "envs" / "mdp"
    if not root.is_dir():
        raise AssertionError(f"isaaclab.envs.mdp is not a directory at {root}")
    return root


def _public_defs(root: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith(
                "_"
            ):
                names.add(node.name)
    return names


def _parkour_mdp_modules() -> list[pathlib.Path]:
    return sorted(p for p in _PARKOUR_MDP.rglob("*.py") if p.name != "__pycache__")


def test_no_parkour_mdp_module_uses_a_star_import():
    offenders: list[str] = []
    for path in _parkour_mdp_modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                offenders.append(str(path.relative_to(_PARKOUR_MDP)))
                break
    assert not offenders, f"star import in {offenders}"


def test_the_cross_layer_name_collisions_are_the_measured_set():
    layers = {
        "isaaclab.envs.mdp": _public_defs(_isaaclab_mdp_root()),
        "instinctlab.envs.mdp": _public_defs(_INSTINCT_MDP),
        "parkour.mdp": _public_defs(_PARKOUR_MDP),
    }
    owners: dict[str, set[str]] = {}
    for layer, names in layers.items():
        for name in names:
            owners.setdefault(name, set()).add(layer)
    collisions = frozenset(name for name, layers_for_name in owners.items() if len(layers_for_name) > 1)
    assert (
        collisions == _KNOWN_COLLISIONS
    ), f"cross-layer MDP name collisions changed: {sorted(collisions)} vs {sorted(_KNOWN_COLLISIONS)}"
    assert layers["isaaclab.envs.mdp"], "isaaclab layer produced no public names"
    assert layers["instinctlab.envs.mdp"], "instinctlab layer produced no public names"
    assert layers["parkour.mdp"], "parkour layer produced no public names"


def _imports_parkour_mdp(tree: ast.AST) -> bool:
    """Only Isaac-only configs bind ``mdp`` to ``instinctlab.tasks.parkour.mdp``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "instinctlab.tasks.parkour.mdp" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "instinctlab.tasks.parkour.mdp" or node.module.endswith("tasks.parkour.mdp"):
                return True
    return False


def _config_mdp_attrs() -> set[str]:
    attrs: set[str] = set()
    for path in sorted(_PARKOUR_CONFIG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        if not _imports_parkour_mdp(tree):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "mdp":
                attrs.add(node.attr)
    return attrs


def test_every_parkour_config_mdp_name_is_in_all():
    referenced = _config_mdp_attrs()
    assert referenced, "parkour config files reference no mdp attributes; the check would pass empty"
    missing = sorted(referenced - set(parkour_mdp.__all__))
    assert not missing, f"parkour configs reference mdp names that are not in __all__: {missing}"


def test_importing_the_package_does_not_import_isaaclab():
    """The lazy lookup is load-bearing: an eager bind would pull Isaac Sim into this test."""
    before = {name for name in sys.modules if name.split(".")[0] == "isaaclab"}
    importlib.reload(parkour_mdp)
    after = {name for name in sys.modules if name.split(".")[0] == "isaaclab"}
    assert after == before
