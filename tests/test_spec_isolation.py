"""Guard: ``instinctlab.spec`` imports with no physics engine installed.

A task declared in ``spec/`` is supposed to be readable, comparable and compilable without deciding
which engine will run it. The moment anything under ``spec/`` imports an engine -- even for a type
annotation evaluated at runtime -- that property is gone, and it goes quietly, because the developer
who breaks it has both engines installed.

So the import is exercised here with the engines blocked outright. Two checks, because they fail
differently: a static one that names the offending module in the failure, and a dynamic one that
catches an engine pulled in through a chain of otherwise innocent imports.

The same applies to the shared machinery in ``engines/`` -- ``base.py``, ``registry.py`` and
``compile.py``, but not the ``engines/<name>/`` packages, which exist to import an engine. Keeping
the machinery clean is what lets the launcher inspect which adapters exist before deciding which
engine to bootstrap, and lets a task be checked against an engine that is not installed.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

import instinctlab
import instinctlab.spec

_ENGINE_ROOTS = frozenset({"isaaclab", "isaacsim", "mjlab", "omni", "pxr", "carb", "mujoco", "warp", "usd"})


def _spec_modules() -> list[pathlib.Path]:
    root = pathlib.Path(instinctlab.spec.__file__).parent
    return sorted(root.rglob("*.py"))


def _imports(source: pathlib.Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


def test_spec_package_is_not_empty() -> None:
    """Otherwise the checks below would pass by having nothing to check."""
    assert len(_spec_modules()) >= 2


@pytest.mark.parametrize("source", _spec_modules(), ids=lambda p: p.name)
def test_no_engine_appears_in_spec_imports(source: pathlib.Path) -> None:
    """Static read of every import statement, including ones inside functions."""
    imported = {name.split(".")[0] for name in _imports(source)}
    leaked = imported & _ENGINE_ROOTS
    assert not leaked, f"{source.name} imports {sorted(leaked)}; spec/ must compile without an engine"


def test_spec_imports_with_engines_blocked() -> None:
    """Dynamic check, which also covers engines reached transitively."""

    class _Blocker:
        def find_module(self, name, path=None):  # noqa: D102 - legacy finder protocol, enough for importlib
            return self if name.split(".")[0] in _ENGINE_ROOTS else None

        def load_module(self, name):  # pragma: no cover - only reached on regression
            raise ImportError(f"spec/ must not need {name}")

    blocker = _Blocker()
    evicted = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in _ENGINE_ROOTS or name.startswith("instinctlab.spec")
    }
    for name in evicted:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        module = importlib.import_module("instinctlab.spec")
        ref = module.EntityRef(bodies=".*_ankle_roll_link", preserve_order=True)
        assert ref.kinds() == {"body"}
    finally:
        sys.meta_path.remove(blocker)
        # Put the originals back. Re-importing left a second copy of every class in ``spec``, and a
        # later test comparing one of them by identity -- ``AgentSpec.resolve() is SimSpec`` --
        # would fail against a class that is by every other measure the same one.
        sys.modules.update(evicted)


def test_the_task_declaration_loads_without_an_engine() -> None:
    """The declaration of a cross-engine task has to be readable in a process running either engine.

    Isaac Lab is installed on every machine these tests run on, so an accidental dependency on it
    would pass here and fail on a machine that has only mjlab -- the worst place to find out. The
    import is therefore done with both engines cut off, the same way ``instinctlab.spec`` is above.

    Shadowing is part of the shared declarations too. There is no engine-specific registration
    fallback: every active task must pass through this same import-safe registry.
    """

    class _Blocker:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] in _ENGINE_ROOTS else None

        def load_module(self, name):  # pragma: no cover - only reached on regression
            raise ImportError(f"the task declaration must not need {name}")

    from tests.task_specs import task_spec

    flat_robot = task_spec("Instinct-Velocity-Flat-G1").robot
    rough_robot = task_spec("Instinct-Velocity-Rough-G1").robot
    parkour_robot = task_spec("Instinct-Parkour-Target-G1").robot
    shadowing_robot = task_spec("Instinct-Shadowing-WholeBody-Plane-G1-v0").robot
    blocker = _Blocker()
    reloaded = "instinctlab.tasks.locomotion.config"
    shadowing_decl = "instinctlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg"
    parkour_decl = "instinctlab.tasks.parkour.config.g1"
    evicted = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in _ENGINE_ROOTS or name.startswith((reloaded, shadowing_decl, parkour_decl))
    }
    for name in evicted:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        declaration = importlib.import_module(f"{reloaded}.g1.flat_env_cfg")
        rough = importlib.import_module(f"{reloaded}.g1.rough_env_cfg")
        agent = importlib.import_module(f"{reloaded}.g1.agents.instinct_rl_ppo_cfg")
        parkour = importlib.import_module("instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg")
        parkour_agent = importlib.import_module("instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg")
        assert declaration.flat_g1(flat_robot).task_id == "Instinct-Velocity-Flat-G1"
        assert rough.rough_g1(rough_robot).task_id == "Instinct-Velocity-Rough-G1"
        assert agent.G1FlatPPORunnerCfg().max_iterations > 0
        assert parkour.parkour_target_g1(parkour_robot).task_id == "Instinct-Parkour-Target-G1"
        assert parkour_agent.G1ParkourTargetPPORunnerCfg().num_steps_per_env == 24
        shadowing = importlib.import_module(shadowing_decl)
        shadowing.g1_plane_shadowing(shadowing_robot).validate()
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(evicted)


def _engine_machinery() -> list[pathlib.Path]:
    """The engine-free part of ``engines/``: its top-level modules, not the per-engine packages."""
    import instinctlab.engines

    root = pathlib.Path(instinctlab.engines.__file__).parent
    return sorted(path for path in root.glob("*.py"))


def test_engine_machinery_is_not_empty() -> None:
    assert len(_engine_machinery()) >= 3


@pytest.mark.parametrize("source", _engine_machinery(), ids=lambda p: p.name)
def test_no_engine_appears_in_the_shared_machinery(source: pathlib.Path) -> None:
    """``compile.py`` reaches an engine only through ``compat``, which imports it inside a call."""
    imported = {name.split(".")[0] for name in _imports(source)}
    leaked = imported & _ENGINE_ROOTS
    assert not leaked, f"{source.name} imports {sorted(leaked)}; engines/ machinery must stay engine-free"


def test_tasks_do_not_import_engine_implementations() -> None:
    root = pathlib.Path(instinctlab.__file__).parent / "tasks"
    violations = {
        str(path.relative_to(root)): sorted(
            name for name in _imports(path) if name == "instinctlab.engines" or name.startswith("instinctlab.engines.")
        )
        for path in root.rglob("*.py")
    }
    violations = {path: imports for path, imports in violations.items() if imports}
    assert violations == {}


def test_engine_implementations_do_not_import_tasks_concrete_assets_or_each_other() -> None:
    root = pathlib.Path(instinctlab.__file__).parent / "engines"
    violations: dict[str, list[str]] = {}
    for engine, other in (("isaacsim", "mjlab"), ("mjlab", "isaacsim")):
        for path in (root / engine).rglob("*.py"):
            forbidden = [
                name
                for name in _imports(path)
                if name == "instinctlab.tasks"
                or name.startswith("instinctlab.tasks.")
                    or name == "instinctlab.assets"
                    or (
                        name.startswith("instinctlab.assets.")
                        and name != "instinctlab.assets.registry"
                    )
                or name == f"instinctlab.engines.{other}"
                or name.startswith(f"instinctlab.engines.{other}.")
            ]
            if forbidden:
                violations[str(path.relative_to(root))] = sorted(forbidden)
    assert violations == {}


def _shared_layer() -> list[pathlib.Path]:
    """Everything an engine may not be named in: the IR, the portable terms, compat, the launcher.

    ``sim/`` belongs here too and was missing. It holds the capability registry that exists so
    shared code does not have to ask which engine it is on. The scan currently finds nothing
    there, which is the point of adding it before it does. ``sim/spawners/`` is excluded: it is
    Isaac Lab's own, loaded lazily, and not shared code. G1 native configurations live in
    ``assets/unitree_g1/isaacsim.py`` and ``assets/unitree_g1/mjlab.py`` with SDK objects
    loaded only by their builders, so those files are not in this scan.
    """
    import instinctlab

    root = pathlib.Path(instinctlab.__file__).parent
    paths = [p for package in ("spec", "mdp", "compat") for p in (root / package).rglob("*.py")]
    paths += [p for p in (root / "sim").glob("*.py")]
    play = root / "play"
    paths += [p for p in play.rglob("*.py")] if play.is_dir() else []
    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    return sorted([*paths, *_engine_machinery(), scripts / "train.py", scripts / "play.py"])


def _engine_names() -> frozenset[str]:
    import instinctlab.engines

    return frozenset(instinctlab.engines.ADAPTERS)


@pytest.mark.parametrize("source", _shared_layer(), ids=lambda p: p.name)
def test_the_shared_layer_never_branches_on_an_engine_name(source: pathlib.Path) -> None:
    """Engine differences may be data keyed by engine name; they may not be control flow.

    The distinction is what makes the third engine cost a row instead of an edit. A table gains a
    key and every reader of it keeps working, while ``if engine == "isaacsim" ... elif ...`` has an
    ``else`` that a new engine falls into -- and the failure is a shared-layer function silently
    taking the wrong limb rather than an adapter that was never written.

    Comparisons only. The engine names appear all over this layer as dictionary keys and in prose,
    which is the form they are supposed to take.
    """
    names = _engine_names()
    branches = [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Compare)
        and any(isinstance(n, ast.Constant) and n.value in names for n in (node.left, *node.comparators))
    ]
    assert not branches, (
        f"{source.name} branches on an engine name: {branches}. Engine-specific behaviour belongs in "
        "a table keyed by engine name, or in the adapter."
    )
