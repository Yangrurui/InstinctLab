"""Guard: one training entry point, and it does not know which engine it is running.

``scripts/train.py`` is where the N+M structure either holds or quietly stops holding. The moment
it grows a branch on the engine name -- to pick a wrapper, to construct an environment, to decide
what a reward's shape is -- every future engine costs a change here as well as in its own
directory, and the cost of the third engine is what this design was bought to control.

So the checks below read the launcher rather than run it: no engine may be named in it outside the
registry lookup, and none of its imports may reach an engine, since the choice has to be made
before Isaac Sim's launcher runs. What it does at runtime is covered by actually training two
iterations on each engine, which needs both simulators and does not belong in a unit test.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

import instinctlab.engines as engines
import instinctlab.tasks.registry as registry

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
_ENTRY = _SCRIPTS / "train.py"
_PLAY = _SCRIPTS / "play.py"
_LAUNCHERS = (_ENTRY, _PLAY)
_ENGINE_ROOTS = frozenset({"isaaclab", "isaacsim", "mjlab", "omni", "pxr", "carb", "mujoco", "warp"})


@pytest.fixture(scope="module")
def entry_source() -> str:
    return _ENTRY.read_text()


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_the_entry_point_exists(launcher: pathlib.Path) -> None:
    assert "def main(" in launcher.read_text()


def test_play_dummy_agents_skip_the_checkpoint() -> None:
    """mjlab play accepts --agent zero/random; those must not require a model_*.pt."""
    tree = ast.parse(_PLAY.read_text())
    choices: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        dest = keywords.get("dest") or (node.args[0] if node.args else None)
        name = dest.value if isinstance(dest, ast.Constant) else None
        if name != "--agent":
            continue
        choice_node = keywords.get("choices")
        if isinstance(choice_node, (ast.Tuple, ast.List)):
            choices = {elt.value for elt in choice_node.elts if isinstance(elt, ast.Constant)}
    assert choices == {"trained", "zero", "random"}

    dummy_assign = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and "zero" in ast.dump(node.value) and "random" in ast.dump(node.value)
        ),
        None,
    )
    assert dummy_assign is not None, "play.py lost the dummy-agent gate"
    dummy_names = {t.id for t in dummy_assign.targets if isinstance(t, ast.Name)}
    assert dummy_names, "dummy-agent gate must bind a name"

    dummy_ifs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id in dummy_names
    ]
    assert dummy_ifs, "play.py lost the dummy-agent branch"

    def _called(nodes: list[ast.stmt], name: str) -> bool:
        return any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
            for stmt in nodes
            for call in ast.walk(stmt)
        )

    dummy_if = dummy_ifs[0]
    assert not _called(dummy_if.body, "_resolve_checkpoint")
    assert _called(dummy_if.orelse, "_resolve_checkpoint")


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_launchers_reject_unknown_arguments(launcher: pathlib.Path) -> None:
    result = subprocess.run(
        [sys.executable, str(launcher), "--engine", "mjlab", "--definitely_misspelled"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --definitely_misspelled" in result.stderr


def test_play_selects_the_highest_numeric_checkpoint(tmp_path: pathlib.Path) -> None:
    module_spec = importlib.util.spec_from_file_location("_instinctlab_play_entry", _PLAY)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    run = tmp_path / "run"
    run.mkdir()
    for name in ("model_9.pt", "model_100.pt", "model_1000.pt"):
        (run / name).touch()
    args = SimpleNamespace(checkpoint=None, logroot=str(tmp_path), engine="mjlab", load_run="run")
    assert module._resolve_checkpoint(args, "unused").name == "model_1000.pt"


def test_train_resume_selects_the_highest_numeric_checkpoint(tmp_path: pathlib.Path) -> None:
    module_spec = importlib.util.spec_from_file_location("_instinctlab_train_entry", _ENTRY)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    run = tmp_path / "20260101_000000"
    run.mkdir()
    for name in ("model_9.pt", "model_100.pt", "model_1000.pt"):
        (run / name).touch()
    args = SimpleNamespace(
        resume=True,
        load_run=None,
        checkpoint=None,
        logroot=str(tmp_path),
        engine="mjlab",
    )
    agent = SimpleNamespace(experiment_name="unused", resume=False, load_run=".*", load_checkpoint=r"model_.*.pt")
    assert module._resolve_resume_checkpoint(args, agent).name == "model_1000.pt"


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_the_entry_point_imports_no_engine(launcher: pathlib.Path) -> None:
    """Including inside functions: an engine imported anywhere here runs before ``bootstrap``."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(launcher.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    leaked = imported & _ENGINE_ROOTS
    assert not leaked, f"{launcher.name} imports {sorted(leaked)}; the engine is chosen at runtime"


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_no_engine_is_named_in_the_entry_point(launcher: pathlib.Path) -> None:
    """A string like ``"mjlab"`` here is a branch waiting to happen.

    The engine names appear once, in the registry the launcher reads, and the launcher's own text
    should not repeat them -- with the sole exception of the usage examples in its docstring, which
    are documentation rather than dispatch.
    """
    tree = ast.parse(launcher.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in [tree, *ast.walk(tree)]
        if isinstance(getattr(node, "body", None), list)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    }
    named = {name for name in engines.names() if any(name in text for text in literals)}
    assert not named, f"{launcher.name} names {sorted(named)}; dispatch belongs in engines/ADAPTERS"


def test_the_launcher_does_not_construct_the_environment_itself(entry_source: str) -> None:
    """Engines disagree about constructor signatures, so ``make_env`` answers that, not the caller."""
    assert "make_env()" in entry_source
    assert "env_cls(" not in entry_source


def test_every_registered_engine_resolves_to_an_adapter() -> None:
    """Importing an adapter must work without its engine; the SDK import is deferred to bootstrap."""
    for name in engines.names():
        adapter = engines.adapter(name)
        assert adapter.name == name, f"{name} maps to an adapter that calls itself {adapter.name!r}"


def test_every_adapter_satisfies_the_protocol() -> None:
    for name in engines.names():
        adapter = engines.adapter(name)
        assert isinstance(adapter, engines.EngineAdapter)
        for required in ("add_cli_args", "bootstrap", "compile", "wrap_for_rl", "capabilities", "play"):
            assert callable(getattr(adapter, required, None)), f"{name} is missing {required}"


def test_an_unknown_engine_says_what_is_known() -> None:
    with pytest.raises(KeyError, match="isaacsim"):
        engines.adapter("bullet")


def test_every_registered_task_agrees_about_its_own_id() -> None:
    for task_id in registry.ids():
        assert registry.spec(task_id).task_id == task_id


def test_an_unknown_task_says_what_is_known() -> None:
    with pytest.raises(KeyError, match="Instinct-Velocity-Flat-G1"):
        registry.spec("Nonexistent-Task")


def test_a_task_can_be_listed_without_being_imported() -> None:
    """Listing must not cost an import; that is the difference from the Gym registry."""
    assert registry.ids()
    assert all(isinstance(path, str) for path in registry.TASKS.values())


def test_the_registries_import_with_engines_blocked() -> None:
    """A launcher reads both before it knows which engine it will start."""

    class _Blocker:
        def find_module(self, name, path=None):  # noqa: D102 - legacy finder protocol, enough here
            return self if name.split(".")[0] in _ENGINE_ROOTS else None

        def load_module(self, name):  # pragma: no cover - only reached on regression
            raise ImportError(f"the launcher must not need {name}")

    blocker = _Blocker()
    watched = ("instinctlab.tasks", "instinctlab.engines")
    evicted = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in _ENGINE_ROOTS or name.startswith(watched)
    }
    for name in evicted:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        assert importlib.import_module("instinctlab.tasks.registry").ids()
        assert importlib.import_module("instinctlab.engines").names()
    finally:
        sys.meta_path.remove(blocker)
        # Restore the originals, or later tests compare freshly imported classes by identity
        # against the ones they already hold and find two of everything.
        sys.modules.update(evicted)


def test_the_agent_config_resolves_without_an_engine() -> None:
    """The D4 boundary: hyperparameters are engine-independent, so reading them must be too."""
    spec = registry.spec("Instinct-Velocity-Flat-G1")
    agent = spec.agent.resolve()()
    values = agent.to_dict()
    assert values["policy"]["actor_hidden_dims"] == [256, 128, 128]
    assert values["algorithm"]["class_name"] == "PPO"
    assert values["num_steps_per_env"] == 24


def test_the_environment_is_given_a_seed(entry_source: str) -> None:
    """Both reference scripts hand the agent's seed to the environment config; this one must too.

    Left alone, ``cfg.seed`` is ``None`` on both engines and neither seeds anything. A run with no
    seed does not fail or warn its way into a log -- it trains, and the randomised masses, frictions
    and pushes come from wherever the process's RNG happened to be, so the run cannot be repeated
    and two engines given the same declaration are not given the same episodes.

    The value is checked, not just the presence of the assignment. Reading only the target passes on
    ``compiled.env_cfg.seed = None``, which is the very state this is meant to rule out.
    """
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in ast.walk(ast.parse(entry_source))
        if isinstance(node, ast.Assign) and node.targets and ast.unparse(node.targets[0]).endswith(".seed")
    }
    assert "compiled.env_cfg.seed" in assignments, (
        "the entry point never seeds the environment; main does this with env_cfg.seed = "
        "agent_cfg.seed and InstinctMJ with cfg.env.seed = seed"
    )
    assert assignments["compiled.env_cfg.seed"] == "agent_cfg.seed", (
        f"the environment is seeded from {assignments['compiled.env_cfg.seed']}, not from the agent's seed; "
        "the two must agree or --seed changes the policy's initialisation without changing the episodes"
    )


TORCH_BACKEND_FLAGS = {
    "torch.backends.cuda.matmul.allow_tf32": True,
    "torch.backends.cudnn.allow_tf32": True,
    "torch.backends.cudnn.deterministic": False,
    "torch.backends.cudnn.benchmark": False,
}


def _assignments(path: pathlib.Path) -> dict[str, object]:
    return {
        ast.unparse(node.targets[0]): ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Assign)
        and node.targets
        and ast.unparse(node.targets[0]).startswith("torch.backends")
        and isinstance(node.value, ast.Constant)
    }


def _calls(path: pathlib.Path) -> set[str]:
    return {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_each_engine_reproduces_its_references_torch_settings() -> None:
    """Each engine's bootstrap must reproduce the torch stack its own reference trained on.

    The references spell the same intent differently: main assigns the flags in its training
    script, InstinctMJ calls ``configure_torch_backends()`` (default ``allow_tf32=True``). This
    test used to assert that InstinctMJ "leaves torch at its defaults" and that the mjlab adapter
    therefore sets nothing. That was false, and because the test passed, mjlab trained with matmul
    TF32 off against a reference that trains with it on -- and against our own Isaac side, which
    sets it. Every two-engine comparison carried that asymmetry.

    The flags stay in each adapter rather than the launcher: the launcher runs for both engines and
    would have to be wrong for one of them.
    """
    engines_dir = pathlib.Path(engines.__file__).parent
    isaac = _assignments(engines_dir / "isaacsim" / "adapter.py")
    mjlab_adapter = engines_dir / "mjlab" / "adapter.py"

    assert isaac == TORCH_BACKEND_FLAGS, f"the Isaac adapter sets {isaac}, main sets {TORCH_BACKEND_FLAGS}"
    assert not _assignments(mjlab_adapter), (
        "the mjlab adapter assigns torch flags directly; call InstinctMJ's own "
        "configure_torch_backends() so this engine follows whatever that helper means"
    )
    assert "configure_torch_backends" in _calls(mjlab_adapter), (
        "the mjlab adapter does not call configure_torch_backends(); InstinctMJ's training script "
        "does, so mjlab would train with matmul TF32 off while its reference has it on"
    )
    assert not _assignments(_ENTRY), "torch backend settings in the launcher apply to every engine"


def test_mjlab_bootstrap_actually_turns_tf32_matmul_on() -> None:
    """Reading the call is not the same as the flag flipping. torch's default here is off."""
    import argparse
    import torch

    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    before = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        MjlabAdapter.bootstrap(argparse.Namespace())
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
        ) = before


_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _legacy_isaac_entry_points() -> list[pathlib.Path]:
    """Scripts that look up a legacy Gym id, wherever they live."""
    candidates = [*(_ROOT / "scripts").rglob("*.py"), *(_ROOT / "source").rglob("*/play.py")]
    return [p for p in candidates if "gym.make(" in (text := p.read_text()) or "gym.registry" in text]


def test_every_entry_point_that_uses_a_gym_id_registers_them() -> None:
    """Registration is an explicit call, and nothing reminds a new entry point to make it.

    ``tasks/__init__.py`` has to stay engine-free so the cross-engine launcher can read the task
    table before an engine is chosen, so registering the legacy Isaac Gym ids stopped being an
    import side effect. Every script that had been relying on that side effect kept importing the
    package and looking correct -- ``import instinctlab.tasks  # noqa: F401`` is exactly what a
    working registration used to look like -- while resolving its own task id to NameNotFound.
    Finding the next one by running it is the expensive way.
    """
    # Resolving a Gym id is the condition, not importing the package. One script reached the ids
    # without naming ``instinctlab.tasks`` at all, riding on whatever a transitive import happened
    # to pull in, and a guard keyed on the import would have kept walking past it.
    delinquent = [
        str(path.relative_to(_ROOT))
        for path in _legacy_isaac_entry_points()
        if "register_legacy_isaac_tasks()" not in path.read_text()
    ]
    assert not delinquent, (
        f"{delinquent} resolve a Gym id but never call register_legacy_isaac_tasks(); importing "
        "instinctlab.tasks no longer registers anything"
    )
