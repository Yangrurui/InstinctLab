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

import instinctlab_engine as engines
import instinctlab.tasks.registry as registry
from tests.engine_packages import ISAACSIM_ENGINE, MJLAB_ENGINE

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


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_launchers_render_help(launcher: pathlib.Path) -> None:
    result = subprocess.run(
        [sys.executable, str(launcher), "--engine", "mjlab", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout


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


def test_train_releases_process_group_and_application_when_training_fails(monkeypatch) -> None:
    module_spec = importlib.util.spec_from_file_location("_instinctlab_train_cleanup", _ENTRY)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    closed = []
    destroyed = []
    app = SimpleNamespace(close=lambda: closed.append("app"))
    engine = SimpleNamespace(bootstrap=lambda args: app)
    args = SimpleNamespace(engine="mjlab", distributed=False, local_rank=None, device="cpu")

    def fail_training(*unused) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_parse", lambda: args)
    monkeypatch.setattr(module, "_train", fail_training)
    monkeypatch.setattr(engines, "adapter", lambda name: engine)

    import instinctlab.training as training

    monkeypatch.setattr(training, "initialize_process_group", lambda run: None)
    monkeypatch.setattr(training, "destroy_process_group", lambda run: destroyed.append(True))

    with pytest.raises(RuntimeError, match="boom"):
        module.main()
    assert destroyed == [True]
    assert closed == ["app"]


def test_play_releases_application_when_playback_fails(monkeypatch) -> None:
    module_spec = importlib.util.spec_from_file_location("_instinctlab_play_cleanup", _PLAY)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    closed = []
    app = SimpleNamespace(close=lambda: closed.append("app"))
    engine = SimpleNamespace(bootstrap=lambda args: app)
    args = SimpleNamespace(
        engine="mjlab",
        viewer="native",
        export_only=False,
        export_onnx=False,
        agent="zero",
    )

    def fail_playback(*unused) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_parse", lambda: args)
    monkeypatch.setattr(module, "_play", fail_playback)
    monkeypatch.setattr(engines, "adapter", lambda name: engine)

    with pytest.raises(RuntimeError, match="boom"):
        module.main()
    assert closed == ["app"]


def test_play_rejects_invalid_export_before_bootstrap(monkeypatch) -> None:
    module_spec = importlib.util.spec_from_file_location("_instinctlab_play_validation", _PLAY)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    bootstrapped = []
    engine = SimpleNamespace(bootstrap=lambda args: bootstrapped.append(args))
    args = SimpleNamespace(
        engine="mjlab",
        viewer="native",
        export_only=True,
        export_onnx=False,
        agent="trained",
    )
    monkeypatch.setattr(module, "_parse", lambda: args)
    monkeypatch.setattr(engines, "adapter", lambda name: engine)

    with pytest.raises(ValueError, match="--export-only requires --export-onnx"):
        module.main()
    assert bootstrapped == []


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


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_production_launchers_require_a_clean_resolution_by_default(
    launcher: pathlib.Path,
) -> None:
    source = launcher.read_text()
    assert '"--strict"' not in source
    assert '"--allow-nonclean-resolution"' in source
    assert "strict=not args.allow_nonclean_resolution" in source
    assert "compiled.resolution.require_clean(" in source


def test_every_registered_engine_resolves_to_an_adapter() -> None:
    """Importing an adapter must work without its engine; the SDK import is deferred to bootstrap."""
    for name in engines.names():
        adapter = engines.adapter(name)
        assert adapter.name == name, f"{name} maps to an adapter that calls itself {adapter.name!r}"


def test_every_adapter_satisfies_the_protocol() -> None:
    for name in engines.names():
        adapter = engines.adapter(name)
        assert isinstance(adapter, engines.EngineAdapter)
        for required in ("add_cli_args", "bootstrap", "compile", "wrap_for_rl", "capabilities"):
            assert callable(getattr(adapter, required, None)), f"{name} is missing {required}"


def test_an_unknown_engine_says_what_is_known() -> None:
    with pytest.raises(KeyError, match="isaacsim"):
        engines.adapter("bullet")


def test_every_registered_task_agrees_about_its_own_id() -> None:
    from tests.task_specs import task_spec

    for task_id in registry.ids():
        assert task_spec(task_id).task_id == task_id


def test_an_unknown_task_says_what_is_known() -> None:
    with pytest.raises(KeyError, match="Instinct-Velocity-Flat-G1"):
        registry.factory("Nonexistent-Task")


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
    watched = ("instinctlab.tasks", "instinctlab_engine_isaacsim", "instinctlab_engine_mjlab")
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
        assert importlib.import_module("instinctlab_engine").names()
    finally:
        sys.meta_path.remove(blocker)
        # Restore the originals, or later tests compare freshly imported classes by identity
        # against the ones they already hold and find two of everything.
        sys.modules.update(evicted)


@pytest.mark.parametrize(
    "operation",
    (
        "import instinctlab_engine",
        "import instinctlab_engine; instinctlab_engine.names()",
        (
            "import argparse, instinctlab_engine; "
            "[instinctlab_engine.adapter(name).add_cli_args(argparse.ArgumentParser()) "
            "for name in instinctlab_engine.names()]"
        ),
    ),
    ids=("import", "discovery", "adapter-cli"),
)
def test_prebootstrap_engine_api_imports_neither_torch_nor_an_sdk(operation: str) -> None:
    """Exercise the real installed entry points in a genuinely fresh interpreter."""
    probe = f"""
import sys
{operation}
forbidden = ("torch", "isaaclab", "isaacsim", "mjlab", "mujoco", "warp", "omni", "pxr", "carb")
leaked = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)
if leaked:
    raise SystemExit("pre-bootstrap imports: " + ", ".join(leaked))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_agent_config_resolves_without_an_engine() -> None:
    """The D4 boundary: hyperparameters are engine-independent, so reading them must be too."""
    from tests.task_specs import task_spec

    spec = task_spec("Instinct-Velocity-Flat-G1")
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
    assert assignments["compiled.env_cfg.seed"] == "distributed.seed(agent_cfg.seed)", (
        f"the environment is seeded from {assignments['compiled.env_cfg.seed']}, not from the agent's seed; "
        "rank zero must use the agent seed and every other rank must use a stable, distinct offset"
    )


def test_train_uses_one_effective_agent_snapshot_for_contract_and_runner(
    entry_source: str,
) -> None:
    assert "agent_config = agent_cfg.to_dict()" in entry_source
    assert "agent_config=agent_config" in entry_source
    assert "OnPolicyRunner(env, agent_config," in entry_source
    assert "json.dump(agent_config, handle," in entry_source
    assert 'manifest["runtime_provenance"] = runtime_provenance(' in entry_source


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
    isaac = _assignments(ISAACSIM_ENGINE / "adapter.py")
    mjlab_adapter = MJLAB_ENGINE / "adapter.py"

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
    from instinctlab_engine_mjlab import MjlabAdapter

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
