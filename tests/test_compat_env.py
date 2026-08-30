"""Guard: the environment surface really has converged, and the parts that did not are handled.

``compat.env`` is small because it claims the two environment classes already agree on nearly
everything a term touches. That claim is the whole justification for terms reading ``env.*``
directly, so it is checked here against both installed engines rather than trusted -- Isaac Lab
with ``ast``, since importing its environment needs ``omni``, and mjlab live where possible.

If an engine upgrade moves ``step_dt`` or renames a manager, the first assertion below fails and
says so, instead of a hundred ported terms failing later for reasons that look unrelated.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from instinctlab.compat.errors import PortabilityError
from instinctlab.compat.env import (
    ENV_TYPE_NAMES,
    PHYSICS_DT_CFG_PATH,
    command_names,
    env_engine,
    get_command,
    has_command,
)

_CONVERGED = (
    "cfg",
    "scene",
    "num_envs",
    "device",
    "episode_length_buf",
    "common_step_counter",
    "extras",
    "physics_dt",
    "step_dt",
    "max_episode_length",
    "max_episode_length_s",
    "action_manager",
    "command_manager",
    "curriculum_manager",
    "event_manager",
    "observation_manager",
    "reward_manager",
    "termination_manager",
)


def _members(path: pathlib.Path, class_name: str) -> set[str]:
    """Public members of a class: methods, properties, annotations and ``self.x =`` bindings."""
    if not path.is_file():  # pragma: no cover - upstream layout change
        pytest.skip(f"{class_name} not found at {path}")
    class_def = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    found: set[str] = set()
    for node in ast.walk(class_def):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            found.add(node.attr)
    return {name for name in found if not name.startswith("_")}


def _isaac_env_members() -> set[str]:
    isaaclab = pytest.importorskip("isaaclab")
    envs = pathlib.Path(isaaclab.__file__).parent / "envs"
    return _members(envs / "manager_based_rl_env.py", "ManagerBasedRLEnv") | _members(
        envs / "manager_based_env.py", "ManagerBasedEnv"
    )


def _mjlab_env_path() -> pathlib.Path:
    mjlab = pytest.importorskip("mjlab")
    return pathlib.Path(mjlab.__file__).parent / "envs/manager_based_rl_env.py"


def _mjlab_env_members() -> set[str]:
    return _members(_mjlab_env_path(), ENV_TYPE_NAMES["mjlab"])


"""
The claims about the engines.
"""


def test_the_documented_surface_is_present_on_both_engines():
    """Every member a portable term may read is spelled the same on both engines.

    This is the assertion the rest of the design leans on. Terms read ``env.step_dt`` and
    ``env.scene`` directly rather than through an accessor, which is only safe while this holds.
    """
    isaac, mjlab = _isaac_env_members(), _mjlab_env_members()
    assert not [name for name in _CONVERGED if name not in isaac], "missing from Isaac Lab"
    assert not [name for name in _CONVERGED if name not in mjlab], "missing from mjlab"


def test_environment_class_names_differ_only_in_capitalisation():
    """The recorded native names are real, and are the near-identical pair the codemod expects."""
    isaac_name, mjlab_name = ENV_TYPE_NAMES["isaacsim"], ENV_TYPE_NAMES["mjlab"]
    assert isaac_name != mjlab_name
    assert isaac_name.lower() == mjlab_name.lower()
    assert _members(_mjlab_env_path(), mjlab_name)  # raises StopIteration if absent
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "envs/manager_based_rl_env.py"
    assert f"class {isaac_name}(" in source.read_text()


@pytest.mark.parametrize("engine", sorted(PHYSICS_DT_CFG_PATH))
def test_physics_dt_reads_the_recorded_config_path(engine: str):
    """``physics_dt`` is spelled the same but reads a different config path on each engine."""
    if engine == "isaacsim":
        isaaclab = pytest.importorskip("isaaclab")
        path = pathlib.Path(isaaclab.__file__).parent / "envs/manager_based_env.py"
        class_name = "ManagerBasedEnv"
    else:
        path = _mjlab_env_path()
        class_name = ENV_TYPE_NAMES["mjlab"]
    class_def = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    body = next(node for node in class_def.body if isinstance(node, ast.FunctionDef) and node.name == "physics_dt")
    expected = "self.cfg." + ".".join(PHYSICS_DT_CFG_PATH[engine])
    assert ast.unparse(body).count(expected) == 1


def test_mjlab_returns_none_for_every_command_when_the_task_declares_none():
    """The trap :func:`get_command` exists for: a lookup that fails without raising."""
    managers = pytest.importorskip("mjlab.managers")
    null = managers.NullCommandManager()
    assert null.active_terms == []
    assert null.get_command("base_velocity") is None


def test_isaac_raises_instead_of_returning_none():
    """Isaac Lab has no null manager, so the same lookup is a ``KeyError`` on a bare dict."""
    isaaclab = pytest.importorskip("isaaclab")
    source = (pathlib.Path(isaaclab.__file__).parent / "managers/command_manager.py").read_text()
    tree = ast.parse(source)
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert not [name for name in classes if "Null" in name]
    class_def = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "CommandManager"
    )
    body = next(node for node in class_def.body if isinstance(node, ast.FunctionDef) and node.name == "get_command")
    assert ast.unparse(body).strip().endswith("return self._terms[name].command")


"""
The accessors, against stubs shaped like the declarations above.
"""


class _Manager:
    def __init__(self, commands: dict[str, object] | None):
        self.active_terms = list(commands or {})
        self._commands = commands or {}

    def get_command(self, name: str):
        return self._commands.get(name)


class _Env:
    def __init__(self, commands: dict[str, object] | None = None):
        self.command_manager = _Manager(commands)


def _engine_env(module: str):
    """An environment class defined, as far as introspection can tell, inside ``module``."""
    return type("Env", (), {"__module__": module})()


def test_command_names_lists_what_the_environment_has():
    assert command_names(_Env({"base_velocity": 1, "pose": 2})) == ["base_velocity", "pose"]
    assert command_names(_Env(None)) == []


def test_has_command_answers_without_raising_on_a_command_free_environment():
    assert has_command(_Env({"base_velocity": 1}), "base_velocity")
    assert not has_command(_Env({"base_velocity": 1}), "pose")
    assert not has_command(_Env(None), "base_velocity")


def test_get_command_returns_the_tensor_when_present():
    sentinel = object()
    assert get_command(_Env({"base_velocity": sentinel}), "base_velocity") is sentinel


def test_get_command_raises_on_both_engines_shapes_of_absence():
    """The point of the accessor: ``None`` and ``KeyError`` both become one named failure."""
    with pytest.raises(PortabilityError, match="no commands"):
        get_command(_Env(None), "base_velocity")
    with pytest.raises(PortabilityError, match="Available: pose"):
        get_command(_Env({"pose": 1}), "base_velocity")


def test_get_command_rejects_a_manager_that_contradicts_itself():
    """Listed as active but returning ``None`` is an engine bug, and is reported as one."""
    env = _Env({"base_velocity": None})
    with pytest.raises(PortabilityError, match="inconsistent"):
        get_command(env, "base_velocity")


@pytest.mark.parametrize(
    ("module", "engine"),
    [("isaaclab.envs.manager_based_rl_env", "isaacsim"), ("mjlab.envs.manager_based_rl_env", "mjlab")],
)
def test_env_engine_identifies_both_engines(module: str, engine: str):
    assert env_engine(_engine_env(module)) == engine


def test_env_engine_sees_through_a_project_subclass():
    """A task's own subclass lives in ``instinctlab``; the engine comes from what it inherits."""
    base = type("Base", (), {"__module__": "mjlab.envs.manager_based_rl_env"})
    subclass = type("LocomotionEnv", (base,), {"__module__": "instinctlab.tasks.locomotion"})
    assert env_engine(subclass()) == "mjlab"


def test_env_engine_refuses_to_guess_at_an_unknown_engine():
    with pytest.raises(PortabilityError, match="engine is unknown"):
        env_engine(_engine_env("newton.envs"))
