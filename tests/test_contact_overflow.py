"""Overflow refusal is a guard, not a JSON dump.

The unit tests use a fake ``wp_data.overflow`` so they run in the default
suite. A live test that writes the real device bits lives in
``test_contact_overflow_mjlab_live.py``.
"""

from __future__ import annotations

import ast
import numpy as np
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from instinctlab.utils.contact_overflow import (
    ALLOW_ENV,
    ContactOverflowError,
    attach_overflow_guard,
    check_contact_overflow,
    contact_budget_snapshot,
    decode_overflow,
    format_overflow_message,
    overflow_allowed,
)
from tests.isaacsim_app import (
    file_has_isaacsim_mark,
    invocation_is_single_file,
    isaacsim_module_paths,
    selects_isaacsim_session,
)

REPO = Path(__file__).resolve().parent.parent
ENTRY = REPO / "scripts" / "train.py"


class _Arr:
    def __init__(self, data):
        self._data = np.asarray(data)

    def numpy(self):
        return self._data


def _fake_env(*, overflow, nacon, nefc, nconmax=256, njmax=768, nworld=None):
    overflow = np.asarray(overflow, dtype=np.int64)
    nworld = overflow.size if nworld is None else nworld
    nefc = np.asarray(nefc, dtype=np.int64)
    env = SimpleNamespace(
        sim=SimpleNamespace(
            wp_data=SimpleNamespace(
                overflow=_Arr(overflow),
                nacon=_Arr([nacon]),
                nefc=_Arr(nefc),
            ),
            cfg=SimpleNamespace(nconmax=nconmax, njmax=njmax, contact_sensor_maxmatch=128),
        ),
        num_envs=nworld,
        device="cpu",
    )
    env.unwrapped = env
    return env


def test_decode_names_the_narrowphase_and_nefc_bits() -> None:
    pytest.importorskip("mujoco_warp")
    flags = decode_overflow((1 << 0) | (1 << 3))
    assert "NEFC" in flags
    assert "NARROWPHASE" in flags


def test_isaac_like_env_is_a_no_op() -> None:
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=SimpleNamespace()))
    assert contact_budget_snapshot(env) is None
    assert check_contact_overflow(env, phase="construction") is None
    assert attach_overflow_guard(env) is env


def _fake_isaac_env(
    *, needed_stack: int, allocated_stack: int, needed_patches: int = 10, allocated_patches: int = 327680
):
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            sim=SimpleNamespace(
                device="cuda:1",
                physics_prim_path="/physicsScene",
                physx=SimpleNamespace(
                    gpu_collision_stack_size=allocated_stack,
                    gpu_max_rigid_patch_count=allocated_patches,
                ),
            )
        ),
        sim=SimpleNamespace(),
        device="cuda:1",
        _physx_scene_stats=SimpleNamespace(
            gpu_mem_collision_stack_size=needed_stack,
            gpu_mem_rigid_patch_count=needed_patches,
            gpu_mem_rigid_contact_count=needed_patches * 3,
        ),
    )
    env.unwrapped = env
    return env


def test_isaac_construction_refuses_when_needed_stack_exceeds_budget(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_isaac_env(needed_stack=813424, allocated_stack=2**14)
    with pytest.raises(ContactOverflowError, match="PhysX construction overflow") as caught:
        check_contact_overflow(env, phase="construction")
    text = str(caught.value)
    assert "collision stack" in text
    assert "813424" in text
    assert "16384" in text
    assert "drops contacts" in text
    assert ALLOW_ENV in text
    assert caught.value.snapshot["any_overflow"] is True
    assert caught.value.snapshot["engine"] == "isaacsim"


def test_isaac_clean_budget_does_not_raise(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_isaac_env(needed_stack=912488, allocated_stack=2**29)
    assert check_contact_overflow(env, phase="construction") is None
    snapshot = contact_budget_snapshot(env)
    assert snapshot is not None
    assert snapshot["any_overflow"] is False
    assert snapshot["gpu_mem_collision_stack_size"] == 912488


def test_isaac_guard_wrapper_raises_after_step(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_isaac_env(needed_stack=100, allocated_stack=2**29)

    def _step(actions):
        env._physx_scene_stats.gpu_mem_collision_stack_size = 2**29 + 1
        return "obs", "rew", "done", {}

    env.step = _step
    env.episode_length_buf = None
    wrapped = attach_overflow_guard(env)
    assert wrapped is not env
    with pytest.raises(ContactOverflowError, match="step overflow"):
        wrapped.step(None)


def test_isaac_adapter_wrap_for_rl_checks_overflow() -> None:
    source = (REPO / "source/instinctlab/instinctlab/engines/isaacsim/adapter.py").read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IsaacSimAdapter")
    wrap = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "wrap_for_rl")
    body = ast.unparse(wrap)
    assert "check_contact_overflow" in body
    assert "construction" in body
    assert "attach_overflow_guard" in body
    assert "InstinctRlVecEnvWrapper" in body


def test_construction_refuses_when_a_world_has_overflow(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[0, 1 << 3, 0], nacon=4200, nefc=[100, 200, 80], nconmax=256, njmax=768)
    with pytest.raises(ContactOverflowError, match="construction overflow") as caught:
        check_contact_overflow(env, phase="construction")
    assert not np.any(env.sim.wp_data.overflow.numpy())
    assert "NARROWPHASE" in str(caught.value)
    assert "1 of 3 worlds" in str(caught.value)
    assert "nconmax=256" in str(caught.value)
    assert "nacon=4200" in str(caught.value)
    assert "njmax=768" in str(caught.value)
    assert "nefc_max=200" in str(caught.value)
    assert ALLOW_ENV in str(caught.value)


def test_step_refuses_when_overflow_appears_after_a_step(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[1 << 0], nacon=100, nefc=[800], nconmax=256, njmax=768)
    with pytest.raises(ContactOverflowError, match="step overflow") as caught:
        check_contact_overflow(env, phase="step")
    assert "NEFC" in str(caught.value)
    assert "nefc_max=800" in str(caught.value)


def test_hfield_message_names_the_compile_time_cap(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[1 << 5], nacon=10, nefc=[10], nconmax=256, njmax=768)
    with pytest.raises(ContactOverflowError, match="mjMAXCONPAIR") as caught:
        check_contact_overflow(env, phase="step")
    assert "HFIELD" in str(caught.value)
    assert "raising nconmax does not clear it" in str(caught.value)


_EPA = 1 << 8


def test_a_rare_epa_horizon_is_reported_and_survived(monkeypatch, capsys) -> None:
    """The bit we cannot fix must not end the run at a rate nobody could act on.

    One event in 545 iterations of a 700-iteration job is what actually happened, and the
    old guard's only options were "die" or "--allow_contact_overflow", which would have
    switched off the nconmax / njmax checks along with it.
    """
    pytest.importorskip("mujoco_warp")
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[_EPA, 0, 0], nacon=10, nefc=[10, 10, 10])
    snapshot = check_contact_overflow(env, phase="step")
    assert snapshot is not None and snapshot["untunable_events"] == 1
    assert not np.any(env.sim.wp_data.overflow.numpy())
    note = capsys.readouterr().out
    assert "EPA_HORIZON" in note and "MJ_MAX_EPAHORIZON=24" in note
    assert "env steps with new overflow" in note
    assert "not a failure" in note
    # Sticky bit was consumed; repeated polls must not inflate the rate.
    for _ in range(5):
        assert check_contact_overflow(env, phase="step") is None
    assert capsys.readouterr().out == ""


def test_sticky_overflow_bit_is_not_recounted_without_rearm(monkeypatch) -> None:
    """Presence left on the device without a new step must not count again."""
    pytest.importorskip("mujoco_warp")
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[_EPA], nacon=10, nefc=[10])
    first = check_contact_overflow(env, phase="step")
    assert first is not None and first["untunable_events"] == 1
    assert not np.any(env.sim.wp_data.overflow.numpy())
    second = check_contact_overflow(env, phase="step")
    assert second is None
    assert getattr(env, "_instinctlab_untunable_overflow")["events"] == 1


def test_rearmed_epa_events_accumulate_to_fatal_threshold(monkeypatch) -> None:
    """Each step that sets EPA anew must still trip the rate ceiling."""
    pytest.importorskip("mujoco_warp")
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[0], nacon=10, nefc=[10])

    def _arm_epa() -> None:
        env.sim.wp_data.overflow._data[0] = _EPA

    _arm_epa()
    with pytest.raises(ContactOverflowError, match="no longer a numerical") as caught:
        for _ in range(300):
            _arm_epa()
            check_contact_overflow(env, phase="step")
    assert "MJ_MAX_EPAHORIZON=24" in str(caught.value)


def test_epa_horizon_becomes_fatal_once_it_stops_being_rare(monkeypatch) -> None:
    """A rate is a different claim from a one-off; past the ceiling it is real degradation."""
    test_rearmed_epa_events_accumulate_to_fatal_threshold(monkeypatch)


def test_epa_horizon_alongside_a_budget_bit_is_still_fatal(monkeypatch) -> None:
    """The concession is for the bit we cannot raise, and does not cover the ones we can."""
    pytest.importorskip("mujoco_warp")
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[_EPA | (1 << 0)], nacon=10, nefc=[900])
    with pytest.raises(ContactOverflowError, match="Raise sim.nconmax"):
        check_contact_overflow(env, phase="step")


def test_escape_hatch_warns_and_does_not_raise(monkeypatch, capsys) -> None:
    monkeypatch.setenv(ALLOW_ENV, "1")
    import instinctlab.utils.contact_overflow as mod

    mod._warned_allow = False
    assert overflow_allowed() is True
    env = _fake_env(overflow=[1 << 3], nacon=5000, nefc=[10], nconmax=256, njmax=768)
    snapshot = check_contact_overflow(env, phase="construction")
    assert snapshot is not None
    assert snapshot["any_overflow"] is True
    err = capsys.readouterr().out
    assert ALLOW_ENV in err
    assert "will not stop the run" in err


def test_clean_check_returns_none_without_raising(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[0, 0], nacon=100, nefc=[50, 40], nconmax=256, njmax=768)
    assert check_contact_overflow(env, phase="construction") is None
    snapshot = contact_budget_snapshot(env)
    assert snapshot is not None
    assert snapshot["any_overflow"] is False
    assert snapshot["nacon"] == 100
    assert snapshot["nefc_max"] == 50


def test_guard_wrapper_raises_after_step(monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    env = _fake_env(overflow=[0], nacon=10, nefc=[10])

    def _step(actions):
        env.sim.wp_data.overflow = _Arr(np.array([1 << 3], dtype=np.int64))
        return "obs", "rew", "done", {}

    env.step = _step
    env.episode_length_buf = None
    wrapped = attach_overflow_guard(env)
    assert wrapped is not env
    with pytest.raises(ContactOverflowError, match="step overflow"):
        wrapped.step(None)


def test_message_is_actionable_without_an_env() -> None:
    text = format_overflow_message(
        {
            "overflow_flags": ["NARROWPHASE"],
            "overflow_max": 8,
            "nworld": 16,
            "worlds_with_overflow": 4,
            "nconmax": 256,
            "njmax": 768,
            "nacon": 5000,
            "nefc_max": 200,
        },
        phase="step",
    )
    assert "4 of 16 worlds" in text
    assert "nconmax=256" in text
    assert "nacon=5000" in text
    assert "njmax=768" in text
    assert "--allow_contact_overflow" in text


def test_train_entry_gates_terrain_split_and_keeps_the_overflow_hatch() -> None:
    source = ENTRY.read_text()
    assert "snapshot_contact_budget" not in source
    assert "dump_contact_peaks" not in source
    assert "log_terrain_split" in source
    assert "allow_contact_overflow" in source
    tree = ast.parse(source)
    gated = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "args.log_terrain_split" not in ast.unparse(node.test):
            continue
        body = ast.unparse(node)
        assert "attach_terrain_split" in body
        gated = True
    assert gated, "attach_terrain_split must sit under if args.log_terrain_split"
    ungated = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and "attach_terrain_split" in ast.unparse(node)
        and not _assign_is_under_log_terrain_split(tree, node)
    ]
    assert not ungated, f"unconditional terrain-split attach: {ungated}"
    assert 'os.environ["INSTINCTLAB_ALLOW_CONTACT_OVERFLOW"] = "1"' in source


def _assign_is_under_log_terrain_split(tree: ast.AST, target: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or "args.log_terrain_split" not in ast.unparse(node.test):
            continue
        if any(child is target for child in ast.walk(node)):
            return True
    return False


def test_mjlab_env_checks_overflow_at_construction_and_after_step() -> None:
    """The live hook is two calls. A unit test that only sees a fake env cannot drop them."""
    source = (REPO / "source/instinctlab/instinctlab/engines/mjlab/env.py").read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TerrainAwareRlEnv")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    assert _calls_check_contact_overflow(init, phase="construction")
    assert _calls_check_contact_overflow(step, phase="step")
    assert "return result" in ast.unparse(step)


def _calls_check_contact_overflow(fn: ast.FunctionDef, *, phase: str) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name != "check_contact_overflow":
            continue
        for kw in node.keywords:
            if kw.arg == "phase" and ast.unparse(kw.value).strip("'\"") == phase:
                return True
    return False


def test_pytest_ini_puts_the_repo_root_on_pythonpath() -> None:
    text = (REPO / "pytest.ini").read_text()
    assert "pythonpath = ." in text


def test_ensure_isaac_app_is_idempotent_without_launching_kit() -> None:
    """The guard that stops a second AppLauncher must not itself start Kit."""
    from tests import isaacsim_app

    sentinel = object()
    previous = isaacsim_app._APP
    isaacsim_app._APP = sentinel
    try:
        assert isaacsim_app.ensure_isaac_app() is sentinel
        assert isaacsim_app.ensure_isaac_app(device="cuda:9") is sentinel
    finally:
        isaacsim_app._APP = previous


def test_isaacsim_live_files_do_not_construct_app_launcher_themselves() -> None:
    """A second AppLauncher in the same process hangs; live files must share one.

    ``pytest tests -m isaacsim`` collects every marked file into one process.
    Each file used to construct ``AppLauncher`` itself. The second call tears
    the first Kit down and never returns (CPU ~100%, GPU 0, SIGTERM ignored).
    """
    offenders: list[str] = []
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        if not file_has_isaacsim_mark(tree):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "AppLauncher")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "AppLauncher")
            ):
                offenders.append(str(path.relative_to(REPO)))
                break
    assert offenders == [], f"isaacsim live files must call ensure_isaac_app: {offenders}"


def test_isaacsim_mark_is_not_a_string_that_merely_says_isaacsim() -> None:
    """Parametrize ids and engine tables mention isaacsim; they are not live Kit tests."""
    assert file_has_isaacsim_mark(REPO / "tests/simulators/test_isaacsim_behavior.py")
    assert file_has_isaacsim_mark(REPO / "tests/test_isaac_contact_budget_live.py")
    assert file_has_isaacsim_mark(REPO / "tests/test_contact_air_time_threshold.py")
    assert not file_has_isaacsim_mark(REPO / "tests/test_compat_sensors.py")
    assert not file_has_isaacsim_mark(REPO / "tests/test_compat_env.py")
    assert not file_has_isaacsim_mark(REPO / "tests/test_contact_overflow_mjlab_live.py")


def test_isaacsim_parent_invocation_is_the_directory_not_one_file() -> None:
    """The combined suite must fork modules; a single-file run must not."""
    from types import SimpleNamespace

    assert invocation_is_single_file(SimpleNamespace(args=[str(REPO / "tests/test_isaac_contact_budget_live.py")]))
    assert not invocation_is_single_file(SimpleNamespace(args=[str(REPO / "tests")]))
    assert not invocation_is_single_file(SimpleNamespace(args=[]))


def test_isaacsim_module_paths_keep_collection_order_without_duplicates() -> None:
    from types import SimpleNamespace

    a = Path("tests/simulators/test_isaacsim_behavior.py")
    b = Path("tests/test_isaac_contact_budget_live.py")
    items = [SimpleNamespace(path=a), SimpleNamespace(path=a), SimpleNamespace(path=b)]
    assert isaacsim_module_paths(items) == [a, b]


def test_selects_isaacsim_session_only_when_the_suite_is_requested() -> None:
    assert selects_isaacsim_session("isaacsim")
    assert selects_isaacsim_session("isaacsim and not foo")
    assert not selects_isaacsim_session("not mjlab and not isaacsim")
    assert not selects_isaacsim_session("not isaacsim")
    assert not selects_isaacsim_session("")
    assert not selects_isaacsim_session("mjlab")


def test_isaacsim_collection_does_not_import_mjlab_warp() -> None:
    """mjlab's site-packages warp 1.16 has no types.array; Isaac's extensions need it.

    ``pytest tests -m isaacsim`` used to import every test module, including
    ``test_contact_overflow_mjlab_live.py``. That ``import warp`` won the process
    before Kit started. Collect-only is enough to catch the regression: it must
    not start Kit, and it must not load warp.
    """
    import subprocess
    import sys

    script = """
import sys
import pytest

class Probe:
    def pytest_collection_finish(self, session):
        print("WARP_IMPORTED", "warp" in sys.modules)
        print("MJLAB_IMPORTED", "mjlab" in sys.modules)
        print("ITEM_COUNT", len(session.items))
        print("ITEMS", " ".join(item.nodeid for item in session.items))

raise SystemExit(pytest.main(
    ["tests", "-m", "isaacsim", "--collect-only", "-q", "-p", "no:cacheprovider"],
    plugins=[Probe()],
))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARP_IMPORTED False" in result.stdout
    assert "MJLAB_IMPORTED False" in result.stdout
    assert "test_contact_overflow_mjlab_live" not in result.stdout
    assert "test_isaacsim_behavior.py" in result.stdout


def test_live_make_env_tests_carry_an_engine_marker() -> None:
    """A live test with no marker is collected by default and skipped by ``-m mjlab``.

    ``test_rough_g1_mjlab_live.py`` shipped that way. The default suite must not
    start a real engine; ``-m mjlab`` / ``-m isaacsim`` must still find the file.
    """
    tests_dir = REPO / "tests"
    unmarked: list[str] = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        text = path.read_text()
        if "make_env()" not in text:
            continue
        tree = ast.parse(text)
        module_marks = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                module_marks.add(ast.unparse(node.value))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            if not _function_calls_make_env(node):
                continue
            marks = " ".join(ast.unparse(d) for d in node.decorator_list)
            marked = (
                "mjlab" in marks or "isaacsim" in marks or any("mjlab" in m or "isaacsim" in m for m in module_marks)
            )
            if not marked:
                unmarked.append(f"{path.relative_to(REPO)}::{node.name}")
    assert unmarked == [], f"live make_env tests without an engine marker: {unmarked}"


def _function_calls_make_env(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "make_env":
            return True
        if isinstance(func, ast.Name) and func.id == "make_env":
            return True
    return False
