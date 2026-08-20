"""Isaac PhysX contact-stack budget is the value main's parkour actually set.

``gpu_collision_stack_size`` used to sit in ``KNOWN_DRIFTS`` as an accepted
omission. The parser that filled main's side of that row was broken until
``618ef20``, so the row never pinned a real number. This file pins the
aligned value and Isaac Lab's default, without starting Kit.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests import reference_main_parkour as main_ref
from tests.test_parkour_main_reference import KNOWN_DRIFTS

REPO = Path(__file__).resolve().parent.parent
ADAPTER = REPO / "source/instinctlab/instinctlab/engines/isaacsim/adapter.py"


def _isaaclab_simulation_cfg() -> Path:
    import isaaclab

    return Path(isaaclab.__file__).resolve().parent / "sim" / "simulation_cfg.py"


def _assigned_in_compile(attr: str) -> ast.AST:
    tree = ast.parse(ADAPTER.read_text())
    compile_fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compile")
    for node in ast.walk(compile_fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if ast.unparse(target) == f"sim.physx.{attr}":
                return node.value
    raise AssertionError(f"compile() does not assign sim.physx.{attr}")


def test_isaac_lab_physx_default_collision_stack_is_2_26() -> None:
    """The default we would silently inherit if the adapter left the field unset."""
    source = _isaaclab_simulation_cfg().read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PhysxCfg")
    field = next(
        node
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "gpu_collision_stack_size"
    )
    assert field.value is not None
    assert main_ref._constant(field.value) == 2**26


def test_main_parkour_raises_collision_stack_to_2_29() -> None:
    assert main_ref.sim_params()["gpu_collision_stack_size"] == 2**29
    assert main_ref.sim_params()["gpu_max_rigid_patch_count"] == 10 * 2**15


def test_adapter_sets_both_main_physx_gpu_buffers_on_mesh_terrain() -> None:
    assert main_ref._constant(_assigned_in_compile("gpu_collision_stack_size")) == 2**29
    assert main_ref._constant(_assigned_in_compile("gpu_max_rigid_patch_count")) == 10 * 2**15


def test_collision_stack_raise_sits_in_the_mesh_terrain_branch() -> None:
    """A plane does not need main's 512 MiB stack; only generator/rough tiles do."""
    tree = ast.parse(ADAPTER.read_text())
    compile_fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compile")
    found = False
    for node in ast.walk(compile_fn):
        if not isinstance(node, ast.If):
            continue
        if "generator" not in ast.unparse(node.test) or "rough" not in ast.unparse(node.test):
            continue
        body = ast.unparse(node)
        assert "gpu_collision_stack_size" in body
        assert "gpu_max_rigid_patch_count" in body
        found = True
    assert found, "compile() lost the generator/rough branch that owns both GPU buffers"


def test_aligned_collision_stack_is_not_a_known_drift() -> None:
    assert "sim/physx/gpu_collision_stack_size" not in KNOWN_DRIFTS
