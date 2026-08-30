"""Isaac PhysX contact-stack budget is the value main's parkour actually set.

``gpu_collision_stack_size`` used to sit in ``KNOWN_DRIFTS`` as an accepted
omission. The parser that filled main's side of that row was broken until
``618ef20``, so the row never pinned a real number. This file pins the
aligned value and Isaac Lab's default, without starting Kit.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from instinctlab_engine_isaacsim.adapter import _configure_sim_contact_budget

from tests import reference_main_parkour as main_ref
from tests.engine_packages import ISAACSIM_ENGINE
from tests.test_parkour_main_reference import KNOWN_DRIFTS

ADAPTER = ISAACSIM_ENGINE / "adapter.py"


def _isaaclab_simulation_cfg() -> Path:
    import isaaclab

    return Path(isaaclab.__file__).resolve().parent / "sim" / "simulation_cfg.py"


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
    sim = SimpleNamespace(physx=SimpleNamespace(), physics_material=None)
    material = object()
    _configure_sim_contact_budget(sim, {}, "rough", material)
    assert sim.physx.gpu_collision_stack_size == 2**29
    assert sim.physx.gpu_max_rigid_patch_count == 10 * 2**15
    assert sim.physics_material is material


def test_task_profile_overrides_generic_mesh_contact_budget() -> None:
    sim = SimpleNamespace(physx=SimpleNamespace(), physics_material=None)
    material = object()
    _configure_sim_contact_budget(
        sim,
        {
            "gpu_max_rigid_patch_count": 123,
            "gpu_max_rigid_contact_count": 456,
            "gpu_collision_stack_size": 789,
            "use_terrain_physics_material": True,
        },
        "shadow_motion_matched",
        material,
    )
    assert sim.physx.gpu_max_rigid_patch_count == 123
    assert sim.physx.gpu_max_rigid_contact_count == 456
    assert sim.physx.gpu_collision_stack_size == 789
    assert sim.physics_material is material


def test_aligned_collision_stack_is_not_a_known_drift() -> None:
    assert "sim/physx/gpu_collision_stack_size" not in KNOWN_DRIFTS


def test_adapter_wrap_for_rl_refuses_physx_overflow() -> None:
    tree = ast.parse(ADAPTER.read_text())
    cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "IsaacSimAdapter")
    wrap = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "wrap_for_rl")
    body = ast.unparse(wrap)
    assert "check_contact_overflow" in body
    assert 'phase="construction"' in body or "phase='construction'" in body
    assert "attach_overflow_guard" in body
