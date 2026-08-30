"""Isaac converts, spawns, and binds a portable OBJ rigid object.

Run on demand:

    pytest -o addopts= -m isaacsim tests/test_isaac_rigid_object_live.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.live_device import resolve_live_device

pytestmark = pytest.mark.isaacsim


def _cube_obj(path: Path) -> Path:
    path.write_text(
        """v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""
    )
    return path


def test_isaac_obj_spawn_reset_collision_and_material(tmp_path: Path) -> None:
    device = resolve_live_device()
    ensure_isaac_app(device=device)

    import torch
    from pxr import Usd, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from instinctlab_engine.spec import RigidObjectRef
    from instinctlab_engine_isaacsim import IsaacSimAdapter

    from tests.task_specs import task_spec

    obj = RigidObjectRef(
        name="box",
        mesh=str(_cube_obj(tmp_path / "box.obj")),
        scale=(0.5, 0.75, 1.25),
        mass=2.5,
        kinematic=True,
        collision_enabled=True,
        friction=0.8,
        initial_position=(0.25, -0.5, 1.5),
    )
    task = task_spec("Instinct-Velocity-Flat-G1", "isaacsim")
    task = replace(
        task,
        scene=replace(task.scene, rigid_objects=(obj,)),
    )
    env = IsaacSimAdapter().compile(task, num_envs=2, device=device).make_env()
    try:
        env.reset()
        expected = env.scene.env_origins + torch.tensor(
            obj.initial_position, device=env.device
        )
        assert torch.allclose(env.scene["box"].data.root_pos_w, expected)

        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/box")
        collision_prims = [
            prim
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        assert collision_prims
        for prim in collision_prims:
            binding, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                "physics"
            )
            assert binding
            assert str(binding.GetPath()).endswith("/box/physicsMaterial")
        direct_binding = UsdShade.MaterialBindingAPI(root).GetDirectBinding("physics")
        assert str(direct_binding.GetMaterialPath()).endswith("/box/physicsMaterial")

        material = stage.GetPrimAtPath("/World/envs/env_0/box/physicsMaterial")
        physics = UsdPhysics.MaterialAPI(material)
        assert physics.GetStaticFrictionAttr().Get() == pytest.approx(0.8)
        assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.8)
    finally:
        env.close()
