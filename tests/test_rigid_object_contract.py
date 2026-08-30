"""Rigid objects have explicit resource, reset, collision, and material semantics."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from instinctlab_engine.spec import RigidObjectRef


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


def _object(path: Path, *, kinematic: bool = True) -> RigidObjectRef:
    return RigidObjectRef(
        name="box",
        mesh=str(path),
        scale=(0.5, 0.75, 1.25),
        mass=2.5,
        kinematic=kinematic,
        collision_enabled=True,
        friction=0.8,
        initial_position=(1.0, 2.0, 3.0),
        initial_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        initial_linear_velocity=(0.1, 0.2, 0.3),
        initial_angular_velocity=(0.4, 0.5, 0.6),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scale", (1.0, 0.0, 1.0), "scale"),
        ("mass", 0.0, "mass"),
        ("friction", -0.1, "friction"),
        ("initial_quaternion_wxyz", (2.0, 0.0, 0.0, 0.0), "normalized"),
    ),
)
def test_rigid_object_rejects_invalid_physical_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    kwargs = {
        "name": "box",
        "mesh": str(tmp_path / "box.obj"),
        "scale": (1.0, 1.0, 1.0),
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        RigidObjectRef(**kwargs)


def test_resource_report_is_sdk_free_and_does_not_hide_a_missing_file(
    tmp_path: Path,
) -> None:
    code = f"""
import json
import sys
from instinctlab_engine.spec import RigidObjectRef
ref = RigidObjectRef(name='box', mesh={str(tmp_path / 'missing.obj')!r}, scale=(1.0, 1.0, 1.0))
print(json.dumps({{'report': ref.resource_report('isaacsim'), 'modules': sorted(sys.modules)}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["report"]["exists"] is False
    assert "torch" not in payload["modules"]
    assert not any(name.startswith(("isaaclab", "mjlab", "mujoco")) for name in payload["modules"])


@pytest.mark.parametrize(
    "module_name",
    (
        "instinctlab_engine_isaacsim.rigid_objects",
        "instinctlab_engine_mjlab.rigid_objects",
    ),
)
def test_missing_resource_fails_before_backend_import(
    tmp_path: Path, module_name: str
) -> None:
    code = f"""
import importlib
import json
import sys
from instinctlab_engine.spec import RigidObjectRef
module = importlib.import_module({module_name!r})
ref = RigidObjectRef(name='box', mesh={str(tmp_path / 'missing.obj')!r}, scale=(1.0, 1.0, 1.0))
try:
    module.rigid_object_cfg(ref{', prim_path="/World/box"' if 'isaacsim' in module_name else ''})
except FileNotFoundError as error:
    print(json.dumps({{'error': str(error), 'modules': sorted(sys.modules)}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert "does not exist" in payload["error"]
    assert "torch" not in payload["modules"]
    assert not any(name.startswith(("isaaclab", "mjlab", "mujoco")) for name in payload["modules"])


@pytest.mark.parametrize("kinematic", (True, False))
def test_mjlab_mesh_resource_compiles_with_declared_object_semantics(
    tmp_path: Path, kinematic: bool
) -> None:
    from instinctlab_engine_mjlab.rigid_objects import rigid_object_cfg

    ref = _object(_cube_obj(tmp_path / "box.obj"), kinematic=kinematic)
    cfg = rigid_object_cfg(ref)
    spec = cfg.spec_fn()
    model = spec.compile()
    body = spec.body("box")
    geom = spec.geom("box_geom")

    assert bool(body.mocap) is kinematic
    assert geom.mass == pytest.approx(2.5)
    assert tuple(geom.friction) == pytest.approx((0.8, 0.005, 0.0001))
    assert geom.contype == 1
    assert geom.conaffinity == 1
    assert tuple(cfg.init_state.pos) == (1.0, 2.0, 3.0)
    assert tuple(cfg.init_state.lin_vel) == (0.1, 0.2, 0.3)
    assert model.nbody == 2
    assert model.nmocap == (1 if kinematic else 0)
    assert model.nq == (0 if kinematic else 7)


def test_mjlab_collision_can_be_explicitly_disabled(tmp_path: Path) -> None:
    from instinctlab_engine_mjlab.rigid_objects import rigid_object_cfg

    ref = replace(
        _object(_cube_obj(tmp_path / "box.obj")),
        collision_enabled=False,
    )
    geom = rigid_object_cfg(ref).spec_fn().geom("box_geom")

    assert geom.contype == 0
    assert geom.conaffinity == 0


def test_hoi_scene_compiles_when_all_selected_object_resources_exist(
    tmp_path: Path,
) -> None:
    from instinctlab_engine_mjlab.adapter import MjlabAdapter

    from tests.task_specs import task_spec

    mesh = _cube_obj(tmp_path / "object.obj")
    task = task_spec("Instinct-Perceptive-HOI-Shadowing-G1-v0")
    objects = tuple(
        replace(obj, mesh=str(mesh), engine_meshes={})
        for obj in task.scene.rigid_objects
    )
    task = replace(task, scene=replace(task.scene, rigid_objects=objects))

    compiled = MjlabAdapter().compile(
        task,
        num_envs=2,
        device="cpu",
        strict=True,
    )

    assert set(compiled.env_cfg.scene.entities) == {
        "robot",
        "floorlamp",
        "largebox",
        "whitechair",
        "trashcan",
        "smalltable",
        "suitcase",
    }
    for name in task.scene.motion_references[0].scene_objects:
        compiled.env_cfg.scene.entities[name].spec_fn().compile()
