"""Isaac adapter must overlay RobotSpec onto the catalog ArticulationCfg.

The lookup table is the USD skeleton. Path, spawn z, merge_fixed_joints, and
torque delay live on the spec the task holds. Without this overlay a parkour
``RobotSpec.overridden`` would train the catalog popsicle on Isaac with no error.
"""

from __future__ import annotations

import sys
import types

from instinctlab.assets.unitree_g1.catalog import make_g1_29dof_robot_spec
from instinctlab.engines.isaacsim.assets import apply_robot_spec, delayed_actuators


class _Box:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        return _Box(**{**self.__dict__, **kwargs})


def _cfg(*, actuators=None) -> _Box:
    return _Box(
        spawn=_Box(
            asset_path="catalog.urdf",
            merge_fixed_joints=True,
            fix_base=True,
            replace_cylinders_with_capsules=False,
        ),
        init_state=_Box(pos=(1.0, 2.0, 3.0)),
        actuators=actuators if actuators is not None else {"legs": _Box(stiffness=1.0)},
    )


def test_apply_robot_spec_zero_delay_does_not_wrap_actuators() -> None:
    robot = make_g1_29dof_robot_spec()
    actuators = {"legs": _Box(stiffness=40.0)}
    out = apply_robot_spec(_cfg(actuators=actuators), robot)
    assert out.actuators is actuators
    assert type(out.actuators["legs"]).__name__ == "_Box"
    assert out.spawn.asset_path.endswith("g1_29dof_torsobase_popsicle.urdf")
    assert out.spawn.merge_fixed_joints is False
    assert out.spawn.fix_base is False
    assert out.spawn.replace_cylinders_with_capsules is True
    assert out.init_state.pos == (0.0, 0.0, 0.82)


def test_apply_robot_spec_reads_the_task_copy_not_the_lookup_table() -> None:
    robot = make_g1_29dof_robot_spec().overridden(
        default_root_pos=(0.0, 0.0, 0.9),
        asset_paths={"isaacsim": "/tmp/g1_29dof_torsoBase_popsicle_with_shoe.urdf"},
        import_options={"isaacsim": {"merge_fixed_joints": True}},
    )
    assert robot.actuator_delay == (0, 0)
    actuators = {"legs": _Box(stiffness=40.0)}
    out = apply_robot_spec(_cfg(actuators=actuators), robot)
    assert out.actuators is actuators
    assert out.spawn.asset_path.endswith("with_shoe.urdf")
    assert out.spawn.merge_fixed_joints is True
    assert out.init_state.pos[2] == 0.9


def test_delayed_actuators_copies_pd_numbers_and_hub_bounds(monkeypatch) -> None:
    created: list[object] = []

    class FakeDelayed:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created.append(self)

    actuators_mod = types.ModuleType("isaaclab.actuators")
    actuators_mod.DelayedPDActuatorCfg = FakeDelayed
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", actuators_mod)
    if "isaaclab" not in sys.modules:
        monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))

    src = {
        "legs": _Box(
            joint_names_expr=[".*_hip_.*"],
            stiffness=40.0,
            damping=1.0,
            effort_limit_sim=88.0,
            extra_field="must_not_copy",
        )
    }
    out = delayed_actuators(src, (0, 2))
    assert len(created) == 1
    wrapped = out["legs"]
    assert wrapped.min_delay == 0
    assert wrapped.max_delay == 2
    assert wrapped.stiffness == 40.0
    assert wrapped.damping == 1.0
    assert wrapped.effort_limit_sim == 88.0
    assert not hasattr(wrapped, "extra_field")


def test_delayed_actuators_retargets_an_existing_delayed_cfg() -> None:
    class DelayedPDActuatorCfg:
        def __init__(self, min_delay: int, max_delay: int, stiffness: float):
            self.min_delay = min_delay
            self.max_delay = max_delay
            self.stiffness = stiffness

        def replace(self, **kwargs):
            return DelayedPDActuatorCfg(
                kwargs.get("min_delay", self.min_delay),
                kwargs.get("max_delay", self.max_delay),
                kwargs.get("stiffness", self.stiffness),
            )

    src = {"legs": DelayedPDActuatorCfg(0, 1, 40.0)}
    out = delayed_actuators(src, (0, 2))
    assert type(out["legs"]).__name__ == "DelayedPDActuatorCfg"
    assert out["legs"].min_delay == 0
    assert out["legs"].max_delay == 2
    assert out["legs"].stiffness == 40.0
