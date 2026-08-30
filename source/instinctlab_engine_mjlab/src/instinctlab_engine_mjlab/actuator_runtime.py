"""Observable runtime adapter for MJLab's native built-in PD actuator."""

from __future__ import annotations

from typing import Any


def _base_actuator(actuator: Any) -> Any:
    while hasattr(actuator, "base_actuator"):
        actuator = actuator.base_actuator
    return actuator


class MjlabBuiltinPdRuntimeAdapter:
    def matches(self, actuator: object) -> bool:
        from mjlab.actuator import BuiltinPdActuator

        return isinstance(_base_actuator(actuator), BuiltinPdActuator)

    def stiffness_groups(self, actuator: Any):
        base = _base_actuator(actuator)
        return ((actuator.target_ids, base.cfg.stiffness),)

    def effort_limit_for_joint(
        self,
        env: Any,
        asset: Any,
        actuator: Any,
        local_index: int,
    ):
        joint_id = int(actuator.target_ids[local_index])
        global_joint_id = int(asset.indexing.joint_ids[joint_id])
        ranges = env.sim.model.jnt_actfrcrange
        if ranges.ndim == 3:
            return ranges[:, global_joint_id].abs().max(dim=-1).values
        return ranges[global_joint_id].abs().max()


BUILTIN_PD_RUNTIME = MjlabBuiltinPdRuntimeAdapter()

__all__ = ["BUILTIN_PD_RUNTIME", "MjlabBuiltinPdRuntimeAdapter"]
