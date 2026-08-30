"""MJLab action terms whose portable semantics differ from upstream defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.envs.mdp import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class PreservingJointPositionAction(JointPositionAction):
    """Joint-position action that honors ``preserve_order`` for joint transmissions.

    Upstream MJLab exposes the flag on the config, but its joint target path calls
    ``find_joints_by_actuator_names`` without forwarding it. The resulting action follows the
    entity's natural joint order even when a portable task explicitly declared another order.
    """

    def _find_targets(self, cfg: PreservingJointPositionActionCfg) -> tuple[list[int], list[str]]:
        target_ids, target_names = super()._find_targets(cfg)
        if not cfg.preserve_order:
            return target_ids, target_names

        _, ordered_names = self._entity.find_joints(
            cfg.actuator_names,
            joint_subset=target_names,
            preserve_order=True,
        )
        entity_index = {name: index for index, name in enumerate(self._entity.joint_names)}
        return [entity_index[name] for name in ordered_names], ordered_names


@dataclass(kw_only=True)
class PreservingJointPositionActionCfg(JointPositionActionCfg):
    """Build :class:`PreservingJointPositionAction` instead of MJLab's order-dropping term."""

    def build(self, env: ManagerBasedRlEnv) -> PreservingJointPositionAction:
        return PreservingJointPositionAction(self, env)


__all__ = ["PreservingJointPositionAction", "PreservingJointPositionActionCfg"]
