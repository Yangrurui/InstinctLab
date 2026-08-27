from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import JointPositionAction

from instinctlab.utils.name_order import resolve_name_indices

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ActionTerm, ActionTermCfg

    from . import action_cfg


class ActionOverridenMixin:
    """Override some action dimensions with the provided values constantly."""

    def __init__(self: ActionTerm, cfg: ActionTermCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)  # type: ignore
        asset = self._env.scene[cfg.asset_cfg.name]
        _, override_joint_names = asset.find_joints(
            cfg.asset_cfg.joint_names,
            preserve_order=cfg.asset_cfg.preserve_order,
        )
        # ``find_joints`` returns articulation-local IDs (BFS on Isaac), but ``action`` is indexed
        # by this term's ordered target names (canonical DFS in unified tasks). Resolve names onto
        # the action axis instead of using native IDs as action columns.
        self._override_action_ids = resolve_name_indices(self._joint_names, override_joint_names)
        self._override_value = cfg.override_value

    def process_actions(self: ActionTerm, action: torch.Tensor):
        _raw_actions = action
        action = _raw_actions.clone()
        action[:, self._override_action_ids] = self._override_value
        super().process_actions(action)
        self._raw_actions[:] = _raw_actions


class ActionOverridenJointPositionAction(ActionOverridenMixin, JointPositionAction):
    """Delayed joint position action term that overrides some action dimensions with the provided values constantly."""

    cfg: action_cfg.ActionOverridenJointPositionActionCfg
