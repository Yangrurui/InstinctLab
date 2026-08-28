"""Public names for the Unitree G1 asset package.

The complete configurations are intentionally kept in ``isaacsim.py`` and
``mjlab.py``. This package file only preserves the engine-neutral import path
used by task declarations; it owns no configuration.
"""

from .isaacsim import (
    G1_29DOF_DEFAULT_JOINT_POS,
    G1_29DOF_DFS_BODY_NAMES,
    G1_29DOF_DFS_COLLISION_BODY_NAMES,
    G1_29DOF_DFS_FRAME_NAMES,
    G1_29DOF_DFS_JOINT_NAMES,
    G1_29DOF_ISAAC_BFS_JOINT_NAMES,
    G1_29DOF_LINKS,
    RESOURCE_ROOT,
    beyondmimic_action_scale,
    g1_symmetric_joint_augmentation,
    make_g1_29dof_parkour_robot_spec,
    make_g1_29dof_robot_spec,
    make_g1_29dof_shadowing_robot_spec,
)

__all__ = [
    "G1_29DOF_DEFAULT_JOINT_POS",
    "G1_29DOF_DFS_BODY_NAMES",
    "G1_29DOF_DFS_COLLISION_BODY_NAMES",
    "G1_29DOF_DFS_FRAME_NAMES",
    "G1_29DOF_DFS_JOINT_NAMES",
    "G1_29DOF_ISAAC_BFS_JOINT_NAMES",
    "G1_29DOF_LINKS",
    "RESOURCE_ROOT",
    "beyondmimic_action_scale",
    "g1_symmetric_joint_augmentation",
    "make_g1_29dof_parkour_robot_spec",
    "make_g1_29dof_robot_spec",
    "make_g1_29dof_shadowing_robot_spec",
]
