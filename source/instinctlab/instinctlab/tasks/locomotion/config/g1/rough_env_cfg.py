"""Rough-ground G1 velocity tracking, the same MDP as the flat task on reference rough ground.

The robot, observations, rewards, events and commands are the flat task's. What changes is the
ground -- ``kind="rough"``, which each adapter fills from its own parkour reference -- and a
``terrain_levels`` curriculum that walks robots up and down that grid. Height scanning is not
here: that is a sensor, not a terrain, and belongs with the raycaster work.

Nothing here imports an engine. The two reference grids already disagree (scale, extra tile,
mjlab's deterministic row difficulty vs Isaac's jitter); they now agree on ``num_cols=20``
and therefore on world width. The declaration names the intent and does not pick a winner.
"""

from __future__ import annotations

from instinctlab import mdp
from instinctlab.spec import (
    ContactSensorRef,
    CurriculumTermSpec,
    SceneSpec,
    TaskSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import (
    COMMAND,
    G1FlatEnvCfg,
)
from instinctlab.tasks.locomotion.terrains import rough_terrain


class G1RoughEnvCfg(G1FlatEnvCfg):
    def __init__(self) -> None:
        super().__init__()
        self.scene = SceneSpec(
            terrain=rough_terrain(),
            contact_sensors=(
                ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True, history_length=3),
            ),
            env_spacing=2.5,
        )
        self.curriculum = {
            "terrain_levels": CurriculumTermSpec(
                func=mdp.terrain_levels_vel,
                params={"command_name": COMMAND},
                level=Requirement.REQUIRED,
            )
        }


def rough_g1() -> TaskSpec:
    return G1RoughEnvCfg().to_task_spec("Instinct-Velocity-Rough-G1")
