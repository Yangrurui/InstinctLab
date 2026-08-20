"""G1 parkour tasks, declared as :class:`~instinctlab.spec.TaskSpec`.

The legacy Isaac-only Gym ids ``Instinct-Parkour-Target-Amp-G1-v0`` and
``Instinct-Parkour-Target-Amp-G1-Play-v0`` live in :mod:`.legacy_gym` so that importing this
package does not pull in gymnasium or Isaac. ``register_legacy_isaac_tasks()`` still imports
that sibling. This package must stay engine-free so a process running either engine can
import it -- which is also why it does not import ``agents``.
"""

from .target_env_cfg import FEET_CONTACT, TORSO_CONTACT, UNDESIRED_CONTACT, parkour_target_g1

__all__ = ["FEET_CONTACT", "TORSO_CONTACT", "UNDESIRED_CONTACT", "parkour_target_g1"]
