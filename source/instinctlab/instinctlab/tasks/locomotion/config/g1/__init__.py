"""G1 flat locomotion, declared as a :class:`~instinctlab.spec.TaskSpec`.

Main's copy of this file registered ``Instinct-Locomotion-Flat-G1-v0`` against an Isaac Lab
env config. That Gym id is gone: the task is compiled from :func:`flat_g1` by
``scripts/train.py``. This package must stay engine-free so a process running either
engine can import it -- which is also why it does not import ``agents``.
"""

from .flat_env_cfg import FEET_CONTACT, UPPER_BODY_CONTACT, flat_g1

__all__ = ["FEET_CONTACT", "UPPER_BODY_CONTACT", "flat_g1"]
