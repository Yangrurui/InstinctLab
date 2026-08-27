"""G1 locomotion tasks, declared as :class:`~instinctlab.spec.TaskSpec`.

Main's copy of this file registered ``Instinct-Locomotion-Flat-G1-v0`` against an Isaac Lab
env config. Those Gym ids are gone: the tasks are compiled from :func:`flat_g1` and
:func:`rough_g1` by ``scripts/train.py``. This package must stay engine-free so a process
running either engine can import it -- which is also why it does not import ``agents``.
"""

from .flat_env_cfg import flat_g1
from .rough_env_cfg import rough_g1

__all__ = ["flat_g1", "rough_g1"]
