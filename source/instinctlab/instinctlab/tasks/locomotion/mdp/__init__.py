"""Locomotion MDP terms for the Isaac Lab task, resolved lazily.

Names are looked up in this package first and in Isaac Lab's own ``mdp`` last, which is the
precedence the release's star imports had. Lookup is deferred rather than eager so that importing
this package does not pull in Isaac Sim.

The retired unified stack re-exported its own re-implementations over this list. They were faithful
in arithmetic and not in signature -- ``feet_air_time_positive_biped`` took a sensor name and body
names where this one takes a ``SceneEntityCfg`` -- so ``G1FlatEnvCfg`` named a function its own
params could not satisfy, and Isaac Lab's manager rejects that at construction. Nothing imports a
term by name, so the task simply stopped building. Do not re-export a second implementation of a
name that already exists here.
"""

_SOURCES = (
    "instinctlab.tasks.locomotion.mdp.rewards",
    "instinctlab.tasks.locomotion.mdp.curriculums",
    "isaaclab.envs.mdp",
)


def __getattr__(name: str):
    from importlib import import_module

    for module_name in _SOURCES:
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)
