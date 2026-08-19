"""Guard: the sim2sim scene may not ask for a capability no backend has to provide.

Most of this file went with the unified stack. Two tests drove a rollout through the unified
environment and recorded which contract calls it made, so that a task exercising an undeclared
capability would be caught before it reached a real engine, and a third checked that restitution
randomisation is skipped rather than attempted on a backend that lacks it. Both needed the
environment as the thing doing the exercising.

The coverage moved rather than disappeared: the compiler stack derives each engine's capabilities
from its own term builders and decides per term what to skip, which
``tests/test_engines_compile.py`` checks directly -- including that an optional term the engine
lacks is skipped with a stated reason, that a required one stops the run, and that ``--strict``
promotes the first into the second.

What is left here is the fixture the behavioural checks in ``tests/simulators/`` bring up, and the
statement that everything it declares is something a backend can be asked for.
"""

from __future__ import annotations

from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.verify.scene import locomotion_flat_scene


def test_the_verification_scene_asks_only_for_capabilities_a_backend_can_serve() -> None:
    scene = locomotion_flat_scene(num_envs=4)
    # The mock backend advertises everything, so this is a statement about the declaration being
    # well-formed rather than about the mock -- it also exercises RuntimeRequirements construction.
    MockSimulatorBackend().capabilities.require(
        scene.requirements.capabilities | scene.requirements.optional_capabilities,
        context="sim2sim verification scene",
    )
