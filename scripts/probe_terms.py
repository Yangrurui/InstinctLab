"""Evaluate the portable terms on one engine, from a state written in by hand.

Half of a two-process comparison: run it once per engine and diff the results. Two processes because
Isaac Sim has to be launched before ``isaaclab`` can be imported, and importing both engines into
one interpreter is not something either expects.

The state is written rather than simulated. A term is a function of state, so agreement on a state
both engines were *put* into is a statement about the terms; agreement after stepping would also be
a statement about the integrators, which is a different and much weaker thing to test.

    python scripts/probe_terms.py --engine mjlab --out /tmp/mjlab.json
    python scripts/probe_terms.py --engine isaacsim --out /tmp/isaacsim.json
"""

from __future__ import annotations

import argparse
import json
import sys
import torch
from pathlib import Path

NUM_ENVS = 4
SEED = 12345


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("isaacsim", "mjlab"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_known_args()[0]


def _state(num_joints: int, device: str) -> dict[str, torch.Tensor]:
    """One pose and one velocity per environment, identical wherever this runs.

    Deliberately away from the default pose and not upright: a term that mixes up a frame or drops a
    rotation is indistinguishable from a correct one when the robot is level and still.
    """
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    randn = lambda *shape: torch.randn(*shape, generator=generator).to(device)
    quat = randn(NUM_ENVS, 4)
    return {
        "root_pos": torch.tensor([[0.0, 0.0, 0.78]], device=device).repeat(NUM_ENVS, 1) + 0.1 * randn(NUM_ENVS, 3),
        "root_quat": quat / quat.norm(dim=-1, keepdim=True),
        "root_lin_vel": 0.5 * randn(NUM_ENVS, 3),
        "root_ang_vel": 0.5 * randn(NUM_ENVS, 3),
        "joint_pos": 0.2 * randn(NUM_ENVS, num_joints),
        "joint_vel": 0.5 * randn(NUM_ENVS, num_joints),
        "action": 0.3 * randn(NUM_ENVS, num_joints),
        "command": torch.tensor([[0.6, -0.2, 0.4]], device=device).repeat(NUM_ENVS, 1),
    }


def _canonical_joint_order(env, catalog: list[str]) -> torch.Tensor:
    """Indices that put this engine's raw joint buffers into the catalog's depth-first order.

    Needed only for writing state. The engines store joint data in whatever order their own model
    description produced -- PhysX walks the tree breadth-first, MuJoCo keeps the file's order, and
    neither is decision D1's order -- so putting both robots into the same sampled state means
    permuting it into each engine's order on the way in.

    Reading needs no such step: the task selects the joint axis by name in the canonical order for
    both the action term and the joint observations, so the terms already return it. That is the
    point of D1, and it is why this comparison can diff the readings directly. A term that came back
    in the engine's order would now show up as a difference, which is the correct outcome rather
    than something to reindex away.
    """
    names = list(env.scene["robot"].joint_names)
    missing = set(catalog) - set(names)
    if missing:
        raise RuntimeError(f"The engine's model is missing joints the catalog names: {sorted(missing)}")
    return torch.tensor([names.index(name) for name in catalog], device=env.device)


def _write_state(env, state: dict[str, torch.Tensor], order: torch.Tensor) -> None:
    """Put the robot into the sampled state, translating the joint axis back to engine order."""
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    engine_order = torch.argsort(order)

    # The link-qualified writers, on both engines, and this is not a stylistic preference. The
    # unqualified ``write_root_state_to_sim`` means the link's velocity on mjlab and the centre of
    # mass's on Isaac Lab, so writing the same thirteen numbers to both leaves the two robots
    # genuinely differently: measured here, the resulting link velocities differ by up to 0.85 m/s.
    # Every velocity-reading term then disagrees, and the disagreement says nothing about the terms.
    pose = torch.cat([state["root_pos"] + env.scene.env_origins, state["root_quat"]], dim=-1)
    velocity = torch.cat([state["root_lin_vel"], state["root_ang_vel"]], dim=-1)
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        state["joint_pos"][:, engine_order], state["joint_vel"][:, engine_order], env_ids=env_ids
    )
    env.scene.write_data_to_sim()
    env.sim.forward() if hasattr(env.sim, "forward") else None


def _force_action_and_command(env, state: dict[str, torch.Tensor], order: torch.Tensor) -> None:
    """Overwrite the last action and the sampled command, which are buffers rather than physics.

    The action buffer is indexed by the term's own selection, which names the joints in the
    canonical order, so the sampled action goes in unpermuted.
    """
    del order
    term = next(iter(env.action_manager._terms.values()))
    for attr in ("_raw_actions", "_raw_action"):
        if hasattr(term, attr):
            getattr(term, attr)[:] = state["action"]
    command_term = env.command_manager._terms["base_velocity"]
    for attr in ("vel_command_b", "_vel_command_b", "command"):
        buffer = getattr(command_term, attr, None)
        if isinstance(buffer, torch.Tensor) and buffer.shape[-1] >= 3:
            buffer[:, :3] = state["command"]
            break


def main() -> int:
    args = _parse()
    if args.engine == "isaacsim":
        from isaaclab.app import AppLauncher

        AppLauncher({"headless": True, "enable_cameras": False})

    from instinctlab.tasks.locomotion.flat_g1 import flat_g1

    spec = flat_g1()
    if args.engine == "isaacsim":
        from instinctlab.engines.isaacsim import IsaacSimAdapter as Adapter
    else:
        from instinctlab.engines.mjlab import MjlabAdapter as Adapter

    compiled = Adapter().compile(spec, num_envs=NUM_ENVS, device=args.device)
    env = (
        compiled.env_cls(cfg=compiled.env_cfg)
        if args.engine == "isaacsim"
        else compiled.env_cls(cfg=compiled.env_cfg, device=args.device)
    )
    env.reset()

    catalog = list(spec.robot.joint_names)
    order = _canonical_joint_order(env, catalog)
    state = _state(len(catalog), str(env.device))
    _write_state(env, state, order)
    _force_action_and_command(env, state, order)

    results = _evaluate(env, spec, order)
    results.update(_readback(env, state))
    results["_meta"] = {
        "engine": args.engine,
        "engine_joint_order": list(env.scene["robot"].joint_names),
        "catalog_joint_order": catalog,
    }
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {len(results) - 1} term readings to {args.out}")

    import os

    # os._exit skips stdio buffers, and this script is normally run with its output redirected, so
    # without the flush the line above is silently dropped -- the same way check_parity.py once
    # reported success while losing everything it had printed.
    sys.stdout.flush()
    os._exit(0)


def _readback(env, state: dict[str, torch.Tensor]) -> dict[str, list]:
    """What the engine says the root state is, next to what was asked for.

    A term comparison is only meaningful if both robots are actually in the same state, and root
    velocity is where that is least obvious: the two engines differ over whether a written velocity
    is the link's or the centre of mass's, and over which frame an angular velocity is expressed in.
    Recording the request beside the reading separates a term that disagrees from a state that was
    never the same.
    """
    data = env.scene["robot"].data
    readings = {"state/requested_lin_vel": state["root_lin_vel"], "state/requested_ang_vel": state["root_ang_vel"]}
    for attr in (
        "root_lin_vel_w",
        "root_ang_vel_w",
        "root_link_lin_vel_w",
        "root_link_ang_vel_w",
        "root_com_lin_vel_w",
        "root_com_ang_vel_w",
        "root_quat_w",
        "root_link_quat_w",
    ):
        value = getattr(data, attr, None)
        if isinstance(value, torch.Tensor):
            readings[f"state/{attr}"] = value
    return {name: value.detach().float().cpu().tolist() for name, value in readings.items()}


def _evaluate(env, spec, order: torch.Tensor) -> dict[str, list]:
    """Every portable term the task declares, evaluated on the written state.

    Terms are called through the compiled configs so that the parameters compared are the ones a
    run would use, not a second set written for the test. Readings are recorded as the terms return
    them: the joint axis is selected by name in the canonical order, so there is nothing to undo.
    """
    del order
    readings: dict[str, list] = {}

    def record(label: str, value: torch.Tensor, joint_axis: bool) -> None:
        del joint_axis
        readings[label] = value.detach().float().cpu().tolist()

    for group_name, group in (
        env.observation_manager._group_obs_term_cfgs.items()
        if hasattr(env.observation_manager, "_group_obs_term_cfgs")
        else []
    ):
        names = env.observation_manager._group_obs_term_names[group_name]
        for name, cfg in zip(names, group, strict=True):
            record(f"obs/{group_name}/{name}", cfg.func(env, **cfg.params), joint_axis=True)

    for name, cfg in zip(env.reward_manager._term_names, env.reward_manager._term_cfgs, strict=True):
        record(f"reward/{name}", cfg.func(env, **cfg.params), joint_axis=False)

    for name, cfg in zip(env.termination_manager._term_names, env.termination_manager._term_cfgs, strict=True):
        record(f"done/{name}", cfg.func(env, **cfg.params).float(), joint_axis=False)

    return readings


if __name__ == "__main__":
    sys.exit(main())
