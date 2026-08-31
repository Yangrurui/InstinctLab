"""Live conformance gate for the repository-external actuator wheel.

The module is shipped inside the fixture wheel so the test cannot accidentally
import its implementation from the InstinctLab checkout.  It starts the chosen
backend before importing torch, constructs a two-environment task, steps the
native joint-position action, and probes the SDK-owned gain and delay state.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path


def joint_position(env):
    """Small portable observation used by the live fixture task."""
    return env.scene["robot"].data.joint_pos


def _isolate_entry_points() -> None:
    """Ignore editable InstinctLab distributions outside this wheel venv."""
    environment_root = Path(sys.prefix).resolve()
    original_entry_points = metadata.entry_points

    def isolated_entry_points(*args, **kwargs):
        discovered = original_entry_points(*args, **kwargs)
        if isinstance(discovered, dict):
            discovered = [
                entry_point
                for group in discovered.groups
                for entry_point in discovered.select(group=group)
            ]
        return metadata.EntryPoints(
            entry_point
            for entry_point in discovered
            if Path(entry_point.dist.locate_file(""))
            .resolve()
            .is_relative_to(environment_root)
        )

    metadata.entry_points = isolated_entry_points


def _parse() -> tuple[argparse.Namespace, object]:
    _isolate_entry_points()
    from instinctlab_engine import adapter, names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", required=True, choices=names())
    chosen, _unknown = chooser.parse_known_args()
    selected = adapter(chosen.engine)
    parser = argparse.ArgumentParser(parents=[chooser])
    selected.add_cli_args(parser)
    return parser.parse_args(), selected


def _task(selected, engine_name: str):
    from instinctlab_engine.spec import (
        ActionTermSpec,
        AgentSpec,
        EntityRef,
        EventTermSpec,
        MdpSpec,
        ObsGroupSpec,
        ObsTermSpec,
        SceneSpec,
        SimSpec,
        TaskSpec,
    )

    robot = selected.robot_spec("fixture_bot/v1")
    joints = EntityRef(
        "robot",
        joints=robot.joint_names,
        preserve_order=True,
    )
    return TaskSpec(
        task_id="Fixture-External-Actuator-Live",
        robot=robot,
        scene=SceneSpec(),
        sim=SimSpec(
            physics_dt=0.01,
            decimation=1,
            episode_length_s=1.0,
        ),
        mdp=MdpSpec(
            observations={
                "policy": ObsGroupSpec(
                    terms={"joint_position": ObsTermSpec(func=joint_position)}
                )
            },
            actions={
                "joint_pos": ActionTermSpec(
                    kind="joint_position",
                    target=joints,
                    params={"scale": 1.0, "use_default_offset": False},
                )
            },
            events={
                "randomize_actuator_gains": EventTermSpec(
                    kind="randomize_actuator_gains",
                    mode="startup",
                    target=joints,
                    params={
                        "stiffness_range": (1.5, 1.5),
                        "damping_range": (2.0, 2.0),
                        "operation": "scale",
                    },
                )
            },
        ),
        agent=AgentSpec(runner="builtins:object"),
        engines=(engine_name,),
    )


def _actuator(asset, engine_name: str):
    actuators = asset.actuators
    values = actuators.values() if isinstance(actuators, dict) else actuators
    selected = tuple(values)
    if len(selected) != 1:
        raise AssertionError(f"expected one external actuator, got {len(selected)}")
    actuator = selected[0]
    if type(actuator).__module__ != (
        f"instinctlab_extension_fixture.{engine_name}_actuator"
    ):
        raise AssertionError(
            f"fixture did not build its external native class: {type(actuator)}"
        )
    model_id = getattr(
        actuator,
        "instinctlab_model_id",
        getattr(actuator.cfg, "instinctlab_model_id", None),
    )
    if model_id != "fixture.stateful.v1":
        raise AssertionError(f"native actuator lost model identity: {model_id!r}")
    return actuator


def _tensor_command(torch, values):
    position = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
    zeros = torch.zeros_like(position)
    return position, zeros


def _assert_isaac_native_state(env, actuator, torch, device: str) -> None:
    from isaaclab.utils.types import ArticulationActions

    torch.testing.assert_close(
        actuator.stiffness,
        torch.full_like(actuator.stiffness, 3.0),
    )
    torch.testing.assert_close(
        actuator.damping,
        torch.full_like(actuator.damping, 0.2),
    )

    def action(values):
        position, zeros = _tensor_command(torch, values)
        position = position.to(device)
        zeros = zeros.to(device)
        return ArticulationActions(
            joint_positions=position,
            joint_velocities=zeros.clone(),
            joint_efforts=zeros.clone(),
        )

    zeros = torch.zeros((2, 1), device=device)
    actuator.reset(slice(None))
    first = actuator.compute(action([0.5, 0.7]), zeros, zeros)
    torch.testing.assert_close(first.joint_efforts, zeros)
    actuator.reset(torch.tensor([1], device=device))
    partial = actuator.compute(action([0.0, 0.0]), zeros, zeros)
    torch.testing.assert_close(
        partial.joint_efforts,
        torch.tensor([[1.5], [0.0]], device=device),
    )
    env.scene["robot"].reset()
    cleared = actuator.compute(action([0.0, 0.0]), zeros, zeros)
    torch.testing.assert_close(cleared.joint_efforts, zeros)


def _assert_mjlab_native_state(env, actuator, torch, device: str) -> None:
    from mjlab.actuator import ActuatorCmd

    num_targets = actuator.num_targets
    pos_ids = actuator.global_ctrl_ids[:num_targets]
    vel_ids = actuator.global_ctrl_ids[num_targets:]
    gain = env.sim.model.actuator_gainprm
    torch.testing.assert_close(
        gain[:, pos_ids, 0],
        torch.full_like(gain[:, pos_ids, 0], 3.0),
    )
    torch.testing.assert_close(
        gain[:, vel_ids, 0],
        torch.full_like(gain[:, vel_ids, 0], 0.2),
    )

    def command(values):
        position, zeros = _tensor_command(torch, values)
        position = position.to(device)
        zeros = zeros.to(device)
        return ActuatorCmd(
            position_target=position,
            velocity_target=zeros.clone(),
            effort_target=zeros.clone(),
            pos=zeros.clone(),
            vel=zeros.clone(),
        )

    zeros = torch.zeros((2, 2), device=device)
    actuator.reset(None)
    warmup = actuator.compute(actuator.apply_delay(command([0.1, 0.2])))
    torch.testing.assert_close(
        warmup,
        torch.tensor([[0.1, 0.0], [0.2, 0.0]], device=device),
    )
    delayed = actuator.compute(actuator.apply_delay(command([0.5, 0.7])))
    torch.testing.assert_close(
        delayed,
        torch.tensor([[0.1, 0.0], [0.2, 0.0]], device=device),
    )
    actuator.reset(torch.tensor([1], device=device))
    partial = actuator.compute(actuator.apply_delay(command([0.0, 0.0])))
    torch.testing.assert_close(
        partial,
        torch.tensor([[0.5, 0.0], [0.0, 0.0]], device=device),
    )
    env.scene["robot"].reset()
    cleared = actuator.compute(actuator.apply_delay(command([0.0, 0.0])))
    torch.testing.assert_close(cleared, zeros)


def _run(args, selected) -> dict[str, object]:
    engine_name = args.engine
    import torch
    from instinctlab_engine.preflight import require_preflight

    task = _task(selected, engine_name)
    report = require_preflight(task, engine_name, selected_adapter=selected)
    required = {
        "action/joint_pos": ["joint_position_command"],
        "event/randomize_actuator_gains": ["gain_randomization"],
    }
    actual_requirements = report["requested_capabilities"]["actuator_by_term"]
    if actual_requirements != required:
        raise AssertionError(actual_requirements)

    compiled = selected.compile(task, num_envs=2, device=args.device, strict=True)
    compiled.env_cfg.seed = 12345
    env = compiled.make_env()
    try:
        env.reset()
        asset = env.scene["robot"]
        actuator = _actuator(asset, engine_name)
        if engine_name == "isaacsim":
            _assert_isaac_native_state(env, actuator, torch, args.device)
        else:
            _assert_mjlab_native_state(env, actuator, torch, args.device)

        # Exercise the selected engine's actual action manager and step loop
        # after the direct delay/reset probes have returned the buffer to zero.
        env.reset()
        actions = torch.full((2, 1), 0.4, device=args.device)
        env.step(actions)
        env.step(actions)
        action_term = env.action_manager.get_term("joint_pos")
        target_names = tuple(
            getattr(
                action_term, "target_names", getattr(action_term, "_joint_names", ())
            )
        )
        if target_names != ("joint",):
            raise AssertionError(f"native action selected {target_names!r}")
        return {
            "engine": engine_name,
            "actuator_class": f"{type(actuator).__module__}.{type(actuator).__name__}",
            "action_class": f"{type(action_term).__module__}.{type(action_term).__name__}",
            "gain_randomization": "native-startup-event",
            "full_reset": "passed",
            "partial_reset": "passed",
            "steps": 2,
        }
    finally:
        env.close()


def main() -> int:
    args, selected = _parse()
    app = selected.bootstrap(args)
    try:
        result = _run(args, selected)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        if app is not None:
            app.close()
    return selected.finalize_process(0)


if __name__ == "__main__":
    raise SystemExit(main())
