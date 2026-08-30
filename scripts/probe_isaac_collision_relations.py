"""Probe Perceptive collision exclusions and PhysX budgets at production scale.

This is a construction/physics diagnostic, not a training run. It verifies
that each cloned robot keeps its own filtered-pair targets, resets the native
Perceptive task, advances zero actions, and records contact/constraint
occupancy after every step.

Example::

    python scripts/probe_isaac_collision_relations.py \
        --num-envs 4096 --steps 5 --headless --device cuda:2 \
        --out logs/diagnostics/perceptive_collision_exclusions_4096.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any


def _parse() -> tuple[argparse.Namespace, Any]:
    from instinctlab_engine import adapter

    engine = adapter("isaacsim")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="Instinct-Perceptive-Shadowing-G1-v0",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    engine.add_cli_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")
    return args, engine


def _verify_filtered_pairs(task: Any, env: Any) -> int:
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics

    targets_by_source: dict[str, set[str]] = {}
    for exclusion in task.scene.collision_exclusions:
        targets_by_source.setdefault(exclusion.body_a, set()).add(exclusion.body_b)

    stage = sim_utils.get_current_stage()
    checked = 0
    for env_index in range(env.num_envs):
        robot_path = f"/World/envs/env_{env_index}/Robot"
        for source_body, target_bodies in targets_by_source.items():
            source_path = f"{robot_path}/{source_body}"
            source = stage.GetPrimAtPath(source_path)
            if not source.IsValid():
                raise RuntimeError(f"missing collision-exclusion source {source_path}")
            relation = UsdPhysics.FilteredPairsAPI(source).GetFilteredPairsRel()
            actual = {str(path) for path in relation.GetTargets()}
            expected = {
                f"{robot_path}/{target_body}" for target_body in target_bodies
            }
            if actual != expected:
                raise RuntimeError(
                    f"collision exclusions for {source_path} are {sorted(actual)}, "
                    f"expected {sorted(expected)}"
                )
            checked += len(expected)
    return checked


def _contact_distribution(sensor: Any, sensor_ref: Any) -> dict[str, Any]:
    import torch
    from instinctlab_engine.bridge.sensors import (
        contact_force_history,
        element_ids,
        element_names,
    )

    history = contact_force_history(sensor, sensor_ref)
    force = torch.linalg.vector_norm(history, dim=-1)
    per_env = force.amax(dim=(1, 2)).float()
    quantiles = torch.quantile(
        per_env,
        torch.tensor((0.0, 0.5, 0.9, 0.95, 0.99, 1.0), device=per_env.device),
    ).detach().cpu().tolist()
    report: dict[str, Any] = dict(
        zip(("min", "median", "p90", "p95", "p99", "max"), quantiles, strict=True)
    )
    report["fraction_over_500"] = float(per_env.gt(500.0).float().mean().cpu())

    selected_names = [
        element_names(sensor)[index] for index in element_ids(sensor, sensor_ref)
    ]
    per_body = force.amax(dim=1)
    body_reports = []
    for index, name in enumerate(selected_names):
        body_force = per_body[:, index].float()
        body_reports.append(
            {
                "body": name,
                "fraction_over_500": float(
                    body_force.gt(500.0).float().mean().cpu()
                ),
                "p95": float(torch.quantile(body_force, 0.95).cpu()),
                "max": float(body_force.max().cpu()),
            }
        )
    report["top_bodies_over_500"] = sorted(
        body_reports,
        key=lambda item: (item["fraction_over_500"], item["p95"]),
        reverse=True,
    )[:8]
    return report


def _run(args: argparse.Namespace, engine: Any, resources: ExitStack) -> dict[str, Any]:
    import torch

    from instinctlab.tasks.registry import asset_id
    from instinctlab.tasks.registry import spec as task_spec
    from instinctlab_engine.diagnostics.contact_overflow import (
        check_contact_overflow,
        contact_budget_snapshot,
    )
    from instinctlab_engine.preflight import require_preflight

    robot = engine.robot_spec(asset_id(args.task))
    task = task_spec(args.task, robot)
    preflight = require_preflight(
        task,
        "isaacsim",
        selected_adapter=engine,
    )
    if len(task.scene.collision_exclusions) != 4:
        raise RuntimeError(
            f"expected four Perceptive collision exclusions, got "
            f"{len(task.scene.collision_exclusions)}"
        )

    compiled = engine.compile(
        task,
        num_envs=args.num_envs,
        device=args.device,
        strict=True,
    )
    compiled.resolution.require_clean()
    compiled.env_cfg.seed = args.seed
    env = compiled.make_env()
    resources.callback(env.close)

    relation_targets_checked = _verify_filtered_pairs(task, env)
    check_contact_overflow(env, phase="construction")
    snapshots = [{"phase": "construction", **contact_budget_snapshot(env)}]

    env.reset()
    check_contact_overflow(env, phase="reset")
    snapshots.append({"phase": "reset", **contact_budget_snapshot(env)})

    contact_ref = task.mdp.terminations["illegal_reset_contact"].params["sensor"]
    contact_sensor = env.scene.sensors[contact_ref.name]
    contacts = {"after_reset": _contact_distribution(contact_sensor, contact_ref)}
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    done_fractions: list[dict[str, float]] = []
    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            result = env.step(actions)
            check_contact_overflow(env, phase=f"step_{step}")
            snapshots.append(
                {"phase": f"step_{step}", **contact_budget_snapshot(env)}
            )
            contacts[f"after_step_{step}"] = _contact_distribution(
                contact_sensor, contact_ref
            )
            robot_data = env.scene["robot"].data
            if not torch.isfinite(robot_data.joint_pos).all():
                raise RuntimeError(f"non-finite robot joint state after step {step}")
            if len(result) == 5:
                _observations, rewards, terminated, truncated, _extras = result
                if not torch.isfinite(rewards).all():
                    raise RuntimeError(f"non-finite rewards after step {step}")
                done_fractions.append(
                    {
                        "step": step,
                        "terminated": float(terminated.float().mean().cpu()),
                        "truncated": float(truncated.float().mean().cpu()),
                    }
                )

    return {
        "engine": "isaacsim",
        "task": args.task,
        "device": args.device,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "collision_exclusions": [
            list(exclusion.pair) for exclusion in task.scene.collision_exclusions
        ],
        "relation_targets_checked": relation_targets_checked,
        "contact_constraint_snapshots": snapshots,
        "illegal_reset_contact_force": contacts,
        "done_fractions": done_fractions,
        "selected_components": preflight["selected_components"],
        "providers": preflight["providers"],
    }


def main() -> int:
    args, engine = _parse()
    with ExitStack() as resources:
        app = engine.bootstrap(args)
        if app is not None:
            resources.callback(app.close)
        try:
            report = _run(args, engine, resources)
        except BaseException:  # Kit shutdown can otherwise hide the original traceback.
            traceback.print_exc()
            raise
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n")
            print(f"[PROBE] Wrote {args.out}", flush=True)
        print(rendered, flush=True)
    return engine.finalize_process(0)


if __name__ == "__main__":
    raise SystemExit(main())
