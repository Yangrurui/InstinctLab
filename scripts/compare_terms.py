"""Diff two term probes and say which terms agree.

Tolerances are per term and each one is argued for rather than set to whatever passed. A term that
reads only kinematics should agree to float precision; a term reading a solver output should not be
expected to agree at all, and saying so is more useful than a loose tolerance that hides the day it
stops agreeing for a different reason.

    python scripts/compare_terms.py /tmp/isaacsim_terms.json /tmp/mjlab_terms.json
"""

from __future__ import annotations

import argparse
import json
import sys
import torch
from pathlib import Path

# Terms expected to agree, with the tolerance each is held to.
#
# 1e-5 is float32 accumulation over 29 joints and a few trigonometric functions -- the two engines
# compute these from the same numbers in a different order, and nothing else should separate them.
TOLERANCES = {
    "obs/policy/base_ang_vel": 1e-5,
    "obs/policy/projected_gravity": 1e-5,
    "obs/policy/velocity_commands": 1e-6,
    "obs/policy/joint_pos": 1e-5,
    "obs/policy/joint_vel": 1e-5,
    "obs/policy/actions": 1e-6,
    "obs/critic/base_lin_vel": 1e-5,
    "obs/critic/base_ang_vel": 1e-5,
    "obs/critic/projected_gravity": 1e-5,
    "obs/critic/velocity_commands": 1e-6,
    "obs/critic/joint_pos": 1e-5,
    "obs/critic/joint_vel": 1e-5,
    "obs/critic/actions": 1e-6,
    "reward/termination_penalty": 1e-6,
    "reward/track_lin_vel_xy_exp": 1e-5,
    "reward/track_ang_vel_z_exp": 1e-5,
    "reward/flat_orientation_l2": 1e-5,
    "reward/stand_still": 1e-5,
    "reward/dof_pos_limits": 1e-5,
    "reward/joint_deviation_hip": 1e-5,
    "reward/joint_deviation_arms": 1e-5,
    "reward/joint_deviation_torso": 1e-5,
    "reward/joint_deviation_knee": 1e-5,
    "reward/lin_vel_z_l2": 1e-5,
    "reward/action_rate_l2": 1e-5,
    "done/time_out": 0.0,
}

# Terms that are not expected to agree, and why. Compared anyway and reported, because the size of
# a disagreement is worth knowing even when its existence is not news.
EXPECTED_TO_DIFFER = {
    "reward/feet_air_time": (
        "Air time accumulates across steps. Both robots were placed into the same state, not "
        "brought there by the same history, so the sensor's timers hold whatever the reset left."
    ),
    "reward/feet_slide": (
        "Reads the contact force, which is the normal component on Isaac Lab and the whole vector "
        "on mjlab, and the link velocity, which Isaac Lab reports about the centre of mass. Each "
        "backend keeps its own implementation for exactly this reason."
    ),
    "reward/dof_acc_l2": (
        "Joint acceleration is a finite difference of measured velocity on one engine and a solver "
        "output on the other. Writing a state gives the two different histories to differentiate."
    ),
    "reward/dof_torques_l2": (
        "Applied torque excludes passive terms on one engine and includes them on the other, and "
        "neither has been stepped since the state was written."
    ),
    "done/base_contact": (
        "Contact state depends on the same force reading as feet_slide, and on whether the written "
        "pose happens to intersect the ground."
    ),
}


def _probe(engine: str, out: Path) -> None:
    """Run one probe in its own interpreter, which is the only way to have both engines."""
    import subprocess

    script = Path(__file__).parent / "probe_terms.py"
    result = subprocess.run(
        [sys.executable, str(script), "--engine", engine, "--out", str(out)],
        capture_output=True,
        text=True,
        # Isaac Sim's shutdown decides the exit status itself, so the file is the only honest
        # signal that the probe worked.
        check=False,
    )
    if not out.is_file():
        print(result.stdout[-3000:], result.stderr[-3000:], sep="\n")
        raise SystemExit(f"The {engine} probe produced nothing.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, nargs="?", default=Path("/tmp/isaacsim_terms.json"))
    parser.add_argument("right", type=Path, nargs="?", default=Path("/tmp/mjlab_terms.json"))
    parser.add_argument("--run", action="store_true", help="run both probes first")
    args = parser.parse_args()

    if args.run:
        for engine, out in (("isaacsim", args.left), ("mjlab", args.right)):
            print(f"probing {engine} ...")
            _probe(engine, out)

    left, right = json.loads(args.left.read_text()), json.loads(args.right.read_text())
    left_meta, right_meta = left.pop("_meta"), right.pop("_meta")

    if left_meta["catalog_joint_order"] != right_meta["catalog_joint_order"]:
        print("the two probes disagree about the catalog's joint order; nothing else is comparable")
        return 1
    same_native_order = left_meta["engine_joint_order"] == right_meta["engine_joint_order"]
    print(f"engines: {left_meta['engine']} vs {right_meta['engine']}")
    print(f"native joint orders identical: {same_native_order} (readings are reindexed either way)")

    # Root-state readings are diagnostics rather than terms, and the two engines expose different
    # sets of them -- Isaac Lab has an unqualified ``root_lin_vel_w`` where mjlab makes the caller
    # say which frame it means. Compared where both have them, reported separately.
    diagnostics = sorted(name for name in set(left) & set(right) if name.startswith("state/"))
    left = {name: value for name, value in left.items() if not name.startswith("state/")}
    right = {name: value for name, value in right.items() if not name.startswith("state/")}

    if set(left) != set(right):
        print(f"\nterms present on one side only: {set(left) ^ set(right)}")
        return 1

    failures, unlisted = [], []
    print(f"\n{'term':<38} {'max |difference|':>17}  verdict")
    print("-" * 78)
    for name in sorted(left):
        gap = (torch.tensor(left[name]) - torch.tensor(right[name])).abs().max().item()
        if name in TOLERANCES:
            ok = gap <= TOLERANCES[name]
            print(f"{name:<38} {gap:>17.3e}  {'agrees' if ok else 'DISAGREES'}")
            if not ok:
                failures.append((name, gap, TOLERANCES[name]))
        elif name in EXPECTED_TO_DIFFER:
            print(f"{name:<38} {gap:>17.3e}  not compared")
        else:
            print(f"{name:<38} {gap:>17.3e}  UNCLASSIFIED")
            unlisted.append(name)

    print("\nstate the two engines were put into, where both report it the same way:")
    for name in diagnostics:
        gap = (
            (
                torch.tensor(json.loads(args.left.read_text())[name])
                - torch.tensor(json.loads(args.right.read_text())[name])
            )
            .abs()
            .max()
            .item()
        )
        print(f"  {name:<36} {gap:>17.3e}")

    agreed = sum(1 for name in left if name in TOLERANCES) - len(failures)
    print(f"\n{agreed} term(s) agree within tolerance, {len(EXPECTED_TO_DIFFER)} not compared.")
    for name, gap, tolerance in failures:
        print(f"  {name}: {gap:.3e} exceeds {tolerance:.0e}")
    for name in unlisted:
        print(f"  {name}: neither a tolerance nor a reason is recorded for this term")
    return 1 if failures or unlisted else 0


if __name__ == "__main__":
    sys.exit(main())
