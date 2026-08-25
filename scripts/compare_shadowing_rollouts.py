"""Compare the fixed-input evidence produced by ``probe_shadowing_rollout.py``."""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacsim", type=Path, required=True)
    parser.add_argument("--mjlab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    with np.load(args.isaacsim) as isaacsim, np.load(args.mjlab) as mjlab:
        numeric = (
            "action",
            "motion_start_s",
            "joint_pos",
            "joint_vel",
            "root_pos",
            "root_quat",
            "root_vel",
            "motion_pos",
            "reward",
        )
        fields = {}
        for name in numeric:
            isaac_value = isaacsim[name]
            mj_value = mjlab[name]
            delta = np.abs(isaac_value - mj_value)
            fields[name] = {"rollout_max_abs": float(delta.max())}
            if name not in {"action", "motion_start_s", "reward"}:
                fields[name]["initial_max_abs"] = float(delta[0].max())
        report = {
            "inputs": {
                "isaacsim": json.loads(str(isaacsim["metadata"])),
                "mjlab": json.loads(str(mjlab["metadata"])),
            },
            "fields": fields,
            "done": {
                "equal": bool(np.array_equal(isaacsim["done"], mjlab["done"])),
                "isaacsim_by_env": isaacsim["done"].T.tolist(),
                "mjlab_by_env": mjlab["done"].T.tolist(),
            },
            "interpretation": (
                "Initial reference, robot joint state, root pose, and fixed action are portable. "
                "Post-step state is engine-native evidence and is not expected to match across "
                "PhysX and MuJoCo; auto-reset makes states after the first done incomparable."
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
