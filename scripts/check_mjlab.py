"""Compile the declared flat G1 task for mjlab and run it.

The counterpart of ``check_parity.py``, and deliberately not the same kind of check. There is no
mjlab golden in this repository: InstinctMJ is the reference implementation but not a dependency,
so there is no config object to diff against field by field. What can be checked here is what
actually matters for the claim -- that the same declaration compiles for a second engine without
being touched, and that the result runs.

The structural comparison against InstinctMJ's config is done by
``tests/test_mjlab_reference.py``, which reads its source rather than importing it.

    python scripts/check_mjlab.py --num-envs 16
"""

from __future__ import annotations

import argparse
import sys
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.tasks.locomotion.config.flat_g1 import flat_g1

    spec = flat_g1()
    compiled = MjlabAdapter().compile(spec, num_envs=args.num_envs, device=args.device)
    print(compiled.resolution.summary_table())

    env = compiled.env_cls(cfg=compiled.env_cfg, device=args.device)
    try:
        obs, _ = env.reset()
        print(f"policy observation groups: {({k: tuple(v.shape) for k, v in obs['policy'].items()})}")

        # Joint order is decision D1, and the action vector is where it becomes observable. The
        # robot's canonical depth-first order has to be what the action term drives, or a policy
        # trained on one engine drives the wrong joints on the other.
        #
        # What this compares is the running engine against the catalog, so it catches mjlab handing
        # back its own order and catches the task naming a different set of joints. It cannot catch
        # the catalog itself being wrong: reverse both and they still agree. That the catalog is the
        # depth-first walk of the URDF is asserted in tests/test_asset_parity.py, and that the task
        # declares it in tests/test_parity_static.py.
        names = list(env.action_manager.get_term("joint_pos").target_names)
        catalog = list(spec.robot.joint_names)
        print(f"action dimension: {env.action_manager.total_action_dim}")
        print(f"the engine drives the catalog's joint order: {names == catalog}")
        if names != catalog:
            # Ends the script non-zero rather than printing and carrying on. This was a bare print
            # for a long time, which made the only runtime check of D1 on this engine incapable of
            # failing -- and it reads as if it were an assertion.
            print(f"  catalog: {catalog[:6]}")
            print(f"  driven : {names[:6]}")
            raise SystemExit("the action term does not drive the catalog's depth-first joint order")

        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(args.steps):
            obs, reward, terminated, _, _ = env.step(actions)
        print(f"stepped {args.steps}x; reward mean {reward.mean().item():.4f}, {int(terminated.sum())} terminated")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
