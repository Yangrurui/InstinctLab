"""Read the episode-length curve out of a training log, and say whether episodes ever end.

The reward curve is the thing everyone watches, and it is the wrong thing to watch for this class of
failure. When mjlab's contact sensor was not asked for the field its contact timers are built from,
both timers stayed at zero for the whole run: ``illegal_contact`` never fired, ``feet_air_time`` paid
nothing, and every episode ran to its time limit. Training still converged -- on a task where falling
over is free -- and the reward curve looked unremarkable. Every static check passed, both parity
comparisons passed, and the one number that said anything was the mean episode length, pinned at
exactly the horizon for all 277 iterations.

So the check here is deliberately not a tolerance between two engines. A tolerance would need a
threshold nobody can justify, and would be either permanently red or permanently green. It is a
structural question instead: do terminations fire at all? An episode length equal to the horizon for
an entire run means the only way an episode can end is by running out of time, which is a statement
about wiring rather than about how well the policy is doing.

Usage::

    python scripts/check_episode_length.py logs/runs/mjlab_gpu1.log
    python scripts/check_episode_length.py logs/runs/isaac_gpu0.log logs/runs/mjlab_gpu1.log

With two logs it also prints the two curves side by side, which is what a cross-engine comparison
looks like when the quantity is behavioural rather than numeric.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

_CURVE = re.compile(r"Mean episode length:\s*([0-9.]+)")


def curve(log: pathlib.Path) -> list[float]:
    """Every mean episode length the runner logged, in order."""
    return [float(match.group(1)) for match in _CURVE.finditer(log.read_text(errors="replace"))]


def horizon() -> int:
    """Steps in a full episode, from the task rather than from a constant written here."""
    from instinctlab.tasks.locomotion.config.flat_g1 import flat_g1

    sim = flat_g1().sim
    return int(round(sim.episode_length_s / sim.step_dt))


def _summarise(name: str, values: list[float], limit: int) -> str:
    head = ", ".join(f"{v:.1f}" for v in values[:3])
    tail = ", ".join(f"{v:.1f}" for v in values[-3:])
    pinned = sum(1 for v in values if abs(v - limit) < 0.5)
    return f"{name}: {len(values)} logged points, first [{head}], last [{tail}], {pinned} at the {limit}-step limit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--min-points",
        dest="min_points",
        type=int,
        default=10,
        help=(
            "Refuse to judge a run with fewer logged points than this, since early episodes are "
            "legitimately long. The runner logs one point per ten training iterations, so the "
            "default asks for roughly a hundred iterations before it will call a run pinned."
        ),
    )
    args = parser.parse_args()

    limit = horizon()
    failures = []
    for log in args.logs:
        values = curve(log)
        if not values:
            failures.append(f"{log}: no episode lengths in it; is this a training log?")
            continue
        print(_summarise(log.name, values, limit))
        if len(values) < args.min_points:
            print(f"  too short to judge ({len(values)} < {args.min_points} logged points)")
            continue
        if all(abs(value - limit) < 0.5 for value in values):
            failures.append(
                f"{log.name}: every episode ran the full {limit} steps, so nothing but the time limit "
                "ever ended one. Terminations are not reaching the environment."
            )

    if len(args.logs) == 2:
        left, right = (curve(log) for log in args.logs)
        if left and right:
            width = min(len(left), len(right))
            print(f"\nboth curves, every {max(1, width // 8)} iterations:")
            for i in range(0, width, max(1, width // 8)):
                print(f"  {i:>5}  {args.logs[0].name}={left[i]:>8.1f}  {args.logs[1].name}={right[i]:>8.1f}")

    for failure in failures:
        print(f"\nFAIL {failure}")
    sys.stdout.flush()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
