"""Compare one of our runs against its own upstream reference run.

The cross-engine comparison (Isaac vs mjlab) is not a parity signal: the two
engines carry documented, deliberate divergences (terrain grid, actuator model,
volume-point counts). The signal that *is* meaningful is each engine against the
reference it was ported from -- mjlab against InstinctMJ, Isaac against main.

Both sides log through the same ``instinct_rl`` runner, so the scalar tags line
up. Everything shared is compared; the tags only one side writes are reported
rather than dropped, because a missing reward term is exactly the kind of drift
this is meant to surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Terms whose absolute scale differs by construction, so a ratio says nothing.
NOISY_PREFIXES = ("Perf/", "Loss/learning_rate", "Policy/")


# The references spell the same quantities differently. Left alone, every
# per-term reward lands in "one side only" and the comparison silently degrades
# to a handful of aggregates -- which is the opposite of what this is for.
# InstinctMJ writes ``Episode_Reward/rewards_<term>/max_episode_len_s`` where we
# write ``Episode_Reward/<term>``; both are the episode sum normalised by the
# max episode length, so they are comparable.
def canonical(tag: str) -> str:
    if tag.startswith("Episode_Reward/rewards_") and tag.endswith("/max_episode_len_s"):
        return "Episode_Reward/" + tag[len("Episode_Reward/rewards_") : -len("/max_episode_len_s")]
    if tag == "Episode_Termination/terrain_out_bound":
        return "Episode_Termination/terrain_out_of_bounds"
    return tag


# Per-term reward spellings the reference also emits; folding them in would
# double-count, so they are dropped once ``canonical`` has picked a variant.
def redundant(tag: str) -> bool:
    return tag.startswith("Episode_Reward/rewards_") and (tag.endswith("/sum") or tag.endswith("/timestep"))


def load(run: Path) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    return {
        canonical(tag): [(e.step, e.value) for e in acc.Scalars(tag)]
        for tag in acc.Tags()["scalars"]
        if not redundant(tag)
    }


def window_mean(series: list[tuple[int, float]], lo: int | None, hi: int | None, tail: int) -> float | None:
    """Mean over an explicit step window, or over each series' own last ``tail`` points.

    The per-series tail is the default because not every tag shares an x-axis: the
    runner writes ``Train/time/*`` against wall-clock seconds and everything else
    against iterations. A single global step window picked from the largest step
    then lands past the end of every iteration-indexed series, and each one drops
    out as "no data in window" -- leaving a comparison that reports agreement on
    two timing scalars and says nothing at all about the rewards.
    """
    if lo is None and hi is None:
        vals = [v for _, v in series[-tail:]]
    else:
        vals = [v for s, v in series if (lo is None or s >= lo) and (hi is None or s <= hi)]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--lo", type=int, default=None)
    ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--top", type=int, default=25, help="How many largest divergences to list.")
    ap.add_argument("--tail", type=int, default=20, help="Points averaged from the end of each series.")
    args = ap.parse_args()

    ours, ref = load(args.ours), load(args.reference)
    lo, hi = args.lo, args.hi
    print(f"ours      = {args.ours}")
    print(f"reference = {args.reference}")
    window = f"iterations [{lo}, {hi}]" if (lo, hi) != (None, None) else f"last {args.tail} points of each series"
    print(f"window    = {window}\n")

    only_ours = sorted(set(ours) - set(ref))
    only_ref = sorted(set(ref) - set(ours))
    if only_ours or only_ref:
        print("== tags on one side only ==")
        for tag in only_ours:
            print(f"  ours only : {tag}")
        for tag in only_ref:
            print(f"  ref  only : {tag}")
        print()

    rows: list[tuple[float, str, float, float]] = []
    for tag in sorted(set(ours) & set(ref)):
        if tag.startswith(NOISY_PREFIXES):
            continue
        a, b = window_mean(ours[tag], lo, hi, args.tail), window_mean(ref[tag], lo, hi, args.tail)
        if a is None or b is None:
            continue
        scale = max(abs(a), abs(b))
        rel = 0.0 if scale == 0 else abs(a - b) / scale
        rows.append((rel, tag, a, b))

    print("== headline ==")
    for tag in ("Train/mean_episode_length", "Train/mean_reward", "Train/mean_reward_0"):
        hit = [r for r in rows if r[1] == tag]
        for rel, t, a, b in hit:
            ratio = a / b if b else float("nan")
            print(f"  {t:44s} ours={a:10.3f}  ref={b:10.3f}  ours/ref={ratio:6.3f}")

    print(f"\n== {args.top} largest relative divergences ==")
    for rel, tag, a, b in sorted(rows, reverse=True)[: args.top]:
        ratio = a / b if b else float("inf")
        print(f"  {rel * 100:5.1f}%  {tag:44s} ours={a:10.4f}  ref={b:10.4f}  ours/ref={ratio:7.3f}")

    close = [r for r in rows if r[0] < 0.05]
    print(f"\n{len(close)}/{len(rows)} shared scalars agree within 5%.")


if __name__ == "__main__":
    main()
