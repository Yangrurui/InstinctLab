"""Parse the two Instinct-Parkour-Target-G1 TensorBoard runs and print an L7 report.

Parity sits on ``Episode_Terrain/aligned_*`` (the documented-aligned sub-terrains),
not on the aggregate episode-length curve. ``volume_points_penetration`` is printed
but not used as a parity signal.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ALIGNED = (
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "pyramid_stairs_high",
    "pyramid_stairs_inv_high",
    "boxes",
    "hf_pyramid_slope_inv",
)
EXCLUDED = ("pyramid_stairs", "pyramid_stairs_inv", "dense_boxes", "mesh_boxes")
SKIP_TERMS = {"volume_points_penetration"}


def load_events(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [(e.step, float(e.value)) for e in acc.Scalars(tag)]
    return out


def series(events: dict[str, list[tuple[int, float]]], *candidates: str) -> list[tuple[int, float]]:
    for name in candidates:
        if name in events:
            return events[name]
    return []


def window(values: list[tuple[int, float]], lo: int, hi: int) -> list[float]:
    return [v for s, v in values if lo <= s <= hi]


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def find_run(root: Path) -> Path:
    if (root / "agent.json").exists():
        return root
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / "agent.json").exists())
    if not candidates:
        raise FileNotFoundError(f"no run directory under {root}")
    return candidates[-1]


def overflow_files(run: Path) -> dict[str, dict]:
    out = {}
    for name in ("overflow_after_construction.json", "overflow_late_training.json", "overflow_peaks.json"):
        path = run / name
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def report(isaac: dict[str, list[tuple[int, float]]], mjlab: dict[str, list[tuple[int, float]]]) -> str:
    lines: list[str] = []
    length_i = series(isaac, "Train/mean_episode_length")
    length_m = series(mjlab, "Train/mean_episode_length")
    reward_i = series(isaac, "Train/mean_reward_0")
    reward_m = series(mjlab, "Train/mean_reward_0")
    aligned_i = series(isaac, "Episode_Terrain/aligned_length", "Episode/Episode_Terrain/aligned_length")
    aligned_m = series(mjlab, "Episode_Terrain/aligned_length", "Episode/Episode_Terrain/aligned_length")
    excluded_i = series(isaac, "Episode_Terrain/excluded_length", "Episode/Episode_Terrain/excluded_length")
    excluded_m = series(mjlab, "Episode_Terrain/excluded_length", "Episode/Episode_Terrain/excluded_length")
    style_i = series(isaac, "Step/discriminator_reward")
    style_m = series(mjlab, "Step/discriminator_reward")
    acc_i = series(isaac, "Train/discriminator_accuracy")
    acc_m = series(mjlab, "Train/discriminator_accuracy")

    n = min(len(length_i), len(length_m), 250)
    lines.append(f"logged iterations: isaac={len(length_i)} mjlab={len(length_m)}")
    for label, a, b in (
        ("aggregate episode length", length_i, length_m),
        ("aligned episode length", aligned_i, aligned_m),
        ("excluded episode length", excluded_i, excluded_m),
        ("mean reward", reward_i, reward_m),
        ("AMP style (disc reward)", style_i, style_m),
        ("AMP discriminator accuracy", acc_i, acc_m),
    ):
        lines.append(
            f"{label}: isaac first={fmt(_first(a))} last={fmt(_last(a))} mid-late={fmt(mean(window(a, n // 2, n)))} | "
            f"mjlab first={fmt(_first(b))} last={fmt(_last(b))} mid-late={fmt(mean(window(b, n // 2, n)))}"
        )

    lines.append("\nPer-terrain last-window length (iters last 20%):")
    lo = max(0, n - max(20, n // 5))
    for name in (*ALIGNED, *EXCLUDED):
        tag = f"Episode_Terrain/length/{name}"
        ia = series(isaac, tag, f"Episode/{tag}")
        mb = series(mjlab, tag, f"Episode/{tag}")
        mark = "ALIGNED" if name in ALIGNED else "EXCLUDED"
        lines.append(f"  [{mark}] {name}: isaac={fmt(mean(window(ia, lo, n)))} mjlab={fmt(mean(window(mb, lo, n)))}")

    lines.append("\nReward terms (mean over last 20% iters; skip volume_points_penetration as a parity signal):")
    terms = sorted(
        {
            tag.split("/", 1)[1]
            for tag in list(isaac) + list(mjlab)
            if tag.startswith("Episode_Reward/") or tag.startswith("Episode/Episode_Reward/")
        }
    )
    for term in terms:
        ia = series(isaac, f"Episode_Reward/{term}", f"Episode/Episode_Reward/{term}")
        mb = series(mjlab, f"Episode_Reward/{term}", f"Episode/Episode_Reward/{term}")
        note = "  [not a parity signal]" if term in SKIP_TERMS else ""
        lines.append(f"  {term}: isaac={fmt(mean(window(ia, lo, n)), 4)} mjlab={fmt(mean(window(mb, lo, n)), 4)}{note}")

    lines.append("\nTerminations (last 20%; units may differ — Isaac proportion, mjlab often count):")
    for name in ("time_out", "terrain_out_of_bounds", "base_contact", "bad_orientation", "root_height"):
        ia = series(isaac, f"Episode_Termination/{name}", f"Episode/Episode_Termination/{name}")
        mb = series(mjlab, f"Episode_Termination/{name}", f"Episode/Episode_Termination/{name}")
        lines.append(f"  {name}: isaac={fmt(mean(window(ia, lo, n)), 4)} mjlab={fmt(mean(window(mb, lo, n)), 4)}")
    return "\n".join(lines)


def _first(values: list[tuple[int, float]]) -> float | None:
    return values[0][1] if values else None


def _last(values: list[tuple[int, float]]) -> float | None:
    return values[-1][1] if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--mjlab", type=Path, required=True)
    args = parser.parse_args()
    isaac_run = find_run(args.isaac)
    mjlab_run = find_run(args.mjlab)
    print(f"isaac run: {isaac_run}")
    print(f"mjlab run: {mjlab_run}")
    print(
        json.dumps({"isaac_overflow": overflow_files(isaac_run), "mjlab_overflow": overflow_files(mjlab_run)}, indent=2)
    )
    print(report(load_events(isaac_run), load_events(mjlab_run)))


if __name__ == "__main__":
    main()
