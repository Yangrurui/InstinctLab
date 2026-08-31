#!/usr/bin/env python3
"""Render the registered Perlin-wave tile from its MJLab native geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--difficulty", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be in [0, 1]")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np
    from instinctlab.terrains.registration import PERLIN_WAVE_KIND, register_terrains
    from instinctlab_engine.registry import TerrainExtensionRegistry
    from instinctlab_engine.spec import SubTerrainSpec, TerrainGeneratorSpec

    tile = SubTerrainSpec(
        kind=PERLIN_WAVE_KIND,
        params={
            "amplitude_range": (0.12, 0.36),
            "num_waves": 4,
            "border_width": 0.0,
            "perlin_cfg": {
                "noise_scale": 0.04,
                "noise_frequency": 20,
                "fractal_octaves": 2,
                "fractal_lacunarity": 2.0,
                "fractal_gain": 0.25,
                "centering": True,
            },
        },
    )
    generator = TerrainGeneratorSpec(
        seed=args.seed,
        size=(8.0, 8.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=1.0,
        sub_terrains={"wave": tile},
    )

    registry = TerrainExtensionRegistry(load_entry_points=False)
    register_terrains(registry)
    builder = registry.sub_terrain("mjlab", PERLIN_WAVE_KIND)
    if builder is None:
        raise RuntimeError(f"MJLab did not register {PERLIN_WAVE_KIND!r}")
    native = builder(tile, generator)
    native.size = generator.size
    native.horizontal_scale = generator.horizontal_scale
    native.vertical_scale = generator.vertical_scale
    native.slope_threshold = generator.slope_threshold

    np.random.seed(args.seed)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = native.function(
        args.difficulty,
        spec,
        np.random.default_rng(args.seed),
    )
    surface = output.instinct_surface_mesh
    vertices = np.asarray(surface.vertices)
    faces = np.asarray(surface.faces)

    figure = plt.figure(figsize=(16, 7), constrained_layout=True)
    surface_axis = figure.add_subplot(1, 2, 1, projection="3d")
    surface_plot = surface_axis.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        cmap="terrain",
        linewidth=0.0,
        antialiased=True,
    )
    surface_axis.set_title("Registered perlin_wave: native MJLab collision mesh")
    surface_axis.set_xlabel("x (m)")
    surface_axis.set_ylabel("y (m)")
    surface_axis.set_zlabel("height (m)")
    surface_axis.view_init(elev=34, azim=-128)
    surface_axis.set_box_aspect((1.0, 1.0, 0.35))
    figure.colorbar(surface_plot, ax=surface_axis, shrink=0.65, label="height (m)")

    top_axis = figure.add_subplot(1, 2, 2)
    top_plot = top_axis.tripcolor(
        vertices[:, 0],
        vertices[:, 1],
        faces,
        vertices[:, 2],
        shading="gouraud",
        cmap="terrain",
    )
    top_axis.set_title("Top-down height review")
    top_axis.set_xlabel("x (m)")
    top_axis.set_ylabel("y (m)")
    top_axis.set_aspect("equal")
    figure.colorbar(top_plot, ax=top_axis, shrink=0.78, label="height (m)")
    figure.suptitle(
        "InstinctLab custom terrain extension | "
        f"difficulty={args.difficulty:.2f}, seed={args.seed}, 4 waves + Perlin noise"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)

    report = {
        "builder": f"{builder.__module__}:{builder.__name__}",
        "difficulty": args.difficulty,
        "engine": "mjlab",
        "faces": int(faces.shape[0]),
        "height_max_m": float(vertices[:, 2].max()),
        "height_min_m": float(vertices[:, 2].min()),
        "kind": PERLIN_WAVE_KIND,
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "vertices": int(vertices.shape[0]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
