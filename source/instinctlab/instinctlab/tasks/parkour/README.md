# Parkour Task

## Basic Usage Guidelines

### Parkour Task

**Task ID:** `Instinct-Parkour-Target-G1`

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set `PARKOUR_MOTION_CLIP` to the reference motion you want to use.

2. Train the policy:
```bash
python scripts/train.py --engine <isaacsim|mjlab> --headless --task Instinct-Parkour-Target-G1
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):

```bash
python scripts/play.py --engine <isaacsim|mjlab> --task Instinct-Parkour-Target-G1 --load_run <run_name>
```

4. Export trained policy (load_run must be provided, absolute path is recommended):

```bash
python scripts/play.py --engine <isaacsim|mjlab> --task Instinct-Parkour-Target-G1 --load_run <run_name> --num_envs 1 --export-onnx --export-only --deployment-runtime onnxruntime
```

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--keyboard_control`: Enable keyboard control during playing
- `--load_run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export a single self-contained `policy.onnx` for onboard deployment
- `--export-only`: Verify the exported policy and exit without opening a viewer
