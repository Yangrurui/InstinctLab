# Unified shadowing tasks

Shadowing is declared as `TaskSpec` and compiled by the selected engine adapter.  Importing this
package does not import Isaac Lab, MJLab, Gym or either native environment configuration.

Use the generic entry points:

```bash
python scripts/train.py --engine isaacsim --task Instinct-Shadowing-WholeBody-Plane-G1-v0
python scripts/train.py --engine mjlab --task Instinct-Shadowing-WholeBody-Plane-G1-v0
python scripts/play.py --engine mjlab --task Instinct-Shadowing-WholeBody-Plane-G1-v0 --checkpoint PATH
```

The registry contains whole-body, perceptive, perceptive VAE, perceptive HOI, BeyondMimic and
Perceptive OneMotion train/play contracts. Their files follow main and InstinctMJ: each family has
its own `*_env_cfg.py`, and each G1 task keeps its concrete values in the corresponding
`config/g1/*_cfg.py`. No task-local CLI mutates configuration at import time.

Native contact, actuator, solver and object semantics intentionally remain in the corresponding
engine adapter. See the repository-root `HANDOFF.md` for the current reference boundary.
