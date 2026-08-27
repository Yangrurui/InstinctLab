# BeyondMimic shadowing

The common BeyondMimic declaration is in `beyondmimic_env_cfg.py`. G1 data and factories are in
`config/g1/beyondmimic_plane_cfg.py`, matching main and InstinctMJ. The files remain engine-neutral
and use the generic train and play scripts documented in the parent README.

The public Unitree-retargeted LAFAN1 source is
`lvhaidong/LAFAN1_Retargeting_Dataset` on Hugging Face. Its G1 CSV files use a pelvis root,
an `xyzw` quaternion, and a legs-first joint order, so they are not runtime clips. Convert them
through the explicit torso-root/DFS bridge before training:

```bash
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python \
  scripts/lafan1_csv_to_instinct.py \
  --src /root/Datasets/LAFAN1_Retargeting_Dataset/g1 \
  --tgt /root/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz
```

The output manifest records source and converted checksums. Both engines must consume the same
converted directory; do not load the public CSV positionally into a native articulation.
