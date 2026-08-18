"""Agent configurations for locomotion, kept where no engine can reach them.

Hyperparameters are engine-independent -- a learning rate does not know which simulator produced
the rollout -- but the modules they used to live in were not: the task package under ``config/g1/``
registers Gym ids and imports Isaac Lab env configs, so reading a hidden layer width from there
required Isaac Sim to be installed and started.

Definitions live here; ``config/g1/agents/instinct_rl_ppo_cfg.py`` re-exports them so main's Gym
registration keeps working against the same class object rather than a copy that can drift.
"""
