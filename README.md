# MARL Asteroids

Team-based multi-agent Asteroids (red vs. blue) built as a PettingZoo `ParallelEnv`, with PPO agents trained
through Stable-Baselines3 + SuperSuit and a curriculum of scenarios. Originally an ASEN 5264 course project.

## Layout

```
marl_asteroids/     Core package
  config.py         GameConfig, EntityState spawn blueprints, scenario presets
  entities.py       Ship / Asteroid / Bullet
  env.py            TeamAsteroidsEnv (physics, collisions, observations)
  rewards.py        DenseRewardWrapper (reward shaping, win condition, time limit)
  policies.py       Scripted policies: dummy, heuristic, evasion, human keyboard
scripts/
  train.py          PPO training vs. scripted opponents (curriculum via load_path)
  evaluate.py       Visual sandbox, single eval, and bulk evaluation matrix -> CSV
  play.py           Play as red_0 with the keyboard
models/legacy/      Checkpoints from the original project
results/legacy/     Original evaluation CSVs, figures, and analysis notebook
archive/            Older scripts kept for reference (not maintained)
runs/               TensorBoard logs and eval logs (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Run everything from the repo root:

```bash
uv run python scripts/play.py        # play against bots
uv run python scripts/train.py       # train (edit the launch config at the bottom)
uv run python scripts/evaluate.py    # set MODE = VISUAL / SINGLE_EVAL / BULK_EVAL
uv run tensorboard --logdir runs/tensorboard
```

## Environment summary

- Toroidal 800x800 map; ships thrust, turn, and shoot (`MultiDiscrete([2, 3, 2])`).
- Observation (109-d): ego state + k-nearest asteroids (5), enemies (3), bullets (3), allies (2),
  each as relative position/velocity + type flags.
- Base reward: +1 kill / -1 death; `DenseRewardWrapper` amplifies it and adds shaping terms.
