# MARL Asteroids

Team-based multi-agent Asteroids (red vs. blue) as a PettingZoo `ParallelEnv`, with a shared-parameter
PPO team policy trained by league self-play (Stable-Baselines3). Originally an ASEN 5264 course project;
the original code, checkpoints, and results are preserved at tag `v0-legacy`.

## Layout

```
marl_asteroids/
  config.py      GameConfig, RewardConfig, scenario registry, static_battle, handicap helpers
  entities.py    Ship / Asteroid / Bullet
  env.py         TeamAsteroidsEnv: physics, collisions, ego-centric observations, rewards
  render.py      Pygame renderer (starfield, flames, particles, trails, HUD) - no sim state
  policies.py    Batched Policy interface: scripted bots, NetworkPolicy, human, DelayedPolicy
  matches.py     Match runner, plain and mirrored (color-swapped) -> MatchStats
  opponents.py   OpponentPool: scripted + frozen snapshots + live self, PFSP sampling
  vec_env.py     TeamVecEnv (SB3 VecEnv, parameter sharing, death masking) + multiprocess version
  training.py    Self-play and win-rate evaluation callbacks
scripts/
  train.py       Train (presets: baseline, selfplay)
  evaluate.py    Evaluation matrix -> CSV (mirrored by default)
  league.py      Round-robin league, Bradley-Terry ratings on the Elo scale
  watch.py       Watch agents, play as a human with difficulty handicaps, record video
tests/           pytest suite (API compliance, seeding, reward/obs/mechanics regressions)
models/legacy/   Checkpoints from the original project (need tag v0-legacy to run)
results/legacy/  Original evaluation CSVs, figures, and analysis notebook
runs/            TensorBoard logs, training logs (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Usage

Run from the repo root.

```bash
# Train
uv run python scripts/train.py --preset baseline --run baseline_v2 --timesteps 8e6
uv run python scripts/train.py --preset selfplay --run selfplay_v2 --timesteps 20e6
uv run tensorboard --logdir runs/tensorboard

# Evaluate (mirrored: every seed played with the teams swapped)
uv run python scripts/evaluate.py --blue models/selfplay_v2/best_model.zip \
    --red heuristic_0.25 heuristic_0.5 heuristic_0.75 heuristic_1.0 --games 500 --out results/sp.csv
uv run python scripts/league.py --agents heuristic_0.5 heuristic_1.0 \
    models/baseline_v2/best_model.zip models/selfplay_v2/best_model.zip

# Watch, play, record
uv run python scripts/watch.py --scenario demo_team2 --blue models/selfplay_v2/best_model.zip
uv run python scripts/watch.py --scenario demo_duel --blue human --red models/selfplay_v2/best_model.zip \
    --difficulty easy
uv run python scripts/watch.py --scenario demo_team3 --games 3 --record demo.mp4
```

Human controls: `W`/`Up` thrust, `A`/`D` or arrows to turn, `Space` to shoot, `Esc` to quit.

`--blue` / `--red` take one spec per team, or a comma-separated list assigning each ship
individually (cycled). So `--blue human,heuristic_1.0` puts you in `blue_0` with an AI wingman, which
is the most direct way to judge whether teammates actually cooperate.

## Scenarios

Randomized spawns, for training and evaluation:

| Name | Teams | Asteroids | Notes |
|---|---|---|---|
| `duel` | 1v1 | 0 | open space |
| `duel_rocks` | 1v1 | 3 | splitting |
| `team2` | 2v2 | 4 | splitting |
| `team3` | 3v3 | 6 | splitting |
| `team2_vs_1` | 1v2 | 4 | imbalanced; rewards cooperation |
| `asteroid_storm` | 2v2 | 10 | dense, fast field |
| `obstacles` | 2v2 | 5 | indestructible cover |

Larger arenas (ships, weapons, and sensing unchanged, so these are genuinely bigger worlds rather
than a zoomed-in view):

| Name | Map | Teams | Asteroids |
|---|---|---|---|
| `duel_large` | 1600² | 1v1 | 6 |
| `team2_large` | 1600² | 2v2 | 8 |
| `team3_large` | 2000² | 3v3 | 15 |
| `team3_huge` | 3200² | 3v3 | 24 |
| `demo_team2_large` | 2400² | 2v2 | 4 (static) |

Any scenario can be grown on the fly with `--scale`, which also scales `max_frames` and spawn
separation, and multiplies the asteroid count (tune with `--asteroid-density`):

```bash
uv run python scripts/watch.py --scenario team3 --scale 5 --asteroid-density 0.4
uv run python scripts/evaluate.py --blue model.zip --red heuristic_1.0 --scale 3   # generalization
```

`scale_map` deliberately does **not** scale `sensor_range`: that length is a tactical scale tied to
weapon reach (`bullet_speed * bullet_lifespan` is about 480px), not a property of the map. A bigger
arena therefore means longer approaches and coarser range estimates, not blindness.

Static and mirror-symmetric, for demos, videos, debugging, and human play: `demo_duel`,
`demo_team2`, `demo_team3`, `demo_open`. These use a 1200x1200 map with proportionally larger ships,
teams spawn perpendicular to each other (so both must turn to engage), and the layout is exactly
symmetric under a 180-degree rotation. Verified by test: identical deterministic policies on both
sides always draw.

For *measurement*, prefer mirrored evaluation over static spawns. A static scenario with
deterministic policies replays the same game every time, so 500 games is a sample size of one.

Scripted opponents: `dummy`, `evasion_easy`, `evasion`, `heuristic_<p>` (acts with probability p).

## Environment

- Toroidal map. Actions `MultiDiscrete([2, 3, 2])` = (thrust, turn, shoot), repeated for
  `action_repeat` = 2 physics frames. Episodes end when a team is eliminated or at `max_frames`.
- Ships: speed capped at `max_speed`, bullets carry the ship's velocity, `fire_rate`-frame reload.
  Friendly fire is on. Ramming and asteroid collisions are lethal.
- Asteroids: with `asteroids_split` a hit rock breaks into two smaller, faster fragments (the
  smallest size is destroyed outright), so clearing a rock creates a new hazard. Without it, rocks
  are indestructible cover that absorbs shots.
- Observation (209-d, ego ship's heading frame, clipped to [-1, 1]): ego state plus k-nearest
  enemies (3), allies excluding self (2), asteroids (8), and bullets that can hurt the ego (6).
  Each slot has a presence flag, **bearing as a unit vector**, a **saturating range**
  (`dist / (dist + sensor_range)`, so 0.5 exactly at `sensor_range`), relative velocity,
  closest-approach time and miss distance. Enemies add the lead-aim intercept direction, the
  predicted miss if I fire now, and whether they are aiming at me. Allies add their heading and my
  predicted friendly-fire miss.
- **Lengths are normalized by `sensor_range` (a fixed length), not by map size**, so the world can be
  resized without changing what the policy sees, and the same field is the natural hook for real
  limited-range sensors later. Verified by test across 800 / 1600 / 4000 maps.
- Bearing is a unit vector and range saturates smoothly, rather than clipping `dx`/`dy` to
  [-1, 1] independently. Clipping components destroyed bearing past the normalization scale: every
  distant object read as `(1, 1)`, so objects at 20°, 45°, and 70° were indistinguishable and large
  maps were unnavigable.
- Reward (`RewardConfig`): kill +1, death -1, team kill -1 (shooter); mixed across the team by
  `team_spirit`; shared win/loss at the end, also delivered to teammates who already died; plus
  potential-based shaping (aim, range, hazard), which cannot change the optimal policy.

## Difficulty handicaps for human play

`watch.py --difficulty easy|normal|hard|brutal` handicaps the AI side without retraining:

- **Reaction delay** (`policies.DelayedPolicy`): the agent acts on an observation from N decisions
  ago. Affects observation-driven policies only; scripted bots read the env directly.
- **Reload time** (`config.set_team_fire_rate`): caps sustained damage output.

Note reload time does not change the opening exchange, because a single hit is lethal and the first
shot decides most duels. Reaction delay is the effective lever today.

## Training setup

- One shared policy controls every blue ship. Each learner ship is a VecEnv slot; 1v1, 2v2, 3v3, and
  2v1 games are mixed in one batch.
- Dead ships keep a zero-observation slot until the game ends so they still receive the team outcome.
- `selfplay` preset: the red team is sampled per game from the live learner (35%), scripted bots
  (20%), or a frozen snapshot (45%, taken every 250k steps, prioritized by how often they beat the
  learner).
- Checkpoints: `models/<run>/best_model.zip` (best mean win rate), `final_model.zip`, `snapshots/`,
  `eval_history.json`. Roughly 3-4k agent-steps/s on 12 cores.

## Changes from the original project

Bugs fixed: the ally observation slot contained the ego ship itself; ship speed was uncapped
(~14.6 px/frame against a 3.0 normalization, so observations left their bounds); friendly-fire kills
paid the kill reward; the asteroid penalty was double-counted while bullets were never penalized;
entities ignored per-scenario config; seeds were ignored; `EvalCallback` ran on the training env and
selected "best" by shaped reward instead of win rate. Later, in this round: asteroid fragments from a
stationary rock never separated, and a static arena at `inset=0.25` was silently asymmetric because
the teams sat exactly antipodally on the torus.
