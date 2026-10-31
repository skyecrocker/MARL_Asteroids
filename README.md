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
  session.py     MatchSpec -> Match: builds playable games (shared by the app and watch.py)
  checkpoints.py Finds trained models compatible with the current observation size
  app/           Desktop app: menus, match setup, game, results, settings (pygame_gui)
scripts/
  app.py         Launch the desktop app (same as `uv run marl-asteroids`)
  train.py       Train (presets: baseline, selfplay)
  evaluate.py    Evaluation matrix -> CSV (mirrored by default)
  league.py      Round-robin league, Bradley-Terry ratings on the Elo scale
  watch.py       Watch agents, play as a human with difficulty handicaps, record video
tests/           pytest suite (API compliance, seeding, reward/obs/mechanics regressions, app)
docs/app.md      Desktop app design
models/phaseA/   Phase A checkpoints: self-play final, baseline, and difficulty-ladder snapshots
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

## Playing: the desktop app

```bash
uv run marl-asteroids
```

The main menu has **Play**, **Gallery**, and **Settings**. Match setup picks the mode, arena size
(Mini 600, Small 1200, Medium 2400, Large 4800), ships per team, spawns (deathmatch and sudden death), asteroid
count, rock type, time limit, and a controller for every ship: you, any compatible trained model
(shown with its rating), or a scripted bot. You start in Blue 1; to just watch, give your ship to an
AI. Your last setup is remembered.

### Game modes

| Mode | Rules |
|---|---|
| **Deathmatch** | Destroyed ships respawn after 3 s, at a random spot away from the enemy, with 1.5 s of spawn protection. Score is enemy ships destroyed by your team's fire (team kills and crashes count for nobody). Most kills when time runs out wins; level scores draw. Always has a clock (default 5 min). |
| **Sudden death** | No respawns. The last team with ships left wins. (What the trained agents were trained on, and what every named scenario uses.) |
| **Station siege** | Each team has a station. Destroyed ships respawn at it after 3 s, with 1.5 s of spawn protection. Destroy the enemy station to stop their respawns, then wipe out the fleet. Default limit 5 min; the station with more health wins a timeout. |
| **Survival** | Blue against growing waves of red. The first wave is half blue's size; each wave adds a ship, up to the red team size. Clearing a wave brings fallen blue ships back and repairs every hull. Lasting until a time limit counts as a win. |

A station's shield regenerates continuously at 1.6 points/s, between one and two ships' damage
output, so a lone attacker barely dents it while two or three take it down in 10-25 s
(`tests/test_modes.py` pins this). Your own fire passes through your station; ships bounce off any
station, and ramming an enemy one hurts only the ship.

The trained agents were trained on sudden death but fight just as well in deathmatch. They can't
see stations or waves, so picking siege or survival swaps AI ships to the objective-aware scripted
bot; you can still pick a trained agent.

In game: `W`/`Up` thrust, `A`/`D` or arrows to turn, `Space` to shoot, `Esc` to pause (resume,
restart, change setup, main menu), `Tab` for the sensor view.

Only models whose observation size matches the current environment are offered, so older
checkpoints never crash the app. Names and ratings come from `manifest.json` files next to the
models (see `models/phaseA/`). Settings live in `~/.config/marl-asteroids/app.json`.

On WSL, if the window does not appear after connecting or disconnecting a monitor, run
`wsl --shutdown` from Windows and reopen the terminal: WSLg can keep a stale monitor layout and
place windows off screen.

## Command line

Run from the repo root. The scripts are still the way to train and evaluate, and `watch.py` offers
recording and a few options the app doesn't. It takes either a named scenario (`--scenario`, the
fixed set the evaluations use) or a mode and arena like the app:
`--mode siege --arena small --blue-ships 2 --red-ships 2`.

```bash
# Train
uv run python scripts/train.py --preset baseline --run baseline_v2 --timesteps 8e6
uv run python scripts/train.py --preset selfplay --run selfplay_v2 --timesteps 20e6
uv run tensorboard --logdir runs/tensorboard

# Evaluate (mirrored: every seed played with the teams swapped)
uv run python scripts/evaluate.py --blue models/selfplay_v2/final_model.zip \
    --red heuristic_0.25 heuristic_0.5 heuristic_0.75 heuristic_1.0 --games 500 --out results/sp.csv
uv run python scripts/league.py --agents heuristic_0.5 heuristic_1.0 \
    models/baseline_v2/best_model.zip models/selfplay_v2/final_model.zip

# Watch, play, record
uv run python scripts/watch.py --scenario demo_team2 --blue models/selfplay_v2/final_model.zip
uv run python scripts/watch.py --scenario demo_duel --blue human --red models/selfplay_v2/final_model.zip \
    --difficulty easy
uv run python scripts/watch.py --scenario demo_team3 --games 3 --record demo.mp4
```

Human controls: `W`/`Up` thrust, `A`/`D` or arrows to turn, `Space` to shoot, `Esc` to quit.

### Follow camera

By default the whole map is scaled into the window, which leaves ships only a few pixels wide on the
large arenas. `--follow` instead centers one ship and shows a local viewport:

```bash
uv run python scripts/watch.py --scenario team3_large --blue human,heuristic_1.0,heuristic_1.0 \
    --red heuristic_0.75 --difficulty easy --view 1300
```

- `--follow auto` (default) tracks your ship when you are playing, otherwise shows the whole map.
  Pass an agent name to track a specific ship, or `none` to force the whole-map view.
- `--view` sets how many world units span the viewport (clamped to the map size).
- Ships outside the viewport appear as team-colored chevrons on the screen edge labeled with their
  distance, and a minimap inset shows the whole map plus the current viewport box (`--no-minimap`
  turns it off).
- The projection wraps across the torus seam, so an enemy just past the map edge is drawn adjacent
  rather than a whole map away. The starfield scrolls and tiles with the camera.

### Seeing what the agent sees

`--show-obs` (or `Tab` while playing) overlays the observation vector decoded back into world space:
the sensor ring, a marker per occupied observation slot with its relative-velocity vector, slot
ranks, and a panel of the ego features in the units the policy receives. Markers land on the real
objects, which is how you can confirm the decode is faithful.

This doubles as a debugging tool and as the basis for human/agent parity: the display is built from
the same numbers the policy gets. It costs nothing during training, since rendering never runs when
`render_mode` is `None`.

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
  `action_repeat` = 2 physics frames. Episodes end when a team is eliminated, or at `max_frames`
  where the healthier team wins the timeout. `max_frames=None` means no clock at all, which is what
  the demo scenarios use so a human game never ends on a timer.
- Ships: speed capped at `max_speed`, bullets carry the ship's velocity, `fire_rate`-frame reload.
  Friendly fire is on.
- **Durability**: 3 hull + 1 shield. Shield absorbs first and regenerates after
  `shield_regen_delay` frames without being hit, so disengaging to recover is a real option; hull
  damage is permanent. Bullets do 1, asteroid impacts 2, ramming 1. Collisions bounce the bodies
  apart and a cooldown gates repeat collision damage, so contact is costly but survivable.
  This is what makes focused fire worth learning: with one-hit kills, whoever shoots first takes
  the kill alone and there is no reason to coordinate.
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
- Reward (`RewardConfig`): kill +1, death -1, team kill -1 (shooter), plus small per-point damage
  terms (+0.15 dealt, -0.15 taken) so the signal does not go sparse now that a kill takes four hits.
  Finishing a target always beats chipping several. Mixed across the team by `team_spirit`; shared
  win/loss at the end, also delivered to teammates who already died; plus potential-based shaping
  (aim, range, hazard), which cannot change the optimal policy.

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

## Results (Phase A: deathmatch with health)

Two agents trained on the current environment (221-d observation, health and shields):

- `baseline_v2`: 8M steps of PPO against scripted bots only (47 min).
- `selfplay_v2`: 20M steps of league self-play (95 min).

**Against the full-strength heuristic** (200 mirrored games each, `results/*_vs_heuristic.csv`):

| Scenario | baseline_v2 | selfplay_v2 (final) | heuristic vs itself |
|---|---|---|---|
| duel | 100% | 93% | 27% win / 47% draw |
| duel_rocks | 98.5% | 92.5% | 43% |
| team2 | 98.5% | 96.5% | 46% |
| team3 | 99.5% | 99% | 47.5% |

**League ratings** (`scripts/league.py`, every pair 100 mirrored games per scenario, Elo scale
anchored at heuristic_1.0 = 1000, `results/league_phaseA.csv`). Differences under roughly 50 are
within noise:

| Agent | duel | team2 | team3 | overall |
|---|---|---|---|---|
| selfplay_v2 final | 1610 | 1847 | 1812 | **1718** |
| selfplay_v2 best | 1597 | 1807 | 1763 | 1685 |
| self-play snapshot @ 10M | 1594 | 1792 | 1778 | 1684 |
| self-play snapshot @ 5M | 1578 | 1754 | 1691 | 1639 |
| baseline_v2 | 1435 | 1720 | 1724 | 1588 |
| self-play snapshot @ 2.5M | 1480 | 1468 | 1415 | 1440 |
| self-play snapshot @ 1M | 1075 | 944 | 1052 | 1026 |
| heuristic_1.0 | 1000 | 1000 | 1000 | 1000 |
| heuristic_0.5 | 917 | 863 | 851 | 880 |
| self-play snapshot @ 250k | 811 | 819 | 935 | 859 |

What this shows:

- **Self-play is stronger, even though it scores lower against the bot.** baseline_v2 wins more
  often against the heuristic, but selfplay_v2 beats baseline_v2 head to head (about 70% in duels,
  65-70% in 2v2). Training only against a fixed bot partly learns to exploit that bot.
- **`best_model.zip` is the wrong checkpoint for self-play runs.** It is chosen by win rate against
  the heuristic, which saturates near 100%, and it loses to `final_model.zip` head to head. Use
  `final_model.zip` (or league ratings) to pick self-play checkpoints.
- **Self-play plateaued around 10M steps.** The 10M snapshot, best, and final are within noise of
  each other, so the second 10M steps added little.
- **The snapshots form a natural difficulty ladder** from below heuristic_0.5 up to the final
  agent, with a gap between 1M (≈1030) and 2.5M (≈1440).
- **No measurable teamwork yet.** The shared-kill rate (share of kills where 2+ teammates damaged
  the victim) is 0.44-0.46 in 2v2 and 0.56-0.58 in 3v3 for selfplay_v2, against 0.43/0.53 for
  baseline_v2 and 0.44/0.44 for the heuristic, which never coordinates. The agents win on
  individual skill. This matches the design argument that deathmatch does not force cooperation.

Checkpoints are in `models/phaseA/` (see its README); full run directories are git-ignored.
Clips in `results/clips/`. Note that static demo arenas with deterministic policies replay the
identical game every time.

## Changes from the original project

Bugs fixed: the ally observation slot contained the ego ship itself; ship speed was uncapped
(~14.6 px/frame against a 3.0 normalization, so observations left their bounds); friendly-fire kills
paid the kill reward; the asteroid penalty was double-counted while bullets were never penalized;
entities ignored per-scenario config; seeds were ignored; `EvalCallback` ran on the training env and
selected "best" by shaped reward instead of win rate. Later, in this round: asteroid fragments from a
stationary rock never separated, and a static arena at `inset=0.25` was silently asymmetric because
the teams sat exactly antipodally on the torus.
