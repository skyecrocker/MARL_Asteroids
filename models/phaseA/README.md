# Phase A checkpoints

Trained on the 221-d observation (deathmatch with health and shields). Copied out of the full run
directories, which are git-ignored.

| File | Source | League Elo (overall) |
|---|---|---|
| `selfplay_final.zip` | `selfplay_v2/final_model.zip`, 20M steps self-play | 1718 |
| `ladder_5M.zip` | self-play snapshot at 5M steps | 1639 |
| `baseline.zip` | `baseline_v2/best_model.zip`, 8M steps vs scripted bots | 1588 |
| `ladder_2.5M.zip` | self-play snapshot at 2.5M steps | 1440 |
| `ladder_1M.zip` | self-play snapshot at 1M steps | 1026 |

Elo is anchored at `heuristic_1.0` = 1000; see `results/league_phaseA.csv`. The ladder snapshots
are candidate easy/medium opponents for human play.
