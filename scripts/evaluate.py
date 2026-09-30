"""Evaluation matrix: blue agents x red agents x scenarios -> CSV.

Agents are scripted names (dummy, evasion, evasion_easy, heuristic_<p>) or checkpoint paths.

Examples:
    uv run python scripts/evaluate.py --blue models/sp_v1/best_model.zip \
        --red heuristic_0.25 heuristic_0.5 heuristic_0.75 heuristic_1.0 --games 500 \
        --out results/sp_v1_combat.csv
    uv run python scripts/evaluate.py --blue models/a.zip --red models/b.zip --scenarios duel team2
"""
from __future__ import annotations

import argparse
import csv
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch as th

from marl_asteroids.config import EVAL_SCENARIOS, SCENARIOS, get_scenario
from marl_asteroids.matches import play_matches
from marl_asteroids.policies import resolve_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", nargs="+", required=True)
    ap.add_argument("--red", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=list(EVAL_SCENARIOS.values()),
                    help=f"Scenario names: {sorted(SCENARIOS)}")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--stochastic", action="store_true", help="Sample network actions instead of argmax")
    ap.add_argument("--out", default=None, help="CSV path")
    args = ap.parse_args()
    th.set_num_threads(4)

    rows = []
    for scenario in args.scenarios:
        for b in args.blue:
            for r in args.red:
                t = time.time()
                blue = resolve_policy(b, seed=args.seed, deterministic=not args.stochastic)
                red = resolve_policy(r, seed=args.seed + 1, deterministic=not args.stochastic)
                stats = play_matches(get_scenario(scenario), blue, red, args.games,
                                     n_parallel=args.parallel, seed=args.seed)
                s = stats.summary()
                row = {"scenario": scenario, "blue": b, "red": r, **{k: round(v, 4) for k, v in s.items()}}
                rows.append(row)
                print(f"{scenario:12s} {os.path.basename(b)[:28]:28s} vs {os.path.basename(r)[:28]:28s} "
                      f"win {s['blue_win_rate']:.3f}  loss {s['red_win_rate']:.3f}  "
                      f"draw {s['draw_timeout_rate'] + s['draw_mutual_rate']:.3f}  "
                      f"tk/g {s['blue_teamkills_per_game']:.2f}  ({time.time() - t:.0f}s)", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
