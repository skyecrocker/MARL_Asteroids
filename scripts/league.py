"""Round-robin league with Bradley-Terry (Elo-scale) ratings.

Every pair of agents plays ``--games`` games per scenario, mirrored (each seed played with the
colors swapped) whenever team sizes are equal. Ratings are fit
by maximum likelihood on win=1 / draw=0.5 / loss=0 scores and reported on the Elo scale, anchored so
that ``--anchor`` (default heuristic_1.0) = 1000.

Example:
    uv run python scripts/league.py --agents dummy heuristic_0.5 heuristic_1.0 \
        models/baseline_v1/best_model.zip models/sp_v1/best_model.zip --scenarios duel team2 team3
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch as th

from marl_asteroids.config import get_scenario
from marl_asteroids.matches import play_matches, play_matches_mirrored
from marl_asteroids.policies import resolve_policy


def label(name: str) -> str:
    """Short unambiguous name: scripted names as-is, checkpoints as '<run>/<file>'."""
    return "/".join(name.split("/")[-2:]) if name.endswith(".zip") else name


def fit_bradley_terry(n: int, results, iters: int = 2000, lr: float = 0.5, prior: float = 1e-3):
    """results: list of (i, j, score_i, games). Returns log-strengths (natural units)."""
    theta = np.zeros(n)
    for _ in range(iters):
        grad = -prior * theta
        for i, j, s, g in results:
            p = 1.0 / (1.0 + np.exp(theta[j] - theta[i]))
            grad[i] += g * (s - p)
            grad[j] -= g * (s - p)
        theta += lr * grad / max(1, sum(r[3] for r in results)) * n
    return theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["duel", "team2", "team3"])
    ap.add_argument("--games", type=int, default=200, help="Games per pair per scenario")
    ap.add_argument("--anchor", default="heuristic_1.0")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="results/league.csv")
    args = ap.parse_args()
    th.set_num_threads(4)

    names = args.agents
    pols = [resolve_policy(a, seed=args.seed + k, deterministic=True) for k, a in enumerate(names)]
    per_scenario = {sc: [] for sc in args.scenarios}
    for sc in args.scenarios:
        cfg = get_scenario(sc)
        for i, j in itertools.combinations(range(len(names)), 2):
            if cfg.n_red == cfg.m_blue:
                # Same seeds with colors swapped, so spawn advantage cancels exactly
                stats = play_matches_mirrored(cfg, pols[i], pols[j], args.games, seed=args.seed)
            else:
                stats = play_matches(cfg, pols[i], pols[j], args.games, seed=args.seed)
            per_scenario[sc].append((i, j, stats.blue_score, stats.games))
            print(f"{sc:8s} {label(names[i])[:30]:30s} vs {label(names[j])[:30]:30s} "
                  f"score {stats.blue_score:.3f}", flush=True)

    elo = 400 / np.log(10)
    anchor = names.index(args.anchor) if args.anchor in names else 0
    table = {}
    for sc, res in list(per_scenario.items()) + [("overall", sum(per_scenario.values(), []))]:
        th_ = fit_bradley_terry(len(names), res)
        table[sc] = 1000 + elo * (th_ - th_[anchor])

    header = ["agent"] + list(table)
    print("\n" + f"{header[0]:>32s}  " + "  ".join(f"{h:>10s}" for h in header[1:]))
    order = np.argsort(-table["overall"])
    rows = []
    for k in order:
        row = [label(names[k])]
        vals = [f"{table[sc][k]:.0f}" for sc in table]
        print(f"{row[0][-32:]:>32s}  " + "  ".join(f"{v:>10s}" for v in vals))
        rows.append(dict(zip(header, row + vals)))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
