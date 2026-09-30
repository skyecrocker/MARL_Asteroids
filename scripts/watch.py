"""Watch (or play) games in a pygame window.

Examples:
    uv run python scripts/watch.py --scenario demo_team2 --blue models/baseline_v1/best_model.zip
    uv run python scripts/watch.py --scenario demo_duel --blue human --red models/baseline_v1/best_model.zip \
        --difficulty easy
    uv run python scripts/watch.py --scenario demo_team3 --blue a.zip --red b.zip --games 5 --record out.mp4
"""
from __future__ import annotations

import argparse

import numpy as np
import pygame

from marl_asteroids.config import SCENARIOS, get_scenario, set_team_fire_rate
from marl_asteroids.env import TeamAsteroidsEnv
from marl_asteroids.policies import DelayedPolicy, resolve_policy

# Handicaps for the AI side when a human is playing: (reaction delay in decisions, reload frames).
# Reload frames dominate: they cap the AI's damage output without breaking its movement or aim.
DIFFICULTY = {
    "easy": (6, 110),
    "normal": (3, 70),
    "hard": (1, 45),
    "brutal": (0, None),  # unhandicapped
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="demo_team2", choices=sorted(SCENARIOS))
    ap.add_argument("--blue", default="heuristic_1.0", help="scripted name, 'human', or checkpoint path")
    ap.add_argument("--red", default="heuristic_1.0")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--no-effects", action="store_true", help="Disable particles/trails/flames")
    ap.add_argument("--difficulty", choices=sorted(DIFFICULTY), default=None,
                    help="Handicap the non-human side (defaults to 'normal' when one side is human)")
    ap.add_argument("--record", default=None, help="Save to a video/gif (needs imageio)")
    args = ap.parse_args()

    human_teams = [t for t in ("blue", "red") if getattr(args, t) == "human"]
    difficulty = args.difficulty or ("normal" if human_teams else "brutal")
    delay, fire_rate = DIFFICULTY[difficulty]

    cfg = get_scenario(args.scenario)
    ai_teams = [t for t in ("blue", "red") if t not in human_teams]
    if human_teams and fire_rate is not None:
        for t in ai_teams:
            cfg = set_team_fire_rate(cfg, t, fire_rate)

    policies = {}
    for team in ("blue", "red"):
        pol = resolve_policy(getattr(args, team), deterministic=not args.stochastic)
        if human_teams and team in ai_teams and delay > 0:
            pol = DelayedPolicy(pol, delay)
        policies[team] = pol
    if human_teams:
        print(f"Difficulty {difficulty}: AI reaction delay {delay} decisions, reload {fire_rate} frames")
        print("Controls: W/Up thrust, A/D or Left/Right turn, Space shoot")

    env = TeamAsteroidsEnv(cfg, render_mode="rgb_array" if args.record else "human",
                           window=args.window, effects=not args.no_effects)
    frames = []
    try:
        for game in range(args.games):
            env.reset(seed=None if args.seed is None else args.seed + game)
            running = True
            while env.agents and running:
                if not args.record:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            running = False
                actions = {}
                for team, pol in policies.items():
                    ags = [a for a in env.agents if a.startswith(team)]
                    if ags:
                        acts = pol.act(np.stack([env.get_obs(a) for a in ags]), [(env, a) for a in ags])
                        actions.update(zip(ags, acts))
                env.step(actions)
                if args.record:
                    frames.append(env.render())
            if not running:
                break
            result = {1: "BLUE wins", 0: "RED wins", -1: "draw"}.get(env.winner, "?")
            print(f"Game {game + 1}: {result} after {env.frame} frames")
            if not args.record:  # hold the end-of-game screen briefly
                for _ in range(int(args.fps * 1.2)):
                    env.render()
    finally:
        env.close()
    if args.record and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.record, frames[::2], fps=max(1, args.fps // 2))
        print(f"Saved {args.record} ({len(frames[::2])} frames)")


if __name__ == "__main__":
    main()
