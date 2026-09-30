"""Watch (or play) games in a pygame window.

Examples:
    uv run python scripts/watch.py --scenario team2 --blue models/sp_v1/best_model.zip --red heuristic_1.0
    uv run python scripts/watch.py --scenario duel --blue human --red models/sp_v1/best_model.zip
    uv run python scripts/watch.py --scenario team3 --blue a.zip --red b.zip --games 5 --record out.mp4
"""
from __future__ import annotations

import argparse

import numpy as np
import pygame

from marl_asteroids.config import SCENARIOS, get_scenario
from marl_asteroids.env import TeamAsteroidsEnv
from marl_asteroids.policies import resolve_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="team2", choices=sorted(SCENARIOS))
    ap.add_argument("--blue", default="heuristic_1.0", help="scripted name, 'human', or checkpoint path")
    ap.add_argument("--red", default="heuristic_1.0")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--record", default=None, help="Save frames to a video/gif (needs imageio)")
    args = ap.parse_args()

    policies = {"blue": resolve_policy(args.blue, deterministic=not args.stochastic),
                "red": resolve_policy(args.red, deterministic=not args.stochastic)}
    env = TeamAsteroidsEnv(get_scenario(args.scenario), render_mode="rgb_array" if args.record else "human")
    env.metadata = dict(env.metadata, render_fps=args.fps)
    frames = []
    for game in range(args.games):
        env.reset(seed=None if args.seed is None else args.seed + game)
        running = True
        while env.agents and running:
            if not args.record:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
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
        print(f"Game {game + 1}: {result} after {env.frame} frames; kills: {env.kill_log}")
    env.close()
    if args.record and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.record, frames[::2], fps=args.fps // 2)
        print(f"Saved {args.record}")


if __name__ == "__main__":
    main()
