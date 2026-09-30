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

from marl_asteroids.config import SCENARIOS, get_scenario, scale_map, set_team_fire_rate
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
    ap.add_argument("--blue", default="heuristic_1.0",
                    help="scripted name, 'human', or checkpoint path. Comma-separated assigns per "
                         "ship, e.g. 'human,heuristic_1.0' to fly with an AI wingman")
    ap.add_argument("--red", default="heuristic_1.0")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Grow the arena by this factor (ships/weapons/sensors unchanged)")
    ap.add_argument("--asteroid-density", type=float, default=1.0,
                    help="Asteroid count multiplier applied when --scale is used")
    ap.add_argument("--follow", default="auto",
                    help="Agent to keep centered ('auto' = your ship if playing, else whole map, "
                         "'none' forces whole map)")
    ap.add_argument("--view", type=float, default=1200.0,
                    help="World units across the viewport in follow mode")
    ap.add_argument("--no-minimap", action="store_true")
    ap.add_argument("--no-effects", action="store_true", help="Disable particles/trails/flames")
    ap.add_argument("--difficulty", choices=sorted(DIFFICULTY), default=None,
                    help="Handicap the non-human side (defaults to 'normal' when one side is human)")
    ap.add_argument("--record", default=None, help="Save to a video/gif (needs imageio)")
    args = ap.parse_args()

    cfg = get_scenario(args.scenario)
    if args.scale != 1.0:
        cfg = scale_map(cfg, args.scale, asteroid_density=args.asteroid_density)
        print(f"Arena {cfg.width:.0f}x{cfg.height:.0f}, {cfg.num_asteroids} asteroids, "
              f"max_frames {cfg.max_frames}")
    team_sizes = {"red": cfg.n_red, "blue": cfg.m_blue}
    # Per-agent specs: one spec applies to the whole team, a comma-separated list assigns per ship
    # (cycled), e.g. --blue human,heuristic_1.0 puts you in blue_0 with an AI wingman.
    specs = {t: [s.strip() for s in getattr(args, t).split(",")] for t in ("red", "blue")}
    assignment = {f"{t}_{i}": specs[t][i % len(specs[t])]
                  for t, n in team_sizes.items() for i in range(n)}
    human_agents = [a for a, s in assignment.items() if s == "human"]

    difficulty = args.difficulty or ("normal" if human_agents else "brutal")
    delay, fire_rate = DIFFICULTY[difficulty]
    human_teams = {a.split("_")[0] for a in human_agents}
    if human_agents and fire_rate is not None:
        for t in ("red", "blue"):
            if t not in human_teams:
                cfg = set_team_fire_rate(cfg, t, fire_rate)

    # One policy instance per distinct spec, so inference stays batched
    cache = {}
    for spec in set(assignment.values()):
        pol = resolve_policy(spec, deterministic=not args.stochastic)
        if human_agents and spec != "human" and delay > 0:
            pol = DelayedPolicy(pol, delay)
        cache[spec] = pol
    policies = {agent: cache[spec] for agent, spec in assignment.items()}

    if human_agents:
        print(f"You are flying: {', '.join(human_agents)}")
        print(f"Difficulty {difficulty}: AI reaction delay {delay} decisions, reload {fire_rate} frames")
        print("Controls: W/Up thrust, A/D or Left/Right turn, Space shoot, Esc quit")
    for agent, spec in sorted(assignment.items()):
        print(f"  {agent:8s} <- {spec}")

    if args.follow == "auto":
        follow = human_agents[0] if human_agents else None
    elif args.follow == "none":
        follow = None
    else:
        follow = args.follow
        if follow not in assignment:
            ap.error(f"--follow {follow!r} is not an agent in this scenario: {sorted(assignment)}")
    if follow:
        print(f"Camera follows {follow} ({args.view:.0f} world units across)")

    env = TeamAsteroidsEnv(cfg, render_mode="rgb_array" if args.record else "human",
                           window=args.window, effects=not args.no_effects,
                           follow=follow, view_size=args.view, minimap=not args.no_minimap)
    frames = []
    try:
        for game in range(args.games):
            env.reset(seed=None if args.seed is None else args.seed + game)
            first = env.render()  # initializes pygame before any policy polls the keyboard
            if args.record and first is not None:
                frames.append(first)
            running = True
            while env.agents and running:
                if not args.record:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            running = False
                groups = {}
                for a in env.agents:
                    pol = policies[a]
                    groups.setdefault(id(pol), (pol, []))[1].append(a)
                actions = {}
                for pol, ags in groups.values():
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
