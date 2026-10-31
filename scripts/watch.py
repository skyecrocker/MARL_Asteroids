"""Watch (or play) games in a pygame window from the command line.

The desktop app (``uv run marl-asteroids``) is the friendlier way to do this; both build matches
through ``marl_asteroids.session``.

Examples:
    uv run python scripts/watch.py --scenario demo_team2 --blue models/phaseA/selfplay_final.zip
    uv run python scripts/watch.py --scenario demo_duel --blue human \
        --red models/phaseA/selfplay_final.zip --difficulty easy
    uv run python scripts/watch.py --scenario demo_team3 --blue a.zip --red b.zip --games 5 --record out.mp4
"""
from __future__ import annotations

import argparse
import time

import pygame

from marl_asteroids.config import SCENARIOS
from marl_asteroids.session import DIFFICULTY, MatchSpec, agent_names, build_config, build_match


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
    ap.add_argument("--show-obs", action="store_true",
                    help="Overlay the observation the agent actually receives (Tab toggles live)")
    ap.add_argument("--obs-agent", default=None,
                    help="Whose observation to overlay (defaults to the followed ship)")
    ap.add_argument("--no-effects", action="store_true", help="Disable particles/trails/flames")
    ap.add_argument("--difficulty", choices=sorted(DIFFICULTY), default=None,
                    help="Handicap the non-human side (defaults to 'normal' when one side is human)")
    ap.add_argument("--record", default=None, help="Save to a video/gif (needs imageio)")
    args = ap.parse_args()

    spec = MatchSpec(scenario=args.scenario, scale=args.scale, asteroid_density=args.asteroid_density,
                     difficulty=args.difficulty, follow=args.follow, view=args.view,
                     stochastic=args.stochastic, seed=args.seed)
    # Per-ship controllers: a single value applies to the whole team; a comma-separated list
    # assigns per ship (cycled), e.g. --blue human,heuristic_1.0 puts you in blue_0 with a wingman.
    for agent in agent_names(build_config(spec)):
        team, idx = agent.split("_")
        options = [s.strip() for s in getattr(args, team).split(",")]
        spec.controllers[agent] = options[int(idx) % len(options)]
    try:
        match = build_match(spec)
    except ValueError as e:
        ap.error(str(e))
    cfg = match.config

    if args.scale != 1.0:
        print(f"Arena {cfg.width:.0f}x{cfg.height:.0f}, {cfg.num_asteroids} asteroids, "
              f"max_frames {cfg.max_frames}")
    if match.human_agents:
        print(f"You are flying: {', '.join(match.human_agents)}")
        print(f"Difficulty {match.difficulty}: AI reaction delay {match.delay} decisions, "
              f"reload {match.fire_rate} frames")
        print("Controls: W/Up thrust, A/D or Left/Right turn, Space shoot, Esc quit")
    for agent, controller in sorted(match.assignment.items()):
        print(f"  {agent:8s} <- {controller}")
    if match.follow:
        print(f"Camera follows {match.follow} ({args.view:.0f} world units across)")

    env = match.make_env(render_mode="rgb_array" if args.record else "human",
                         window=args.window, effects=not args.no_effects,
                         minimap=not args.no_minimap, show_obs=args.show_obs, obs_agent=args.obs_agent)
    env.metadata = dict(env.metadata, render_fps=args.fps)
    frames = []
    try:
        for game in range(args.games):
            match.reset(game)
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
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB \
                                and env.renderer is not None:
                            env.renderer.show_obs = not env.renderer.show_obs
                match.step()
                if args.record:
                    frames.append(env.render())
            if not running:
                break
            result = {1: "BLUE wins", 0: "RED wins", -1: "draw"}.get(env.winner, "?")
            print(f"Game {game + 1}: {result} after {env.frame} frames")
            if args.record:  # hold the result banner for ~1s in the clip
                final = env.render()
                frames.extend([final] * max(1, round(args.fps / cfg.action_repeat)))
            else:  # hold the end-of-game screen for ~1.2s of wall time, independent of --fps
                hold_until = time.monotonic() + 1.2
                while time.monotonic() < hold_until:
                    env.render()
    finally:
        match.close()
    if args.record and frames:
        import imageio.v2 as imageio
        # One frame per env.step, which covers ``action_repeat`` physics frames, so real-time
        # playback is fps / action_repeat.
        out_fps = max(1, round(args.fps / cfg.action_repeat))
        imageio.mimsave(args.record, frames, fps=out_fps)
        print(f"Saved {args.record} ({len(frames)} frames at {out_fps} fps, "
              f"{len(frames) / out_fps:.1f}s)")


if __name__ == "__main__":
    main()
