"""Batched head-to-head match runner (evaluation, Elo, eval callbacks)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import GameConfig
from .env import BLUE, RED, TeamAsteroidsEnv
from .policies import Policy


@dataclass
class MatchStats:
    blue_wins: int = 0
    red_wins: int = 0
    draws_timeout: int = 0
    draws_mutual: int = 0
    lengths: List[int] = field(default_factory=list)  # physics frames
    kills: Dict[str, int] = field(default_factory=lambda: {"blue": 0, "red": 0})
    teamkills: Dict[str, int] = field(default_factory=lambda: {"blue": 0, "red": 0})
    env_deaths: Dict[str, int] = field(default_factory=lambda: {"blue": 0, "red": 0})  # asteroid/ram

    @property
    def games(self) -> int:
        return self.blue_wins + self.red_wins + self.draws_timeout + self.draws_mutual

    @property
    def blue_win_rate(self) -> float:
        return self.blue_wins / max(1, self.games)

    @property
    def blue_score(self) -> float:
        """Win = 1, draw = 0.5, loss = 0 (for Elo)."""
        return (self.blue_wins + 0.5 * (self.draws_timeout + self.draws_mutual)) / max(1, self.games)

    def summary(self) -> Dict[str, float]:
        n = max(1, self.games)
        return {
            "games": self.games,
            "blue_win_rate": self.blue_wins / n,
            "red_win_rate": self.red_wins / n,
            "draw_timeout_rate": self.draws_timeout / n,
            "draw_mutual_rate": self.draws_mutual / n,
            "avg_frames": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "blue_kills_per_game": self.kills["blue"] / n,
            "blue_teamkills_per_game": self.teamkills["blue"] / n,
            "blue_env_deaths_per_game": self.env_deaths["blue"] / n,
        }


def _team_str(agent: str) -> str:
    return agent.split("_")[0]


def record_game(env: TeamAsteroidsEnv, stats: MatchStats):
    if env.winner == BLUE:
        stats.blue_wins += 1
    elif env.winner == RED:
        stats.red_wins += 1
    elif env.frame >= env.config.max_frames:
        stats.draws_timeout += 1
    else:
        stats.draws_mutual += 1
    stats.lengths.append(env.frame)
    for shooter, victim in env.kill_log:
        if shooter in env.ships:
            key = "teamkills" if _team_str(shooter) == _team_str(victim) else "kills"
            getattr(stats, key)[_team_str(shooter)] += 1
        else:
            stats.env_deaths[_team_str(victim)] += 1


def play_matches_mirrored(config: GameConfig, a: Policy, b: Policy, n_games: int,
                          n_parallel: int = 16, seed: Optional[int] = 0) -> MatchStats:
    """Mirrored evaluation: every seed is played twice with the teams swapped.

    Cancels spawn/orientation advantage, which otherwise adds bias when one side happens to start in
    a better position. Returns stats from ``a``'s perspective (reported in the "blue" fields).
    Requires symmetric team sizes.
    """
    if config.n_red != config.m_blue:
        raise ValueError("mirrored evaluation requires symmetric teams (n_red == m_blue)")
    half = n_games // 2
    first = play_matches(config, a, b, half, n_parallel, seed)
    second = play_matches(config, b, a, n_games - half, n_parallel, seed)  # same seeds, swapped colors
    return _merge_swapped(first, second)


def _merge_swapped(first: MatchStats, second: MatchStats) -> MatchStats:
    """Combine a normal run with a color-swapped run, keeping 'blue' = the first run's blue policy."""
    m = MatchStats()
    m.blue_wins = first.blue_wins + second.red_wins
    m.red_wins = first.red_wins + second.blue_wins
    m.draws_timeout = first.draws_timeout + second.draws_timeout
    m.draws_mutual = first.draws_mutual + second.draws_mutual
    m.lengths = first.lengths + second.lengths
    for field in ("kills", "teamkills", "env_deaths"):
        fa, fb = getattr(first, field), getattr(second, field)
        getattr(m, field)["blue"] = fa["blue"] + fb["red"]
        getattr(m, field)["red"] = fa["red"] + fb["blue"]
    return m


def play_matches(config: GameConfig, blue: Policy, red: Policy, n_games: int,
                 n_parallel: int = 16, seed: Optional[int] = 0, render: bool = False) -> MatchStats:
    """Play ``n_games`` games, ``n_parallel`` at a time, with batched policy inference."""
    stats = MatchStats()
    n_parallel = max(1, min(n_parallel, n_games))
    envs = [TeamAsteroidsEnv(config, render_mode="human" if (render and k == 0) else None)
            for k in range(n_parallel)]
    started = 0
    active = []
    for k, env in enumerate(envs):
        env.reset(seed=None if seed is None else seed + started)
        started += 1
        active.append(env)

    while active:
        ctx = {"blue": [], "red": []}
        for env in active:
            for a in env.agents:
                ctx[_team_str(a)].append((env, a))
        actions = {}
        for team, policy in (("blue", blue), ("red", red)):
            if ctx[team]:
                obs = np.stack([env.get_obs(a) for env, a in ctx[team]])
                acts = policy.act(obs, ctx[team])
                for (env, a), act in zip(ctx[team], acts):
                    actions.setdefault(id(env), {})[a] = act
        still = []
        for env in active:
            env.step(actions.get(id(env), {}))
            if env.game_over:
                record_game(env, stats)
                if started < n_games:
                    env.reset(seed=None if seed is None else seed + started)
                    started += 1
                    still.append(env)
                else:
                    env.close()
            else:
                still.append(env)
        active = still
    return stats
