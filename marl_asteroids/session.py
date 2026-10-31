"""Building playable matches from a plain-data description.

``MatchSpec`` is JSON-friendly (the app saves it as "last setup"); ``build_match`` turns it into a
``Match`` holding the final config, one policy per ship, and the camera choice. Both the desktop app
and ``scripts/watch.py`` go through here, so they build games identically. The game loop itself
stays in each frontend, because they pace and draw differently.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config import (SURVIVAL_MAX_ENEMIES, GameConfig, get_scenario, make_game, scale_map,
                     set_team_fire_rate)
from .env import TeamAsteroidsEnv
from .matches import MatchStats, record_game
from .policies import DelayedPolicy, NetworkPolicy, Policy, resolve_policy

PHYSICS_FPS = 60  # physics frames per second at normal speed

# Handicaps for AI ships when a human plays: (reaction delay in decisions, reload frames).
# Reaction delay is the effective lever; reload mostly caps sustained damage (see README).
DIFFICULTY: Dict[str, Tuple[int, Optional[int]]] = {
    "easy": (6, 110),
    "normal": (3, 70),
    "hard": (1, 45),
    "brutal": (0, None),  # unhandicapped
}

HUMAN = "human"
KEEP = "scenario"  # sentinel: leave this setting as the scenario defines it


@dataclass
class MatchSpec:
    """Everything needed to set up a game. All fields are JSON-serializable.

    Two ways to describe the arena: set ``arena`` (mini/small/medium/large) and ``mode`` to build a
    game from scratch (what the app does), or leave ``arena`` as None to start from a named
    ``scenario`` (what the command-line scripts and evaluation suites use).
    """
    scenario: str = "demo_team2"
    mode: str = "deathmatch"             # see config.MODES; used when ``arena`` is set
    arena: Optional[str] = None          # see config.ARENA_SIZES
    fixed_spawns: bool = True            # deathmatch with ``arena``: line teams up vs. random
    # Controller per ship ("human", a scripted name such as "heuristic_1.0", or a checkpoint path).
    # Ships not listed use ``default_controller``.
    controllers: Dict[str, str] = field(default_factory=dict)
    default_controller: str = "heuristic_1.0"
    n_red: Optional[int] = None          # None = scenario default
    m_blue: Optional[int] = None
    scale: float = 1.0                   # arena size multiplier (see config.scale_map)
    asteroid_density: float = 1.0        # asteroid count multiplier applied with scale
    num_asteroids: Optional[int] = None  # explicit count, applied after scaling
    asteroids_split: Optional[bool] = None
    # Seconds, None for no time limit, or KEEP for the scenario's (or mode's) own limit
    time_limit: Union[float, None, str] = KEEP
    difficulty: Optional[str] = None     # None = "normal" if a human plays, else unhandicapped
    follow: str = "auto"                 # "auto" (your ship if playing), "none", or an agent name
    view: float = 1200.0                 # world units across the viewport when following
    stochastic: bool = False             # sample network actions instead of argmax
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchSpec":
        """Tolerant of missing and unknown keys, so old saved setups keep loading."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


def build_config(spec: MatchSpec) -> GameConfig:
    """Build the game config from the spec (handicaps are applied in ``build_match``)."""
    if spec.arena is not None:
        keep_time = spec.time_limit == KEEP
        return make_game(
            spec.mode, spec.arena,
            m_blue=2 if spec.m_blue is None else int(spec.m_blue),
            # Survival waves keep growing, so red is always the full enemy pool
            n_red=(SURVIVAL_MAX_ENEMIES if spec.mode == "survival"
                   else 2 if spec.n_red is None else int(spec.n_red)),
            num_asteroids=spec.num_asteroids,
            asteroids_split=True if spec.asteroids_split is None else bool(spec.asteroids_split),
            fixed_spawns=spec.fixed_spawns,
            time_limit=None if keep_time else spec.time_limit,
            use_mode_time_limit=keep_time)

    cfg = get_scenario(spec.scenario)
    # Team sizes first, so scale_map scales the trimmed spawn lists. Extra ships spawn randomly.
    if spec.n_red is not None:
        cfg.n_red = int(spec.n_red)
        cfg.red_spawns = list(cfg.red_spawns or [])[:cfg.n_red]
    if spec.m_blue is not None:
        cfg.m_blue = int(spec.m_blue)
        cfg.blue_spawns = list(cfg.blue_spawns or [])[:cfg.m_blue]
    if spec.scale != 1.0:
        cfg = scale_map(cfg, spec.scale, asteroid_density=spec.asteroid_density)
    if spec.num_asteroids is not None:
        cfg.num_asteroids = max(0, int(spec.num_asteroids))
        cfg.asteroid_spawns = list(cfg.asteroid_spawns or [])[:cfg.num_asteroids]
    if spec.asteroids_split is not None:
        cfg.asteroids_split = bool(spec.asteroids_split)
    if spec.time_limit != KEEP:
        cfg.max_frames = None if spec.time_limit is None else max(1, int(spec.time_limit * PHYSICS_FPS))
    if cfg.n_red < 1 or cfg.m_blue < 1:
        raise ValueError("each team needs at least one ship")
    return cfg


def agent_names(cfg: GameConfig) -> List[str]:
    return [f"blue_{i}" for i in range(cfg.m_blue)] + [f"red_{i}" for i in range(cfg.n_red)]


# Loaded networks are stateless at inference, so share them across matches (rematches and setup
# changes don't reload from disk). Keyed by path, file mtime, and determinism.
_NETWORK_CACHE: Dict[tuple, NetworkPolicy] = {}


def _load_policy(controller: str, deterministic: bool) -> Policy:
    if controller.endswith(".zip") or "/" in controller:
        key = (os.path.abspath(controller), os.path.getmtime(controller), deterministic)
        if key not in _NETWORK_CACHE:
            _NETWORK_CACHE[key] = NetworkPolicy.load(controller, name=controller,
                                                     deterministic=deterministic)
        return _NETWORK_CACHE[key]
    return resolve_policy(controller, deterministic=deterministic)


@dataclass
class Match:
    spec: MatchSpec
    config: GameConfig
    assignment: Dict[str, str]          # agent -> controller string
    policies: Dict[str, Policy]         # agent -> policy (instances shared per controller)
    human_agents: List[str]
    difficulty: str
    delay: int
    fire_rate: Optional[int]
    follow: Optional[str]
    env: Optional[TeamAsteroidsEnv] = None

    def make_env(self, render_mode: Optional[str] = None, **render_options) -> TeamAsteroidsEnv:
        opts = dict(follow=self.follow, view_size=self.spec.view)
        opts.update(render_options)
        self.env = TeamAsteroidsEnv(self.config, render_mode=render_mode, **opts)
        return self.env

    def reset(self, game_index: int = 0):
        seed = None if self.spec.seed is None else self.spec.seed + game_index
        self.env.reset(seed=seed)

    def actions(self) -> Dict[str, np.ndarray]:
        """One action per living ship, batched per policy instance."""
        env = self.env
        groups: Dict[int, tuple] = {}
        for a in env.agents:
            pol = self.policies[a]
            groups.setdefault(id(pol), (pol, []))[1].append(a)
        out = {}
        for pol, ags in groups.values():
            acts = pol.act(np.stack([env.get_obs(a) for a in ags]), [(env, a) for a in ags])
            out.update(zip(ags, acts))
        return out

    def step(self) -> bool:
        """Advance one decision (``action_repeat`` physics frames). Returns True when the game ended."""
        self.env.step(self.actions())
        return self.env.game_over

    def stats(self) -> MatchStats:
        stats = MatchStats()
        if self.env is not None and self.env.game_over:
            record_game(self.env, stats)
        return stats

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def build_match(spec: MatchSpec) -> Match:
    cfg = build_config(spec)
    agents = agent_names(cfg)
    unknown = sorted(set(spec.controllers) - set(agents))
    if unknown:
        raise ValueError(f"controllers given for ships not in this match: {unknown}")
    assignment = {a: spec.controllers.get(a, spec.default_controller) for a in agents}
    human_agents = [a for a, c in assignment.items() if c == HUMAN]
    if len(human_agents) > 1:
        raise ValueError(f"only one ship can be human-controlled (got {human_agents})")

    if spec.difficulty is not None and spec.difficulty not in DIFFICULTY:
        raise ValueError(f"unknown difficulty {spec.difficulty!r}; choose from {sorted(DIFFICULTY)}")
    difficulty = spec.difficulty or ("normal" if human_agents else "brutal")
    delay, fire_rate = DIFFICULTY[difficulty]
    if not human_agents:  # handicaps exist to make human play fair; AI-vs-AI stays unhandicapped
        delay, fire_rate = 0, None
    human_teams = {a.split("_")[0] for a in human_agents}
    if fire_rate is not None:
        for team in ("red", "blue"):
            if team not in human_teams:
                cfg = set_team_fire_rate(cfg, team, fire_rate)

    # One policy instance per distinct controller so inference stays batched. The reaction delay
    # applies to every AI ship, the human's wingmen included; the reload handicap (above) only to
    # teams without the human.
    instances: Dict[str, Policy] = {}
    for controller in set(assignment.values()):
        pol = _load_policy(controller, deterministic=not spec.stochastic)
        if controller != HUMAN and delay > 0:
            pol = DelayedPolicy(pol, delay)
        instances[controller] = pol
    policies = {a: instances[c] for a, c in assignment.items()}

    if spec.follow == "auto":
        follow = human_agents[0] if human_agents else None
    elif spec.follow == "none":
        follow = None
    else:
        if spec.follow not in assignment:
            raise ValueError(f"cannot follow {spec.follow!r}: not a ship in this match {agents}")
        follow = spec.follow

    return Match(spec=spec, config=cfg, assignment=assignment, policies=policies,
                 human_agents=human_agents, difficulty=difficulty, delay=delay,
                 fire_rate=fire_rate, follow=follow)
