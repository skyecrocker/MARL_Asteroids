"""Game, reward, and scenario configuration."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

GAME_WIDTH = 800.0
GAME_HEIGHT = 800.0


@dataclass
class EntityState:
    """Blueprint for spawning a specific entity (ship or asteroid).

    Positions are sampled from N(x, x_noise) etc. If ``random_spawn`` is True the
    position/heading are uniform over the map instead.
    """
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0  # Radians (ships)
    x_noise: float = 0.0
    y_noise: float = 0.0
    vx_noise: float = 0.0
    vy_noise: float = 0.0
    angle_noise: float = 0.0
    fire_rate: Optional[int] = None  # Frames between shots; None -> GameConfig.fire_rate
    size: int = 3  # Asteroid size (3 large, 2 medium, 1 small)
    random_spawn: bool = False


@dataclass
class RewardConfig:
    """Reward terms. Sparse terms are the objective; shaping is potential-based.

    Per-agent individual reward ``ind_i`` = kill/teamkill/death events + shaping.
    Final reward: ``r_i = (1 - team_spirit) * ind_i + team_spirit * mean_{j in team}(ind_j)``
    plus the shared team outcome (win/loss/draw) at episode end, which is also
    delivered to teammates that already died.
    """
    kill: float = 1.0
    teamkill: float = -1.0  # Applied to the shooter when friendly fire kills an ally
    death: float = -1.0
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    team_spirit: float = 0.5

    # Potential-based shaping F = gamma * phi(s') - phi(s); preserves the optimal policy.
    shaping_scale: float = 0.1
    shaping_gamma: float = 0.995  # Should match the learner's discount
    aim_weight: float = 1.0       # Potential for pointing the lead-aim at the nearest enemy
    approach_weight: float = 0.3  # Potential for being within firing range of an enemy
    hazard_weight: float = 1.0    # Negative potential for imminent collisions
    engage_range: float = 250.0   # px; full approach credit inside this distance
    engage_falloff: float = 400.0 # px; approach credit decays to 0 this far beyond engage_range


@dataclass
class GameConfig:
    # Map
    width: float = GAME_WIDTH
    height: float = GAME_HEIGHT

    # Teams / scenario
    n_red: int = 2
    m_blue: int = 2
    num_asteroids: int = 5
    asteroids_split: bool = False
    friendly_fire: bool = True

    # Optional explicit spawn blueprints; missing entries spawn uniformly at random.
    red_spawns: List[EntityState] = field(default_factory=list)
    blue_spawns: List[EntityState] = field(default_factory=list)
    asteroid_spawns: List[EntityState] = field(default_factory=list)
    min_spawn_dist: float = 150.0      # Between ships
    asteroid_clear_radius: float = 120.0  # Asteroids never spawn this close to a ship
    asteroid_speed_range: tuple = (0.5, 1.5)

    # Ships
    ship_radius: float = 10.0
    max_speed: float = 4.0     # Hard speed cap (the original had no cap: terminal ~14.6)
    turn_speed: float = 0.12   # Radians per frame
    thrust_power: float = 0.15
    fire_rate: int = 30        # Frames between shots
    drag_coefficient: float = 0.99

    # Bullets
    bullet_radius: float = 2.0
    bullet_speed: float = 6.0  # Added to the ship's velocity
    bullet_lifespan: int = 80

    # Asteroids
    asteroid_large_radius: float = 30.0
    asteroid_medium_radius: float = 15.0
    asteroid_small_radius: float = 7.0
    # When asteroids_split is True, a hit rock breaks into two smaller, faster fragments
    # (size 1 fragments are destroyed outright). When False, rocks are indestructible cover.
    asteroid_split_speedup: float = 1.4
    asteroid_split_spread: float = math.pi / 4
    # Floor on fragment speed: a slow or stationary parent would otherwise produce fragments that
    # never separate and sit stacked on the parent's position.
    asteroid_split_min_speed: float = 0.7

    # Episode
    max_frames: int = 1200     # Truncation limit in physics frames
    action_repeat: int = 2     # Physics frames per agent decision

    # Observation length scale (px). Relative positions/distances are normalized by this and
    # clipped, NOT by the map size, so the world can be resized without changing what the policy
    # sees. Also the natural place to attach a real sensor range later.
    sensor_range: float = 500.0

    # Observation slots (k-nearest of each type, sorted by distance)
    obs_enemies: int = 3
    obs_allies: int = 2
    obs_asteroids: int = 8  # Splitting can put many fragments nearby
    obs_bullets: int = 6

    rewards: RewardConfig = field(default_factory=RewardConfig)

    def asteroid_radius(self, size: int) -> float:
        return {3: self.asteroid_large_radius,
                2: self.asteroid_medium_radius,
                1: self.asteroid_small_radius}[size]

    def replace(self, **changes) -> "GameConfig":
        """Return a copy with fields replaced (nested dataclasses are copied too)."""
        return dataclasses.replace(self, rewards=dataclasses.replace(self.rewards), **changes)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _line_formation(n: int, x: float, y_center: float, spacing: float, angle: float) -> List[EntityState]:
    """n ships in a vertical line at x, centered on y_center, all facing ``angle``."""
    y0 = y_center - spacing * (n - 1) / 2
    angle = (angle + math.pi) % (2 * math.pi) - math.pi  # normalize to [-pi, pi]
    return [EntityState(x=x, y=y0 + i * spacing, angle=angle) for i in range(n)]


def static_battle(n: int, width: float, height: float, asteroids: Sequence[tuple] = (),
                  spacing: float = 320.0, inset: float = 0.3, facing: float = math.pi / 2,
                  **kw) -> GameConfig:
    """Deterministic n-vs-n setup: red on the left, blue on the right, exactly 180-degree symmetric.

    A rotation by 180 degrees about the map center maps red onto blue, so neither side has a
    positional advantage: red faces ``facing`` and blue faces ``facing + pi``. The default
    perpendicular heading means both teams must turn to engage, which gives a human time to react
    instead of starting in the crosshairs.

    ``inset`` must be greater than 0.25: at exactly 0.25 the teams sit half a map apart, which is
    the antipodal case on a torus where "shortest path to the enemy" is a tie. Both sides then
    resolve it to the same world direction, which silently breaks the mirror symmetry.

    Fully reproducible (no spawn noise), for demos, videos, debugging, and human play. Prefer
    mirrored evaluation over static spawns for *measurement*: with deterministic policies a static
    scenario replays the same game every time.
    """
    return GameConfig(
        n_red=n, m_blue=n, width=width, height=height,
        num_asteroids=len(asteroids),
        red_spawns=_line_formation(n, width * inset, height / 2, spacing, facing),
        blue_spawns=_line_formation(n, width * (1 - inset), height / 2, spacing, facing + math.pi),
        asteroid_spawns=[EntityState(x=x, y=y, vx=vx, vy=vy, size=size)
                         for x, y, vx, vy, size in asteroids],
        **kw,
    )


# Demo maps are larger (more room to react) with proportionally larger ships so everything stays
# readable on screen. Difficulty for human play comes mainly from the AI handicaps in
# set_team_fire_rate / policies.DelayedPolicy, not from map size alone.
DEMO_W = DEMO_H = 1200.0
DEMO_SCALE = dict(ship_radius=15.0, bullet_radius=3.5, max_speed=4.5,
                  asteroid_large_radius=38.0, asteroid_medium_radius=20.0, asteroid_small_radius=10.0,
                  max_frames=2000, min_spawn_dist=220.0)
_DEMO_ROCKS = [
    (DEMO_W * 0.5, DEMO_H * 0.25, 0.0, 0.35, 3),
    (DEMO_W * 0.5, DEMO_H * 0.75, 0.0, -0.35, 3),
    (DEMO_W * 0.32, DEMO_H * 0.5, 0.3, 0.0, 2),
    (DEMO_W * 0.68, DEMO_H * 0.5, -0.3, 0.0, 2),
]

SCENARIOS: Dict[str, GameConfig] = {
    # --- Training / evaluation (randomized spawns) ---
    "duel": GameConfig(n_red=1, m_blue=1, num_asteroids=0),
    "duel_rocks": GameConfig(n_red=1, m_blue=1, num_asteroids=3, asteroids_split=True),
    "team2": GameConfig(n_red=2, m_blue=2, num_asteroids=4, asteroids_split=True),
    "team3": GameConfig(n_red=3, m_blue=3, num_asteroids=6, asteroids_split=True),
    # Imbalanced games encourage cooperation (focus fire, covering a weaker side).
    "team2_vs_1": GameConfig(n_red=1, m_blue=2, num_asteroids=4, asteroids_split=True),
    # Dense field: asteroid dodging and splitting dominate.
    "asteroid_storm": GameConfig(n_red=2, m_blue=2, num_asteroids=10, asteroids_split=True,
                                 asteroid_speed_range=(0.8, 2.0)),
    # Indestructible cover: rocks block shots instead of breaking apart.
    "obstacles": GameConfig(n_red=2, m_blue=2, num_asteroids=5, asteroids_split=False),

    # --- Static, symmetric, human-scale (demos, videos, debugging) ---
    "demo_duel": static_battle(1, DEMO_W, DEMO_H, _DEMO_ROCKS[:2], asteroids_split=True, **DEMO_SCALE),
    "demo_team2": static_battle(2, DEMO_W, DEMO_H, _DEMO_ROCKS, spacing=300.0,
                               asteroids_split=True, **DEMO_SCALE),
    "demo_team3": static_battle(3, DEMO_W, DEMO_H, _DEMO_ROCKS, spacing=290.0,
                               asteroids_split=True, **DEMO_SCALE),
    "demo_open": static_battle(1, DEMO_W, DEMO_H, **DEMO_SCALE),
}

# Evaluation suite mirroring the legacy report's levels 1-4.
EVAL_SCENARIOS: Dict[str, str] = {
    "L1 1v1 open": "duel",
    "L2 1v1 rocks": "duel_rocks",
    "L3 2v2 rocks": "team2",
    "L4 3v3 rocks": "team3",
}


def get_scenario(name: str, **overrides) -> GameConfig:
    """Fetch a scenario by name, optionally overriding fields."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}. Available: {sorted(SCENARIOS)}")
    return SCENARIOS[name].replace(**overrides)


def set_team_fire_rate(config: GameConfig, team: str, fire_rate: int) -> GameConfig:
    """Return a copy where every ship on ``team`` ('red'/'blue') reloads in ``fire_rate`` frames.

    A slower trigger is the most effective way to handicap an AI opponent for human play: it keeps
    the agent's movement and aim intact but caps its damage output. Requires no retraining.
    """
    cfg = config.replace()
    n = cfg.n_red if team == "red" else cfg.m_blue
    spawns = list(cfg.red_spawns if team == "red" else cfg.blue_spawns)
    while len(spawns) < n:
        spawns.append(EntityState(random_spawn=True))
    spawns = [dataclasses.replace(bp, fire_rate=fire_rate) for bp in spawns]
    if team == "red":
        cfg.red_spawns = spawns
    else:
        cfg.blue_spawns = spawns
    return cfg
