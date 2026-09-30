"""Game, reward, and scenario configuration."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    # Episode
    max_frames: int = 1200     # Truncation limit in physics frames
    action_repeat: int = 2     # Physics frames per agent decision

    # Observation slots (k-nearest of each type, sorted by distance)
    obs_enemies: int = 3
    obs_allies: int = 2
    obs_asteroids: int = 6
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
# Scenarios. Everything uses random spawns unless noted.
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, GameConfig] = {
    "duel": GameConfig(n_red=1, m_blue=1, num_asteroids=0),
    "duel_rocks": GameConfig(n_red=1, m_blue=1, num_asteroids=2),
    "team2": GameConfig(n_red=2, m_blue=2, num_asteroids=2),
    "team3": GameConfig(n_red=3, m_blue=3, num_asteroids=4),
    # 2v1 / 1v2 imbalance scenarios encourage cooperation (focus fire, covering).
    "team2_vs_1": GameConfig(n_red=1, m_blue=2, num_asteroids=2),
    "hidden_target": GameConfig(
        n_red=1, m_blue=1, num_asteroids=2,
        red_spawns=[EntityState(x=GAME_WIDTH / 3, y=GAME_HEIGHT / 2, angle=0)],
        blue_spawns=[EntityState(x=2 * GAME_WIDTH / 3, y=GAME_HEIGHT / 2,
                                 angle=math.pi / 2, angle_noise=0.10)],
        asteroid_spawns=[
            EntityState(x=GAME_WIDTH / 2, y=GAME_HEIGHT / 2),
            EntityState(x=GAME_WIDTH / 4, y=GAME_HEIGHT / 2),
        ],
    ),
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
