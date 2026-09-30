from dataclasses import dataclass, field
import math
import random
from typing import List

GAME_WIDTH = 800.0
GAME_HEIGHT = 800.0

@dataclass
class EntityState:
    """Blueprint for spawning a specific entity."""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0 # Radians (used for ships)
    x_noise: float = 0.0
    y_noise: float = 0.0
    vx_noise: float = 0.0
    vy_noise: float = 0.0
    angle_noise: float = 0.0 # Noise stddev for each parameter
    fire_rate: int = 30
    size: int = 3      # Size (used for asteroids)
    random_spawn: bool = False


@ dataclass
class GameConfig:
    # Environment Setup
    width: float = GAME_WIDTH
    height: float = GAME_HEIGHT

    # Scenarios
    n_red: int = 2
    m_blue: int = 2
    num_asteroids: int = 5
    asteroids_split: bool = False
    friendly_fire: bool = True

    # If these lists are provided, the env uses them. If empty, it randomizes.
    red_spawns: List[EntityState] = field(default_factory=list)
    blue_spawns: List[EntityState] = field(default_factory=list)
    asteroid_spawns: List[EntityState] = field(default_factory=list)

    # Ships
    ship_radius: float = 10.0
    max_speed: float = 3.0
    turn_speed: float = 0.12
    thrust_power: float = 0.15
    bullet_cooldown: int = 30

    drag_coefficient: float = 0.99

    # Bullets
    bullet_radius: float = 2.0
    bullet_speed: float = 4.0
    bullet_lifespan: int = 80

    # Asteroids
    asteroid_large_radius: float = 30.0
    asteroid_medium_radius: float = 15.0
    asteroid_small_radius: float = 7.0

SCENARIO_DUEL = GameConfig(
    n_red=2,
    m_blue=2,
    num_asteroids=1,
)

# Stage 1: Target Practice
'''Learn to defeat a single stationary enemy with no asteroids'''
SCENARIO_CURRICULUM_LEVEL1 = GameConfig(
    n_red=1, 
    m_blue=1, 
    num_asteroids=0,
    red_spawns=[EntityState(fire_rate=99999, random_spawn=True)],
    blue_spawns=[] 
)

# Stage 2: 
'''Learn to defeat '''
SCENARIO_CURRICULUM_LEVEL2 = GameConfig(
    n_red=1, 
    m_blue=1, 
    num_asteroids=2,
    red_spawns=[EntityState(fire_rate=180, random_spawn=True)],
    blue_spawns=[] 
)

# Stage 3: 
'''Learn to defeat a single strong enemy bot with asteroids'''
SCENARIO_CURRICULUM_LEVEL3 = GameConfig(
    n_red=1, 
    m_blue=1, 
    num_asteroids=2,
    red_spawns=[EntityState(fire_rate=30, random_spawn=True)],
    blue_spawns=[] 
)

# Stage 4: 
'''Learn to defeat a single strong enemy bot with asteroids'''
SCENARIO_CURRICULUM_LEVEL4 = GameConfig(
    n_red=2, 
    m_blue=2, 
    num_asteroids=2,
    red_spawns=[EntityState(fire_rate=30, random_spawn=True), EntityState(fire_rate=30, random_spawn=True)],
    blue_spawns=[] 
)


# Evaluation Scenarios
SCENARIO_EVAL_LEVEL1 = GameConfig(n_red=1, m_blue=1, num_asteroids=0, red_spawns=[], blue_spawns=[])
SCENARIO_EVAL_LEVEL2 = GameConfig(n_red=1, m_blue=1, num_asteroids=2, red_spawns=[], blue_spawns=[])
SCENARIO_EVAL_LEVEL3 = GameConfig(n_red=2, m_blue=2, num_asteroids=2, red_spawns=[], blue_spawns=[])
SCENARIO_EVAL_LEVEL4 = GameConfig(n_red=3, m_blue=3, num_asteroids=4, red_spawns=[], blue_spawns=[])

# Fair Duel 
'''Learn to defeat a single strong enemy bot with asteroids'''
SCENARIO_DUEL = GameConfig(
    n_red=1, 
    m_blue=1, 
    num_asteroids=0,
    red_spawns=[],
    blue_spawns=[] 
)

# S
'''Stationary enemy is hidden behind asteroids, so the agent must learn to navigate around them to get a clean shot'''
SCENARIO_HIDDEN_TARGET = GameConfig(
    n_red=1, m_blue=1, num_asteroids=2,
    red_spawns=[EntityState(x=int(GAME_WIDTH/3), y=int(GAME_HEIGHT/2), angle=0)],
    blue_spawns=[EntityState(x=int(2*GAME_WIDTH/3), y=int(GAME_HEIGHT/2), angle=math.pi/2, angle_noise=0.10)],
    asteroid_spawns=[
        EntityState(x=int(GAME_WIDTH/2), y=int(GAME_HEIGHT/2), vx=0.0, vy=0.0),
        EntityState(x=int(GAME_WIDTH/4), y=int(GAME_HEIGHT/2), vx=0.0, vy=0.0),
    ]
)