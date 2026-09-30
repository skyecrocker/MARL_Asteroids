"""Physical entities. All parameters come from the GameConfig instance, not class defaults."""
from __future__ import annotations

import numpy as np

from .config import GameConfig


class Entity:
    def __init__(self, x, y, vx, vy, radius):
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([vx, vy], dtype=np.float64)
        self.radius = float(radius)
        self.active = True  # False when destroyed


class Ship(Entity):
    def __init__(self, x, y, team_id: int, agent_name: str, config: GameConfig, fire_rate=None):
        super().__init__(x, y, 0.0, 0.0, config.ship_radius)
        self.team_id = team_id  # 0 red, 1 blue
        self.agent_name = agent_name
        self.angle = 0.0  # Radians
        self.fire_rate = int(fire_rate if fire_rate is not None else config.fire_rate)
        self.cooldown = 0
        self.thrusting = False  # Last commanded thrust; used only for rendering the flame


class Asteroid(Entity):
    def __init__(self, x, y, vx, vy, config: GameConfig, size: int = 3):
        super().__init__(x, y, vx, vy, config.asteroid_radius(size))
        self.size = size


class Bullet(Entity):
    def __init__(self, x, y, vx, vy, owner_name: str, team_id: int, config: GameConfig):
        super().__init__(x, y, vx, vy, config.bullet_radius)
        self.owner_name = owner_name
        self.team_id = team_id
        self.lifespan = config.bullet_lifespan
