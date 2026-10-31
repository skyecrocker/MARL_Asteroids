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

        self.max_hull = int(config.max_hull)
        self.max_shield = int(config.max_shield)
        self.hull = self.max_hull
        self.shield = self.max_shield
        self.frames_since_damage = 10 ** 9  # Large so a fresh ship is never mid-regen
        self._regen_progress = 0
        self.collision_cooldown = 0  # Gates repeat collision damage while bodies overlap

        # Respawning modes
        self.respawn_timer = None    # Frames until this (destroyed) ship returns; None = not coming back
        self.protection = 0          # Frames of invulnerability left after spawning
        self.reserve = False         # Survival: not part of the current wave (neither alive nor dead)

    def respawn(self, x, y, angle, protection: int = 0):
        """Bring a destroyed (or reserve) ship back at full health."""
        self.pos[:] = (x, y)
        self.vel[:] = 0.0
        self.angle = angle
        self.hull, self.shield = self.max_hull, self.max_shield
        self.cooldown = 0
        self.collision_cooldown = 0
        self.frames_since_damage = 10 ** 9
        self._regen_progress = 0
        self.thrusting = False
        self.active = True
        self.reserve = False
        self.respawn_timer = None
        self.protection = int(protection)

    @property
    def hp(self) -> int:
        return self.hull + self.shield

    @property
    def max_hp(self) -> int:
        return self.max_hull + self.max_shield

    def apply_damage(self, amount: int) -> int:
        """Absorb ``amount`` damage, shield first. Returns the damage actually applied.

        Spawn protection absorbs everything (the bullet is still used up)."""
        if not self.active or amount <= 0 or self.protection > 0:
            return 0
        before = self.hp
        self.frames_since_damage = 0
        self._regen_progress = 0
        absorbed = min(self.shield, amount)
        self.shield -= absorbed
        self.hull -= amount - absorbed
        if self.hull <= 0:
            self.hull = 0
            self.active = False
        return before - self.hp

    def tick_shield(self, delay: int, frames_per_point: int):
        """Regenerate shield after enough frames without taking damage."""
        self.frames_since_damage += 1
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        if self.protection > 0:
            self.protection -= 1
        if self.shield >= self.max_shield or self.frames_since_damage < delay:
            return
        self._regen_progress += 1
        if self._regen_progress >= max(1, frames_per_point):
            self._regen_progress = 0
            self.shield += 1


class Station(Entity):
    """A team's base in station siege: stationary, solid, with a large hull and a shield that
    regenerates continuously at a fixed rate (even while under fire)."""

    def __init__(self, x, y, team_id: int, config: GameConfig):
        super().__init__(x, y, 0.0, 0.0, config.station_radius)
        self.team_id = team_id
        self.max_hull = int(config.station_hull)
        self.max_shield = int(config.station_shield)
        self.hull = self.max_hull
        self.shield = self.max_shield
        self.regen_per_frame = float(config.station_shield_regen) / 60.0
        self._regen = 0.0

    @property
    def hp(self) -> int:
        return self.hull + self.shield

    @property
    def max_hp(self) -> int:
        return self.max_hull + self.max_shield

    @property
    def health_fraction(self) -> float:
        return self.hp / max(1, self.max_hp)

    def apply_damage(self, amount: int) -> int:
        if not self.active or amount <= 0:
            return 0
        before = self.hp
        absorbed = min(self.shield, amount)
        self.shield -= absorbed
        self.hull -= amount - absorbed
        if self.hull <= 0:
            self.hull = 0
            self.shield = 0
            self.active = False
        return before - self.hp

    def tick(self):
        if not self.active:
            return
        if self.shield >= self.max_shield:
            self._regen = 0.0  # no banking regen while full
            return
        self._regen += self.regen_per_frame
        # Tolerance: repeated float additions undershoot (10 x 0.1 = 0.9999...), which would
        # delay each point by a frame.
        while self._regen >= 1.0 - 1e-9 and self.shield < self.max_shield:
            self._regen = max(0.0, self._regen - 1.0)
            self.shield += 1


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
