"""Physical entities. All parameters come from the GameConfig instance, not class defaults."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import GameConfig


@dataclass(frozen=True)
class ShipStats:
    """Per-ship performance numbers. Every ship carries one; by default it is built from the
    GameConfig, so existing modes behave exactly as when these were global. Roguelite upgrades and
    enemy variants override them. Kept as plain numbers so a later training phase can feed them
    to the policy as observation features."""
    max_hull: int
    max_shield: int
    fire_rate: int          # frames between shots
    thrust_power: float
    max_speed: float
    bullet_lifespan: int    # frames; bullet range is about bullet_speed * bullet_lifespan
    bullet_damage: int = 1  # damage per hit on ships, stations, and rocks it can break
    break_size: int = 0     # largest asteroid size this ship's shots can break (0 = none)

    @classmethod
    def from_config(cls, config: GameConfig, fire_rate: Optional[int] = None) -> "ShipStats":
        return cls(max_hull=int(config.max_hull), max_shield=int(config.max_shield),
                   fire_rate=int(fire_rate if fire_rate is not None else config.fire_rate),
                   thrust_power=float(config.thrust_power), max_speed=float(config.max_speed),
                   bullet_lifespan=int(config.bullet_lifespan),
                   bullet_damage=int(config.bullet_damage),
                   break_size=int(config.breakable_asteroid_size))

    def as_dict(self) -> Dict[str, float]:
        return {k: (int(v) if isinstance(v, int) else float(v))
                for k, v in dataclasses.asdict(self).items()}

    def replace(self, **changes) -> "ShipStats":
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class Loadout:
    """What a roguelite wave puts in a ship slot: its stats plus cosmetic/payout labels."""
    stats: ShipStats
    variant: Optional[str] = None   # "standard" | "scout" | "heavy" | "sniper" | "wingman"
    tier: Optional[str] = None      # "basic" | "veteran" | "elite"


class Entity:
    def __init__(self, x, y, vx, vy, radius):
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([vx, vy], dtype=np.float64)
        self.radius = float(radius)
        self.active = True  # False when destroyed


class Ship(Entity):
    def __init__(self, x, y, team_id: int, agent_name: str, config: GameConfig, fire_rate=None,
                 stats: Optional[ShipStats] = None):
        super().__init__(x, y, 0.0, 0.0, config.ship_radius)
        self.team_id = team_id  # 0 red, 1 blue
        self.agent_name = agent_name
        self.angle = 0.0  # Radians
        self.cooldown = 0
        self.thrusting = False  # Last commanded thrust; used only for rendering the flame
        self.variant: Optional[str] = None  # roguelite look / payout labels (None elsewhere)
        self.tier: Optional[str] = None

        self.apply_stats(stats or ShipStats.from_config(config, fire_rate), refill=True)
        self.frames_since_damage = 10 ** 9  # Large so a fresh ship is never mid-regen
        self._regen_progress = 0
        self.collision_cooldown = 0  # Gates repeat collision damage while bodies overlap

        # Respawning modes
        self.respawn_timer = None    # Frames until this (destroyed) ship returns; None = not coming back
        self.protection = 0          # Frames of invulnerability left after spawning
        self.reserve = False         # Survival: not part of the current wave (neither alive nor dead)

    def apply_stats(self, stats: ShipStats, refill: bool = False):
        """Set this ship's performance numbers. ``refill`` restores full hull and shield;
        otherwise current values are clamped to the new maxima."""
        self.max_hull = int(stats.max_hull)
        self.max_shield = int(stats.max_shield)
        self.fire_rate = int(stats.fire_rate)
        self.thrust_power = float(stats.thrust_power)
        self.max_speed = float(stats.max_speed)
        self.bullet_lifespan = int(stats.bullet_lifespan)
        self.bullet_damage = int(stats.bullet_damage)
        self.break_size = int(stats.break_size)
        if refill:
            self.hull, self.shield = self.max_hull, self.max_shield
        else:
            self.hull = min(self.hull, self.max_hull)
            self.shield = min(self.shield, self.max_shield)

    @property
    def stats(self) -> ShipStats:
        return ShipStats(max_hull=self.max_hull, max_shield=self.max_shield,
                         fire_rate=self.fire_rate, thrust_power=self.thrust_power,
                         max_speed=self.max_speed, bullet_lifespan=self.bullet_lifespan,
                         bullet_damage=self.bullet_damage, break_size=self.break_size)

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


class Convoy(Station):
    """Escort: a slow, unarmed ship with a station's shield, driven along a fixed route."""

    def __init__(self, x, y, team_id: int, config: GameConfig):
        super().__init__(x, y, team_id, config)
        self.radius = float(config.convoy_radius)
        self.max_hull = self.hull = int(config.convoy_hull)
        self.max_shield = self.shield = int(config.convoy_shield)
        self.regen_per_frame = float(config.convoy_shield_regen) / 60.0
        self.distance = 0.0   # px travelled along the route
        self.moving = False   # escorted this frame (for the HUD)


class Flag:
    """Capture the flag: one per team. Lives at its base, on a carrier, or dropped in space."""

    def __init__(self, team_id: int, base_x: float, base_y: float):
        self.team_id = team_id
        self.base = np.array([base_x, base_y], dtype=np.float64)
        self.pos = self.base.copy()
        self.carrier = None       # agent name carrying it
        self.drop_timer = None    # frames until a dropped flag returns home

    @property
    def at_base(self) -> bool:
        return self.carrier is None and self.drop_timer is None

    def go_home(self):
        self.pos[:] = self.base
        self.carrier = None
        self.drop_timer = None


class Asteroid(Entity):
    def __init__(self, x, y, vx, vy, config: GameConfig, size: int = 3):
        super().__init__(x, y, vx, vy, config.asteroid_radius(size))
        self.size = size
        self.hp = int(config.asteroid_hp.get(size, 1))  # hits this rock takes to break


class Bullet(Entity):
    def __init__(self, x, y, vx, vy, owner_name: str, team_id: int, config: GameConfig,
                 lifespan: Optional[int] = None, damage: Optional[int] = None):
        super().__init__(x, y, vx, vy, config.bullet_radius)
        self.owner_name = owner_name
        self.team_id = team_id
        self.lifespan = int(lifespan if lifespan is not None else config.bullet_lifespan)
        self.max_lifespan = self.lifespan
        self.damage = int(damage if damage is not None else config.bullet_damage)
