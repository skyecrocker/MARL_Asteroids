"""Game, reward, and scenario configuration."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

    # Per-point-of-damage terms. These keep the signal from going sparse now that a kill takes
    # several hits, and they reward winning the damage exchange even without a kill. Kept small
    # relative to ``kill`` so finishing a target always beats chipping several.
    damage_dealt: float = 0.15
    damage_taken: float = -0.15
    teamdamage: float = -0.15  # Applied to the shooter for friendly-fire damage

    # Station siege: the shooter is paid per point of station damage; the defending team shares the
    # same amount as a penalty. Destroying a station pays the shooter ``station_kill``.
    station_damage: float = 0.05
    station_kill: float = 1.0

    # Capture the flag: picking up the enemy flag, capturing it, and returning your own
    flag_pickup: float = 0.2
    flag_capture: float = 1.0
    flag_return: float = 0.2

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

    # Durability. A ship absorbs ``max_shield + max_hull`` damage before dying. Shield takes damage
    # first and regenerates after ``shield_regen_delay`` frames without being hit, so disengaging to
    # recover is a real option; hull damage is permanent.
    max_hull: int = 3
    max_shield: int = 1
    shield_regen_delay: int = 180   # Frames without damage before the shield starts coming back
    shield_regen_frames: int = 60   # Frames per shield point restored
    bullet_damage: int = 1
    asteroid_damage: int = 2        # Rocks are heavy: half a ship's health per impact
    # Ramming is mutual, so making it a two-hit kill lets both sides trade themselves away and
    # removes tactical depth. Kept light so bumping is a real cost but shooting stays the way to kill.
    ram_damage: int = 1
    # Overlapping bodies would otherwise re-trigger collision damage every frame, which kills a
    # full-health ship in two frames and makes collisions lethal again. Contacts bounce apart, and
    # this cooldown gates *collision* damage only (bullets are consumed on impact, so they can
    # never double-hit).
    collision_damage_cooldown: int = 45
    bounce_restitution: float = 0.85

    # Game mode (see MODES). The default, sudden death (no respawns, last team standing), is what
    # every named scenario, training run and evaluation uses; the app picks modes explicitly.
    mode: str = "sudden_death"
    respawn_delay: int = 180        # Frames a destroyed ship waits before respawning (siege, deathmatch)
    spawn_protection: int = 90      # Frames of invulnerability after (re)spawning

    # Station siege. Each team has a station; ships respawn at it while it stands. Its shield
    # regenerates continuously (not after a quiet period like ships), and the regen also soaks hull
    # damage, so a station only takes damage faster than regen * time. The rate sits between one and
    # two ships' *realistic* output: a ship fires at most 60/fire_rate = 2 shots/s but lands far
    # fewer while maneuvering, so 1.6/s leaves a lone attacker needing over a minute uncontested,
    # while two or three wear a station down in roughly 20 s / 10 s. (A first tuning at 2.2/s with
    # 12 shield and 24 hull stalemated every bot game: stations never dropped below 90%.)
    station_radius: float = 38.0
    station_hull: int = 16
    station_shield: int = 10
    station_shield_regen: float = 1.6  # shield points per second
    # Stations sit half a map apart, the farthest two points can be on a torus, so neither the
    # direct path nor the wrap-around path between them is shorter.
    station_x: Tuple[float, float] = (0.25, 0.75)  # (red, blue) as fractions of the map width

    # Survival. Blue faces waves of red. Wave k has min(n_red, ceil(m_blue / 2) + (k - 1) *
    # wave_growth) ships, so the first wave is a warm-up where blue outnumbers red, and n_red caps
    # the wave size. When a wave is cleared, fallen blue ships respawn and all blue hulls are
    # repaired.
    wave_growth: int = 1
    wave_delay: int = 150           # Frames between clearing a wave and the next one arriving
    wave_spawn_distance: float = 0.35  # New enemies appear at least this fraction of the map away

    # Capture the flag. Bases sit at ``station_x`` like siege stations. Fly through the enemy flag
    # to take it and through your own base to score; touch your own dropped flag to send it home.
    flag_radius: float = 26.0         # pickup / capture distance from a flag or base center
    flag_return_time: int = 900       # frames a dropped flag waits before going home by itself
    flag_carrier_speed: float = 0.7   # carrier speed cap as a fraction of max_speed
    captures_to_win: int = 3

    # Escort. Blue escorts a convoy from x = convoy_start to convoy_start + convoy_route (fractions
    # of the map width) along the middle. It only moves while a blue ship is within convoy_escort_radius,
    # and its shield regenerates like a station's, so stopping it takes a coordinated attack.
    convoy_radius: float = 30.0
    convoy_hull: int = 24
    convoy_shield: int = 10
    convoy_shield_regen: float = 1.2   # shield points per second (weaker than a station's 1.6)
    convoy_speed: float = 0.45         # px per frame while escorted
    convoy_escort_radius: float = 250.0
    convoy_start: float = 0.15
    convoy_route: float = 0.7

    # Episode. ``max_frames=None`` means no truncation at all (used by the demo scenarios so a
    # human game never ends on a clock). Training scenarios must keep a limit.
    max_frames: Optional[int] = 1200
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
# max_frames=None: demo and human games never end on a clock.
DEMO_SCALE = dict(ship_radius=15.0, bullet_radius=3.5, max_speed=4.5,
                  asteroid_large_radius=38.0, asteroid_medium_radius=20.0, asteroid_small_radius=10.0,
                  max_frames=None, min_spawn_dist=220.0)
_DEMO_ROCKS = [
    (DEMO_W * 0.5, DEMO_H * 0.25, 0.0, 0.35, 3),
    (DEMO_W * 0.5, DEMO_H * 0.75, 0.0, -0.35, 3),
    (DEMO_W * 0.32, DEMO_H * 0.5, 0.3, 0.0, 2),
    (DEMO_W * 0.68, DEMO_H * 0.5, -0.3, 0.0, 2),
]

def scale_map(config: GameConfig, factor: float, asteroid_density: float = 1.0,
              **overrides) -> GameConfig:
    """Grow the arena by ``factor`` while keeping ships, weapons, and sensing unchanged.

    Scales the map, the truncation limit (travel time grows roughly linearly with size), and spawn
    separation. Asteroid count scales as ``factor * asteroid_density``: preserving true density would
    need ``factor**2``, which gets expensive and chaotic fast, so the default thins the field out.

    Deliberately does NOT scale ``sensor_range``. That length is a tactical scale tied to weapon
    reach (bullet_speed * bullet_lifespan is about 480 px), not a property of the map. Bearing to
    every tracked object stays exact at any distance, so a bigger arena means longer approaches with
    coarser range estimates rather than blindness.
    """
    def scale_spawns(spawns: List[EntityState]) -> List[EntityState]:
        # Positions are absolute, so they must scale with the map or a static layout drifts into a
        # corner and loses its symmetry. Velocities are physics and stay as they are.
        return [dataclasses.replace(bp, x=bp.x * factor, y=bp.y * factor,
                                    x_noise=bp.x_noise * factor, y_noise=bp.y_noise * factor)
                for bp in spawns]

    asteroid_spawns = scale_spawns(config.asteroid_spawns or [])
    return config.replace(
        width=config.width * factor,
        height=config.height * factor,
        max_frames=None if config.max_frames is None else int(config.max_frames * factor),
        min_spawn_dist=config.min_spawn_dist * factor,
        # Never drop below the explicitly placed rocks, so static layouts stay static.
        num_asteroids=max(round(config.num_asteroids * factor * asteroid_density),
                          len(asteroid_spawns)),
        red_spawns=scale_spawns(config.red_spawns or []),
        blue_spawns=scale_spawns(config.blue_spawns or []),
        asteroid_spawns=asteroid_spawns,
        **overrides,
    )


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

# --- Larger arenas: longer approaches, more room to maneuver and disengage ---
# Ship, weapon, and sensor scales are unchanged, so these are genuinely bigger worlds rather than a
# zoomed-in view. Useful for human play (more reaction time) and as a robustness check.
SCENARIOS.update({
    "duel_large": scale_map(SCENARIOS["duel_rocks"], 2.0),          # 1600x1600
    "team2_large": scale_map(SCENARIOS["team2"], 2.0),              # 1600x1600
    "team3_large": scale_map(SCENARIOS["team3"], 2.5),              # 2000x2000
    "team3_huge": scale_map(SCENARIOS["team3"], 4.0),               # 3200x3200
    # asteroid_density=0 keeps exactly the 4 placed rocks, so the layout stays fully deterministic.
    "demo_team2_large": scale_map(SCENARIOS["demo_team2"], 2.0, asteroid_density=0.0),  # 2400x2400
})

# ---------------------------------------------------------------------------
# Modes and arenas (what the app builds games from)
# ---------------------------------------------------------------------------

MODES: Dict[str, str] = {
    "deathmatch": "Destroyed ships respawn. The team with the most kills when time runs out wins.",
    "sudden_death": "No respawns. The last team with ships left wins.",
    "siege": "Ships respawn while their station stands. Destroy the enemy station, then their "
             "fleet. A lone ship can't break a station's shield.",
    "survival": "Blue holds out against growing waves of red. Fallen allies return between waves.",
    "ctf": "Take the enemy flag to your base. Carriers fly slower. First to 3 captures wins.",
    "escort": "Blue escorts a convoy across the map; it only moves with a blue ship nearby. "
              "Red wins by destroying it or running out the clock.",
}
MODE_NAMES: Dict[str, str] = {"deathmatch": "Deathmatch", "sudden_death": "Sudden death",
                              "siege": "Station siege", "survival": "Survival",
                              "ctf": "Capture the flag", "escort": "Escort"}
# Default time limit per mode in seconds (None = no limit). Deathmatch needs one (that's how it
# ends); siege gets one because it can stall if both teams turtle (healthier station wins).
MODE_TIME_LIMIT: Dict[str, Optional[float]] = {"deathmatch": 300.0, "sudden_death": None,
                                               "siege": 300.0, "survival": None,
                                               "ctf": 300.0, "escort": 300.0}
MAX_TEAM_SIZE = 8          # ships per team the app offers
SURVIVAL_MAX_ENEMIES = 20  # enemy pool in survival: waves grow by one each until this many at once
# Modes where destroyed ships come back. In these the env keeps every agent until the game ends.
RESPAWN_MODES = ("deathmatch", "siege", "survival", "ctf", "escort")

ARENA_SIZES: Dict[str, float] = {"mini": 600.0, "small": 1200.0, "medium": 2400.0, "large": 4800.0}
# Roughly constant density up to medium, sparser on large so it doesn't become an asteroid field
ARENA_ASTEROIDS: Dict[str, int] = {"mini": 2, "small": 5, "medium": 14, "large": 40}


def make_game(mode: str = "deathmatch", arena: str = "small", m_blue: int = 2, n_red: int = 2,
              num_asteroids: Optional[int] = None, asteroids_split: bool = True,
              fixed_spawns: bool = True, time_limit: Optional[float] = None,
              use_mode_time_limit: bool = True) -> GameConfig:
    """Build a game from the handful of choices the app exposes.

    Physics are the defaults the agents were trained on, so trained models play every arena.
    ``fixed_spawns`` (deathmatch and sudden death) lines the teams up facing each other's flank;
    otherwise ships spawn at random. Deathmatch respawns are always random, far from the enemy. Siege ships always spawn at their station, and survival puts blue in the middle
    with red waves arriving from far away. Asteroids are always placed at random.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
    if arena not in ARENA_SIZES:
        raise ValueError(f"unknown arena size {arena!r}; choose from {list(ARENA_SIZES)}")
    if m_blue < 1 or n_red < 1:
        raise ValueError("each team needs at least one ship")
    size = ARENA_SIZES[arena]
    if use_mode_time_limit:
        time_limit = MODE_TIME_LIMIT[mode]
    cfg = GameConfig(
        mode=mode, width=size, height=size, m_blue=m_blue, n_red=n_red,
        num_asteroids=ARENA_ASTEROIDS[arena] if num_asteroids is None else max(0, num_asteroids),
        asteroids_split=asteroids_split,
        max_frames=None if time_limit is None else max(1, int(time_limit * 60)),
        # Spawning ships 150 px apart is fine on a small map but too close to react on a big one
        min_spawn_dist=max(150.0, size * 0.15),
    )
    if mode in ("deathmatch", "sudden_death") and fixed_spawns:
        def formation(n, x, facing):
            spacing = min(320.0, size * 0.75 / max(1, n))
            return _line_formation(n, x, size / 2, spacing, facing)
        cfg.red_spawns = formation(n_red, size * 0.3, math.pi / 2)
        cfg.blue_spawns = formation(m_blue, size * 0.7, -math.pi / 2)
    elif mode == "survival":
        spacing = min(90.0, size * 0.6 / max(1, m_blue))  # never wider than the map
        cfg.blue_spawns = _line_formation(m_blue, size / 2, size / 2, spacing, -math.pi / 2)
    return cfg


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
