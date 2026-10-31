"""Roguelite survival: the flagship mode.

You fly ``blue_0`` against waves of red. Between waves a shop sells upgrades for your ship, AI
wingmen, and upgrades for the wingmen as a group. The run ends when your ship is destroyed.

Everything here is plain data plus pure functions (catalogs, enemy variants, intelligence tiers,
the wave schedule, the run state), and one controller, ``RogueliteRun``, that drives the env. No
pygame and no app imports, so a whole run can be played headless in tests. Tuning lives in
``DEFAULT_ROGUELITE``.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import ARENA_SIZES, ROGUELITE_MODE, SURVIVAL_MAX_ENEMIES, GameConfig
from .entities import Loadout, ShipStats
from .env import BLUE, PLAYER, RED, TeamAsteroidsEnv
from .policies import HumanPolicy, Policy, ReactionDelayPolicy, ScriptedPolicy, heuristic_action

VARIANT_ORDER = ("standard", "scout", "heavy", "sniper")
TIER_ORDER = ("basic", "veteran", "elite")
WINGMAN = "wingman"  # the look/policy key wingmen use in place of an enemy variant
# Rock sizes 1..4. Firepower level n breaks rocks up to size n, so the top firepower level reaches
# the largest rock. Rocks are indestructible to a ship that can't break them.
ROCK_TIERS = 4
ROCK_HP = {1: 1, 2: 2, 3: 4, 4: 8}  # hits to break each size at the base damage of 1
FIREPOWER_EFFECTS = (("bullet_damage", 1), ("break_size", 1))

# ------------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Tier:
    """Scripted bot skill. ``p`` is the chance per decision that the bot acts at all (the legacy
    heuristic knob); ``reaction_frames`` is how stale its actions are."""
    name: str
    label: str
    p: float
    reaction_frames: int


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    stats: ShipStats
    standoff: float = 150.0   # px the bot closes to before it stops thrusting


@dataclass(frozen=True)
class UpgradeDef:
    id: str
    name: str
    target: str                              # "player" | "wingmen"
    effects: Tuple[Tuple[str, float], ...]   # (ShipStats field, change per level)
    base_cost: int
    max_level: int
    floors: Tuple[Tuple[str, float], ...] = ()  # lower bounds after effects, e.g. fire_rate
    blurb: str = ""


@dataclass(frozen=True)
class WaveBand:
    """Composition rules from ``from_wave`` until the next band starts."""
    from_wave: int
    variant_weights: Tuple[Tuple[str, float], ...]
    veteran_share: float   # share of the non-elite ships that are veterans (rest basic)
    elite_share: float     # target share of the whole wave that is elite


@dataclass(frozen=True)
class EnemySpec:
    variant: str
    tier: str


_STANDARD = ShipStats(max_hull=3, max_shield=1, fire_rate=70, thrust_power=0.15, max_speed=4.0,
                      bullet_lifespan=80)


def _default_variants() -> Dict[str, Variant]:
    return {
        "standard": Variant("standard", "Standard", _STANDARD),
        "scout": Variant("scout", "Scout", _STANDARD.replace(max_hull=2, fire_rate=60,
                                                             thrust_power=0.20, max_speed=5.2,
                                                             bullet_lifespan=70)),
        "heavy": Variant("heavy", "Heavy", _STANDARD.replace(max_hull=6, max_shield=2, fire_rate=90,
                                                             thrust_power=0.11, max_speed=3.0)),
        "sniper": Variant("sniper", "Sniper", _STANDARD.replace(max_hull=2, fire_rate=110,
                                                                thrust_power=0.13, max_speed=3.6,
                                                                bullet_lifespan=140),
                          standoff=320.0),
    }


DEFAULT_TIERS = (Tier("basic", "Basic", p=0.6, reaction_frames=12),
                 Tier("veteran", "Veteran", p=0.8, reaction_frames=6),
                 Tier("elite", "Elite", p=1.0, reaction_frames=2))

ENGINE = (("thrust_power", 0.02), ("max_speed", 0.3))
DEFAULT_CATALOG = (
    UpgradeDef("player.hull", "Hull", "player", (("max_hull", 1),), 60, 5, blurb="+1 hull"),
    UpgradeDef("player.shield", "Shield", "player", (("max_shield", 1),), 50, 4, blurb="+1 shield"),
    UpgradeDef("player.fire_rate", "Fire rate", "player", (("fire_rate", -4),), 70, 5,
               floors=(("fire_rate", 12),), blurb="faster reload"),
    UpgradeDef("player.firepower", "Firepower", "player", FIREPOWER_EFFECTS, 80, ROCK_TIERS,
               blurb="+1 damage, breaks bigger rocks"),
    UpgradeDef("player.engine", "Engines", "player", ENGINE, 60, 4, blurb="more thrust and speed"),
    UpgradeDef("wing.buy", "Buy wingman", "wingmen", (), 120, 3, blurb="one more AI ship"),
    UpgradeDef("wing.tier", "Intelligence", "wingmen", (), 150, 2, blurb="smarter, faster bots"),
    UpgradeDef("wing.hull", "Hull", "wingmen", (("max_hull", 1),), 80, 5, blurb="+1 hull"),
    UpgradeDef("wing.shield", "Shield", "wingmen", (("max_shield", 1),), 70, 4, blurb="+1 shield"),
    UpgradeDef("wing.fire_rate", "Fire rate", "wingmen", (("fire_rate", -5),), 90, 5,
               floors=(("fire_rate", 15),), blurb="faster reload"),
    UpgradeDef("wing.firepower", "Firepower", "wingmen", FIREPOWER_EFFECTS, 100, ROCK_TIERS,
               blurb="+1 damage, breaks bigger rocks"),
    UpgradeDef("wing.engine", "Engines", "wingmen", ENGINE, 80, 4, blurb="more thrust and speed"),
)

DEFAULT_PAYOUTS = {
    "standard": {"basic": 10, "veteran": 15, "elite": 22},
    "scout": {"basic": 12, "veteran": 18, "elite": 26},
    "heavy": {"basic": 18, "veteran": 27, "elite": 40},
    "sniper": {"basic": 15, "veteran": 22, "elite": 33},
}

DEFAULT_SCHEDULE = (
    WaveBand(1, (("standard", 1.0),), veteran_share=0.0, elite_share=0.0),
    WaveBand(3, (("standard", 0.5), ("scout", 0.25), ("heavy", 0.25)), 0.0, 0.0),
    WaveBand(6, (("standard", 0.4), ("scout", 0.2), ("heavy", 0.2), ("sniper", 0.2)), 0.4, 0.0),
    WaveBand(9, (("standard", 0.3), ("scout", 0.25), ("heavy", 0.25), ("sniper", 0.2)), 0.7, 0.25),
    WaveBand(12, (("standard", 0.25), ("scout", 0.25), ("heavy", 0.25), ("sniper", 0.25)), 1.0, 0.6),
)


@dataclass(frozen=True)
class RogueliteConfig:
    arena: str = "medium"
    num_asteroids: int = 14          # large rocks (size 3) in the field
    huge_asteroids: int = 4          # the largest rocks (size 4), placed at the start of a run
    asteroid_hp: Dict[int, int] = field(default_factory=lambda: dict(ROCK_HP))
    start_credits: int = 60
    clear_bonus_base: int = 20
    clear_bonus_per_wave: int = 5
    base_wave_size: int = 2
    max_enemies: int = SURVIVAL_MAX_ENEMIES
    max_fleet: int = 4   # ships including the player
    cost_growth: float = 1.5
    player_base: ShipStats = ShipStats(max_hull=3, max_shield=1, fire_rate=30, thrust_power=0.15,
                                       max_speed=4.0, bullet_lifespan=80)
    wingman_base: ShipStats = ShipStats(max_hull=3, max_shield=1, fire_rate=45, thrust_power=0.15,
                                        max_speed=4.0, bullet_lifespan=80)
    variants: Dict[str, Variant] = field(default_factory=_default_variants)
    tiers: Tuple[Tier, ...] = DEFAULT_TIERS   # weakest first
    catalog: Tuple[UpgradeDef, ...] = DEFAULT_CATALOG
    payouts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        v: dict(t) for v, t in DEFAULT_PAYOUTS.items()})
    schedule: Tuple[WaveBand, ...] = DEFAULT_SCHEDULE
    elite_majority_from: int = 12

    def item(self, item_id: str) -> UpgradeDef:
        for u in self.catalog:
            if u.id == item_id:
                return u
        raise KeyError(f"unknown shop item {item_id!r}")

    def tier(self, name: str) -> Tier:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(f"unknown tier {name!r}")

    def band(self, wave: int) -> WaveBand:
        best = self.schedule[0]
        for b in self.schedule:
            if b.from_wave <= wave:
                best = b
        return best

    def validate(self) -> None:
        """Raise ValueError if the tuning tables break a rule the game relies on."""
        if self.arena not in ARENA_SIZES:
            raise ValueError(f"unknown arena {self.arena!r}")
        if not 1 <= self.max_fleet:
            raise ValueError("max_fleet must be at least 1 (the player)")
        if self.base_wave_size < 1 or self.max_enemies < self.base_wave_size:
            raise ValueError("wave sizes must be between 1 and max_enemies")
        names = [t.name for t in self.tiers]
        if not names or len(set(names)) != len(names):
            raise ValueError("tiers must be non-empty with unique names")
        for weak, strong in zip(self.tiers, self.tiers[1:]):
            if strong.reaction_frames > weak.reaction_frames or strong.p < weak.p:
                raise ValueError(f"tier {strong.name} must react at least as fast and as often "
                                 f"as {weak.name}")
        for v in self.variants:
            pays = self.payouts.get(v)
            if pays is None or any(t not in pays for t in names):
                raise ValueError(f"missing kill payouts for variant {v!r}")
            if any(pays[a] > pays[b] for a, b in zip(names, names[1:])):
                raise ValueError(f"payouts for {v!r} must not drop with tier")
        ids = [u.id for u in self.catalog]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate shop item ids")
        for u in self.catalog:
            if u.base_cost < 1 or u.max_level < 0 or u.target not in ("player", "wingmen"):
                raise ValueError(f"bad shop item {u.id!r}")
        if sorted(self.asteroid_hp) != list(range(1, ROCK_TIERS + 1)) or min(self.asteroid_hp.values()) < 1:
            raise ValueError(f"asteroid_hp needs a positive entry for each size 1..{ROCK_TIERS}")
        for item_id in ("player.firepower", "wing.firepower"):
            if item_id not in ids:
                raise ValueError(f"missing shop item {item_id!r}")
            u = self.item(item_id)
            reach = sum(delta for name, delta in u.effects if name == "break_size") * u.max_level
            if reach < ROCK_TIERS:
                raise ValueError(f"{item_id} must be able to break the largest rock")
        if "wing.buy" in ids and self.item("wing.buy").max_level != self.max_fleet - 1:
            raise ValueError("wing.buy max_level must be max_fleet - 1")
        if "wing.tier" in ids and self.item("wing.tier").max_level != len(self.tiers) - 1:
            raise ValueError("wing.tier max_level must be len(tiers) - 1")
        if not self.schedule or self.schedule[0].from_wave != 1:
            raise ValueError("the wave schedule must start at wave 1")
        if any(b.from_wave >= c.from_wave for b, c in zip(self.schedule, self.schedule[1:])):
            raise ValueError("wave bands must be in increasing from_wave order")
        for b in self.schedule:
            positive = [v for v, w in b.variant_weights if w > 0]
            if not positive or any(v not in self.variants for v in positive):
                raise ValueError(f"band from wave {b.from_wave} has unknown or no variants")
            if wave_size(self, b.from_wave) < len(positive):
                raise ValueError(f"band from wave {b.from_wave} has more variants than ships")
            if b.elite_share > 0 and wave_size(self, b.from_wave) < 2:
                raise ValueError(f"band from wave {b.from_wave} needs 2+ ships for elites")
        if self.band(self.elite_majority_from).elite_share <= 0.5:
            raise ValueError("the band at elite_majority_from must have elite_share > 0.5")


DEFAULT_ROGUELITE = RogueliteConfig()


def make_roguelite_config(cfg: RogueliteConfig = DEFAULT_ROGUELITE) -> GameConfig:
    """The env config for a run: fixed arena, 4 blue and 20 red slots, no clock, no friendly fire
    (your own shots would otherwise make wingmen a liability). Rocks are breakable only through
    firepower, so ships start with ``breakable_asteroid_size=0``. Physics stay at the defaults."""
    size = ARENA_SIZES[cfg.arena]
    return GameConfig(mode=ROGUELITE_MODE, width=size, height=size, m_blue=cfg.max_fleet,
                      n_red=cfg.max_enemies, num_asteroids=cfg.num_asteroids,
                      num_huge_asteroids=cfg.huge_asteroids, asteroid_hp=dict(cfg.asteroid_hp),
                      breakable_asteroid_size=0, asteroids_split=True, friendly_fire=False,
                      max_frames=None, min_spawn_dist=max(150.0, size * 0.15))


# --------------------------------------------------------------------------- run state


@dataclass
class RunState:
    """Everything about a run in progress, as plain values (saved runs and meta-progression can
    build on ``to_dict``/``from_dict`` later)."""
    wave: int = 1
    waves_cleared: int = 0
    credits: int = 0
    credits_earned: int = 0
    player_kills: int = 0
    fleet_kills: int = 0     # kills by any blue ship, yours included
    wingmen: int = 0
    upgrades: Dict[str, int] = field(default_factory=dict)  # item id -> level (not wing.buy)
    seed: int = 0

    @classmethod
    def new(cls, cfg: RogueliteConfig = DEFAULT_ROGUELITE, seed: Optional[int] = None) -> "RunState":
        if seed is None:
            seed = random.SystemRandom().randrange(2 ** 31)
        return cls(credits=cfg.start_credits, seed=int(seed),
                   upgrades={u.id: 0 for u in cfg.catalog if u.id != "wing.buy"})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunState":
        """Tolerant of missing and unknown keys."""
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        if "upgrades" in kwargs:
            kwargs["upgrades"] = {str(k): int(v) for k, v in dict(kwargs["upgrades"]).items()}
        return cls(**kwargs)


class PurchaseError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------------------------ shop


def upgrade_cost(base_cost: int, level: int, growth: float = 1.5) -> int:
    return int(round(base_cost * growth ** level))


def level_of(state: RunState, item_id: str) -> int:
    if item_id == "wing.buy":
        return state.wingmen
    return int(state.upgrades.get(item_id, 0))


def item_cost(cfg: RogueliteConfig, state: RunState, item_id: str) -> int:
    u = cfg.item(item_id)
    return upgrade_cost(u.base_cost, level_of(state, item_id), cfg.cost_growth)


def can_buy(state: RunState, item_id: str, cfg: RogueliteConfig = DEFAULT_ROGUELITE) -> Tuple[bool, str]:
    """(allowed, reason). Reasons: "maxed", "no wingmen", "credits", or "" when allowed."""
    u = cfg.item(item_id)
    if level_of(state, item_id) >= u.max_level:
        return False, "maxed"
    if u.target == "wingmen" and item_id != "wing.buy" and state.wingmen <= 0:
        return False, "no wingmen"
    if state.credits < item_cost(cfg, state, item_id):
        return False, "credits"
    return True, ""


def buy(state: RunState, item_id: str, cfg: RogueliteConfig = DEFAULT_ROGUELITE) -> RunState:
    """A new state with one more level of ``item_id``. Raises PurchaseError if it can't be bought."""
    ok, reason = can_buy(state, item_id, cfg)
    if not ok:
        raise PurchaseError(reason)
    cost = item_cost(cfg, state, item_id)
    upgrades = dict(state.upgrades)
    wingmen = state.wingmen
    if item_id == "wing.buy":
        wingmen += 1
    else:
        upgrades[item_id] = upgrades.get(item_id, 0) + 1
    return dataclasses.replace(state, credits=state.credits - cost, upgrades=upgrades,
                               wingmen=wingmen)


def effective_stats(cfg: RogueliteConfig, state: RunState, target: str) -> ShipStats:
    """Base stats for ``target`` ("player" or "wingmen") plus every bought level for it."""
    base = cfg.player_base if target == "player" else cfg.wingman_base
    values = dataclasses.asdict(base)
    floors: Dict[str, float] = {}
    for u in cfg.catalog:
        if u.target != target:
            continue
        lvl = level_of(state, u.id) if u.effects else 0
        for name, delta in u.effects:
            values[name] = values[name] + delta * lvl
        floors.update(dict(u.floors))
    for name, low in floors.items():
        values[name] = max(values[name], low)
    out = {}
    for f in dataclasses.fields(ShipStats):
        v = values[f.name]
        out[f.name] = int(round(v)) if isinstance(getattr(base, f.name), int) else float(v)
    out["fire_rate"] = max(1, out["fire_rate"])
    return ShipStats(**out)


def wingman_tier(cfg: RogueliteConfig, state: RunState) -> Tier:
    return cfg.tiers[min(level_of(state, "wing.tier"), len(cfg.tiers) - 1)]


@dataclass(frozen=True)
class ShopRow:
    id: str
    column: str          # "player" | "wingmen"
    name: str
    blurb: str
    level_text: str
    cost: Optional[int]  # None when maxed
    enabled: bool
    maxed: bool
    reason: str


def shop_rows(state: RunState, cfg: RogueliteConfig = DEFAULT_ROGUELITE) -> List[ShopRow]:
    rows = []
    for u in cfg.catalog:
        lvl = level_of(state, u.id)
        maxed = lvl >= u.max_level
        if u.id == "wing.buy":
            level_text = f"{lvl}/{u.max_level} owned"
        elif u.id == "wing.tier":
            level_text = wingman_tier(cfg, state).label
        else:
            level_text = f"Lv {lvl}/{u.max_level}"
        ok, reason = can_buy(state, u.id, cfg)
        rows.append(ShopRow(u.id, u.target, u.name, u.blurb, level_text,
                            None if maxed else item_cost(cfg, state, u.id), ok, maxed, reason))
    return rows


# ------------------------------------------------------------------- waves and economy


def wave_clear_bonus(cfg: RogueliteConfig, wave: int) -> int:
    return cfg.clear_bonus_base + cfg.clear_bonus_per_wave * wave


def on_wave_cleared(state: RunState, cfg: RogueliteConfig = DEFAULT_ROGUELITE) -> RunState:
    bonus = wave_clear_bonus(cfg, state.wave)
    return dataclasses.replace(state, waves_cleared=state.waves_cleared + 1,
                               credits=state.credits + bonus,
                               credits_earned=state.credits_earned + bonus)


def advance_wave(state: RunState) -> RunState:
    return dataclasses.replace(state, wave=state.wave + 1)


def wave_size(cfg: RogueliteConfig, wave: int) -> int:
    """About one more ship every two waves, capped at max_enemies."""
    return min(cfg.max_enemies, cfg.base_wave_size + (max(1, wave) - 1) // 2)


def elite_count(cfg: RogueliteConfig, wave: int) -> int:
    """Elites in wave ``wave``. Ratcheted so the elite *share* never drops as waves grow (plain
    rounding of share * size would dip), and capped at size - 1 to leave a veteran slot."""
    e_prev, n_prev = 0, 1
    e = 0
    for k in range(1, max(1, wave) + 1):
        n = wave_size(cfg, k)
        share = cfg.band(k).elite_share
        if share <= 0 and e_prev == 0:
            e = 0
        else:
            e = max(math.ceil(share * n - 1e-9), math.ceil(e_prev * n / n_prev - 1e-9))
            e = min(n - 1, e)
        e_prev, n_prev = e, n
    return e


def _largest_remainder(n: int, weights: Sequence[Tuple[str, float]]) -> List[str]:
    """``n`` variant names: one of each weighted variant, the rest shared out by weight."""
    pos = [(v, w) for v, w in weights if w > 0]
    counts = {v: 1 for v, _ in pos}
    extra = n - len(pos)
    if extra > 0:
        total = sum(w for _, w in pos)
        quotas = [(v, w / total * extra) for v, w in pos]
        floors = {v: int(math.floor(q)) for v, q in quotas}
        left = extra - sum(floors.values())
        order = sorted(range(len(quotas)), key=lambda i: (-(quotas[i][1] - floors[quotas[i][0]]), i))
        for i in order[:left]:
            floors[quotas[i][0]] += 1
        for v in counts:
            counts[v] += floors[v]
    out = []
    for v, _ in pos:
        out += [v] * counts[v]
    return out[:n]


def wave_composition(cfg: RogueliteConfig, wave: int, seed: int) -> Tuple[EnemySpec, ...]:
    """The enemies of wave ``wave``: depends only on (cfg, wave, seed)."""
    n = wave_size(cfg, wave)
    band = cfg.band(wave)
    e = elite_count(cfg, wave)
    rest = n - e
    v = 0
    if band.veteran_share > 0 and rest > 0:
        v = max(1, int(math.floor(band.veteran_share * rest + 0.5)))
        v = min(v, rest)
    tiers = ["elite"] * e + ["veteran"] * v + ["basic"] * (rest - v)
    variants = _largest_remainder(n, band.variant_weights)
    rng = np.random.default_rng([int(seed) % (2 ** 32), int(wave)])
    order = rng.permutation(len(variants))
    return tuple(EnemySpec(variants[i], t) for i, t in zip(order, tiers))


def kill_credits(events: Sequence[Tuple[str, str]], looks: Mapping[str, Tuple[str, str]],
                 cfg: RogueliteConfig = DEFAULT_ROGUELITE) -> Tuple[int, int, int]:
    """(credits, player_kills, fleet_kills) for new ``(source, victim)`` kill events. Only blue
    shots on red ships pay; rams and asteroids pay nothing."""
    credits = player = fleet = 0
    for source, victim in events:
        if not (source.startswith("blue_") and victim.startswith("red_")):
            continue
        variant, tier = looks.get(victim, ("standard", "basic"))
        credits += int(cfg.payouts.get(variant, {}).get(tier, 0))
        fleet += 1
        if source == PLAYER:
            player += 1
    return credits, player, fleet


def hud_line(state: RunState, enemies_left: int) -> str:
    return f"Wave {state.wave}  ·  {enemies_left} left  ·  {state.credits} cr"


# ----------------------------------------------------------------------------- the run


class RogueliteRun:
    """Drives one run: the env, the policies, and the run state.

    Phases: "idle" (before start), "wave" (fighting), "shop" (wave cleared, waiting for
    ``next_wave``), "over" (the player's ship was destroyed).
    """

    def __init__(self, cfg: RogueliteConfig = DEFAULT_ROGUELITE, state: Optional[RunState] = None,
                 player_policy: Optional[Policy] = None, seed: Optional[int] = None):
        cfg.validate()
        self.cfg = cfg
        self.state = state if state is not None else RunState.new(cfg, seed)
        self.player_policy = player_policy if player_policy is not None else HumanPolicy()
        self.game_config = make_roguelite_config(cfg)
        self.env: Optional[TeamAsteroidsEnv] = None
        self.phase = "idle"
        self.follow = PLAYER
        self.human_agents = [PLAYER] if isinstance(self.player_policy, HumanPolicy) else []
        self.policies: Dict[str, Policy] = {}
        self.looks: Dict[str, Tuple[str, str]] = {}  # red agent -> (variant, tier) this wave
        self._cache: Dict[Tuple[str, str], ReactionDelayPolicy] = {}
        self._kills_seen = 0

    # setup -------------------------------------------------------------------
    def make_env(self, render_mode: Optional[str] = None, **render_options) -> TeamAsteroidsEnv:
        self.env = TeamAsteroidsEnv(self.game_config, render_mode=render_mode, **render_options)
        return self.env

    def start(self) -> None:
        if self.env is None:
            self.make_env()
        self.env.reset(seed=self.state.seed)
        self._start_wave()

    def _policy(self, tier: Tier, key: str, standoff: float) -> ReactionDelayPolicy:
        cache_key = (tier.name, key)
        pol = self._cache.get(cache_key)
        if pol is None:
            inner = ScriptedPolicy(f"{key}_{tier.name}", heuristic_action,
                                   seed=self.state.seed + len(self._cache) + 1, p=tier.p,
                                   standoff=standoff)
            delay = int(round(tier.reaction_frames / max(1, self.game_config.action_repeat)))
            pol = self._cache[cache_key] = ReactionDelayPolicy(inner, delay)
        return pol

    def fleet_loadouts(self) -> Dict[str, Loadout]:
        cfg, state = self.cfg, self.state
        tier = wingman_tier(cfg, state)
        wing = effective_stats(cfg, state, "wingmen")
        fleet = {PLAYER: Loadout(effective_stats(cfg, state, "player"))}
        for i in range(1, min(state.wingmen, cfg.max_fleet - 1) + 1):
            fleet[f"blue_{i}"] = Loadout(wing, WINGMAN, tier.name)
        return fleet

    def _start_wave(self) -> None:
        cfg, state, env = self.cfg, self.state, self.env
        fleet = self.fleet_loadouts()
        comp = wave_composition(cfg, state.wave, state.seed)
        enemies = [Loadout(cfg.variants[s.variant].stats, s.variant, s.tier) for s in comp]
        env.start_wave(state.wave, enemies, fleet)
        wing_tier = wingman_tier(cfg, state)
        self.policies = {PLAYER: self.player_policy}
        for agent in fleet:
            if agent != PLAYER:
                self.policies[agent] = self._policy(wing_tier, WINGMAN, 150.0)
        self.looks = {}
        for k, s in enumerate(comp):
            agent = f"red_{k}"
            self.policies[agent] = self._policy(cfg.tier(s.tier), s.variant,
                                                cfg.variants[s.variant].standoff)
            self.looks[agent] = (s.variant, s.tier)
        for pol in self._cache.values():
            pol.reset()
        self._kills_seen = len(env.kill_log)
        self.phase = "wave"

    # play --------------------------------------------------------------------
    def actions(self) -> Dict[str, np.ndarray]:
        """One action per ship in play, batched per policy instance."""
        env = self.env
        groups: Dict[int, tuple] = {}
        for agent, pol in self.policies.items():
            if env.ships[agent].active:
                groups.setdefault(id(pol), (pol, []))[1].append(agent)
        out = {}
        for pol, agents in groups.values():
            acts = pol.act(np.stack([env.get_obs(a) for a in agents]), [(env, a) for a in agents])
            out.update(zip(agents, acts))
        return out

    def step(self) -> str:
        """Advance one decision. Returns the phase afterwards."""
        if self.phase != "wave":
            raise RuntimeError(f"step() during the {self.phase!r} phase")
        env = self.env
        env.step(self.actions())
        new = env.kill_log[self._kills_seen:]
        self._kills_seen = len(env.kill_log)
        credits, mine, fleet = kill_credits(new, self.looks, self.cfg)
        if credits or mine or fleet:
            s = self.state
            self.state = dataclasses.replace(s, credits=s.credits + credits,
                                             credits_earned=s.credits_earned + credits,
                                             player_kills=s.player_kills + mine,
                                             fleet_kills=s.fleet_kills + fleet)
        if env.game_over:
            self.phase = "over"
        elif not env.wave_active:
            self.state = on_wave_cleared(self.state, self.cfg)
            self.phase = "shop"
        return self.phase

    def buy(self, item_id: str) -> None:
        if self.phase != "shop":
            raise RuntimeError(f"buy() during the {self.phase!r} phase")
        self.state = buy(self.state, item_id, self.cfg)

    def next_wave(self) -> None:
        if self.phase != "shop":
            raise RuntimeError(f"next_wave() during the {self.phase!r} phase")
        self.state = advance_wave(self.state)
        self._start_wave()

    # info --------------------------------------------------------------------
    def enemies_left(self) -> int:
        return 0 if self.env is None else self.env._team_alive(RED)

    def hud_text(self) -> str:
        return hud_line(self.state, self.enemies_left())

    def summary(self) -> dict:
        s = self.state
        return {"mode": ROGUELITE_MODE, "wave": s.wave, "waves_cleared": s.waves_cleared,
                "player_kills": s.player_kills, "fleet_kills": s.fleet_kills,
                "credits_earned": s.credits_earned, "credits": s.credits, "wingmen": s.wingmen}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
