"""Policies that control a set of ships.

All policies share one batched interface::

    actions = policy.act(obs, contexts)   # obs: (N, obs_dim), contexts: [(env, agent_name), ...]

Scripted policies use the contexts (exact physics state); network policies use the obs.
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch as th

Context = Tuple["TeamAsteroidsEnv", str]  # noqa: F821

NOOP = np.array([0, 0, 0], dtype=np.int64)


class Policy:
    name: str = "policy"

    def act(self, obs: np.ndarray, contexts: Sequence[Context]) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"


# ---------------------------------------------------------------- scripted


def _angle_diff(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def _steer(angle_diff, tol=0.1):
    return 2 if angle_diff > tol else (1 if angle_diff < -tol else 0)


def heuristic_action(env, agent_name, rng, p: float = 1.0, danger_zone: float = 100.0,
                     collision_margin: float = 3.5):
    """The original 'smart' bot: flee the nearest threat, otherwise turn to the enemy and shoot.

    With probability ``1 - p`` it does nothing this step (the legacy "p% optimal" knob).

    ``collision_margin`` (in ego radii) adds a late break-off from imminent ship contact. Without it
    the bot coasts straight into whatever it is shooting at and trades itself away by ramming, which
    makes for a weak baseline.
    """
    if rng.random() > p:
        return NOOP.copy()
    ego = env.ships[agent_name]
    if not ego.active:
        return NOOP.copy()

    # Imminent ship collision: thrust away from contact rather than coasting into it.
    collide_r = ego.radius * collision_margin
    for name, other in env.ships.items():
        if name == agent_name or not other.active:
            continue
        dx, dy, d = env.wrapped_delta(ego.pos, other.pos)
        if d < collide_r:
            escape = _angle_diff(math.atan2(-dy, -dx), ego.angle)
            return np.array([1 if abs(escape) < 1.2 else 0, _steer(escape), 0])

    closest, tdx, tdy = float("inf"), 0.0, 0.0
    for ast in env.asteroids:
        if ast.active:
            dx, dy, d = env.wrapped_delta(ego.pos, ast.pos)
            if d < closest:
                closest, tdx, tdy = d, dx, dy
    for b in env.bullets:
        if b.active and b.team_id != ego.team_id:
            dx, dy, d = env.wrapped_delta(ego.pos, b.pos)
            if d < closest:
                closest, tdx, tdy = d, dx, dy
    stations = getattr(env, "stations", {})
    for st in stations.values():  # solid obstacles, measured to the surface
        if st.active:
            dx, dy, d = env.wrapped_delta(ego.pos, st.pos)
            if d - st.radius < closest:
                closest, tdx, tdy = d - st.radius, dx, dy
    if closest < danger_zone:
        diff = _angle_diff(math.atan2(-tdy, -tdx), ego.angle)
        return np.array([1 if abs(diff) < 1.5 else 0, _steer(diff), 0])

    if getattr(env, "flags", None):
        return _ctf_action(env, ego, agent_name)
    if getattr(env, "convoy", None) is not None and ego.team_id == 1:
        return _escort_action(env, ego, agent_name)
    if stations:  # siege, and escort attackers (the convoy is blue's "station")
        return _siege_attack(env, ego, agent_name, stations)

    best, tgt = float("inf"), None
    for name, other in env.ships.items():
        if name != agent_name and other.active and other.team_id != ego.team_id:
            dx, dy, d = env.wrapped_delta(ego.pos, other.pos)
            if d < best:
                best, tgt = d, (dx, dy)
    if tgt is None:
        return NOOP.copy()
    diff = _angle_diff(math.atan2(tgt[1], tgt[0]), ego.angle)
    return np.array([1 if abs(diff) < 0.5 and best > 150 else 0, _steer(diff), 1 if abs(diff) < 0.2 else 0])


# ------------------------------------------------------------- objective-mode helpers


def _enemies(env, ego, agent_name):
    return [(n, s) for n, s in env.ships.items()
            if n != agent_name and s.active and s.team_id != ego.team_id]


def _fight(env, ego, other):
    """The deathmatch attack: turn to the target, close to 150 px, fire when lined up."""
    dx, dy, d = env.wrapped_delta(ego.pos, other.pos)
    diff = _angle_diff(math.atan2(dy, dx), ego.angle)
    return np.array([1 if abs(diff) < 0.5 and d > 150 else 0, _steer(diff), 1 if abs(diff) < 0.2 else 0])


def _go_to(env, ego, agent_name, pos, arrive=0.0):
    """Fly to a point (``arrive`` px is close enough), taking any shot that lines up on the way."""
    dx, dy, d = env.wrapped_delta(ego.pos, pos)
    diff = _angle_diff(math.atan2(dy, dx), ego.angle)
    thrust = 1 if abs(diff) < 0.6 and d > arrive else 0
    reach = env.config.bullet_speed * env.config.bullet_lifespan * 0.8
    shoot = 0
    for _, other in _enemies(env, ego, agent_name):
        ex, ey, ed = env.wrapped_delta(ego.pos, other.pos)
        if ed < reach and abs(_angle_diff(math.atan2(ey, ex), ego.angle)) < 0.12:
            shoot = 1
            break
    return np.array([thrust, _steer(diff) if d > arrive else 0, shoot])


def _nearest_enemy(env, ego, agent_name, around=None, within=float("inf")):
    """Closest enemy ship to ``around`` (default: me) within ``within`` px of it, or None."""
    center = ego.pos if around is None else around
    best, best_d = None, within
    for _, other in _enemies(env, ego, agent_name):
        d = env._wrapped_dist(center, other.pos)
        if d < best_d:
            best, best_d = other, d
    return best


def _ctf_action(env, ego, agent_name):
    """Even-numbered ships run for the enemy flag, odd ones defend; anyone with the flag goes home,
    and the team hunts whoever is carrying its own flag."""
    team = ego.team_id
    own, theirs = env.flags[team], env.flags[1 - team]
    if theirs.carrier == agent_name:
        return _go_to(env, ego, agent_name, own.base)
    runner = int(agent_name.split("_")[1]) % 2 == 0
    if own.carrier is not None:
        thief = env.ships[own.carrier]
        if not runner or env._wrapped_dist(ego.pos, thief.pos) < 350:
            return _fight(env, ego, thief)
    if own.drop_timer is not None and not runner:
        return _go_to(env, ego, agent_name, own.pos)  # touch it to send it home
    if runner:
        if theirs.carrier is None:
            return _go_to(env, ego, agent_name, theirs.pos)
        carrier = env.ships[theirs.carrier]  # a teammate has it: cover them
        threat = _nearest_enemy(env, ego, agent_name, around=carrier.pos, within=350)
        if threat is not None:
            return _fight(env, ego, threat)
        return _go_to(env, ego, agent_name, carrier.pos, arrive=120)
    raider = _nearest_enemy(env, ego, agent_name, around=own.base, within=500)
    if raider is None:
        raider = _nearest_enemy(env, ego, agent_name, within=300)
    if raider is not None:
        return _fight(env, ego, raider)
    return _go_to(env, ego, agent_name, own.base, arrive=160)


def _escort_action(env, ego, agent_name):
    """Blue in escort: stay close enough to keep the convoy moving, fight what comes near it."""
    convoy = env.convoy
    team = env.team_agents[ego.team_id]
    i, n = team.index(agent_name), len(team)
    radius = env.config.convoy_escort_radius
    # Station ahead of the convoy, spread across its path
    post = ((convoy.pos[0] + 140.0) % env.width,
            (convoy.pos[1] + (i - (n - 1) / 2) * 70.0) % env.height)
    if env._wrapped_dist(ego.pos, convoy.pos) > radius * 0.75:
        return _go_to(env, ego, agent_name, post)
    threat = _nearest_enemy(env, ego, agent_name, around=convoy.pos, within=450)
    if threat is not None:
        return _fight(env, ego, threat)
    return _go_to(env, ego, agent_name, post, arrive=60)


# Siege target preference, as multipliers on distance (lower = more attractive):
PUSH = 0.75         # the enemy station is a bit more attractive than an equally distant ship
PUSH_CONVOY = 0.4   # escort attackers go for the convoy over all but the closest escorts
DEFEND = 0.55       # enemies near our own station are the most urgent
DEFEND_RADIUS = 420.0
STANDOFF = 230.0    # px from the station surface: inside bullet range (~480), out of ramming range


def _siege_attack(env, ego, agent_name, stations):
    """Pick between enemy ships and the enemy station, defending home first."""
    home = stations.get(ego.team_id)
    enemy_team = 1 - ego.team_id
    options = []  # (score, dx, dy, dist, is_station, radius)
    for name, other in env.ships.items():
        if name == agent_name or not other.active or other.team_id == ego.team_id:
            continue
        dx, dy, d = env.wrapped_delta(ego.pos, other.pos)
        mult = 1.0
        if home is not None and home.active and env._wrapped_dist(other.pos, home.pos) < DEFEND_RADIUS:
            mult = DEFEND
        options.append((d * mult, dx, dy, d, False, other.radius))
    target_station = stations.get(enemy_team)
    if target_station is not None and target_station.active:
        dx, dy, d = env.wrapped_delta(ego.pos, target_station.pos)
        # Escort attackers live for the convoy: it is the only way to win before the clock
        push = PUSH_CONVOY if getattr(env, "convoy", None) is target_station else PUSH
        options.append((d * push, dx, dy, d, True, target_station.radius))
    if not options:
        return NOOP.copy()
    _, dx, dy, d, is_station, radius = min(options, key=lambda o: o[0])
    diff = _angle_diff(math.atan2(dy, dx), ego.angle)
    if is_station:
        # A big target: anything within its angular radius hits. Hold off at standoff range, and
        # only fire when the shot can arrive (bullets fly bullet_speed * lifespan, ~480 px).
        tol = max(0.2, math.atan2(radius * 0.8, max(d, 1.0)))
        thrust = 1 if abs(diff) < 0.5 and d - radius > STANDOFF else 0
        reach = env.config.bullet_speed * env.config.bullet_lifespan * 0.85
        shoot = 1 if abs(diff) < tol and d - radius < reach else 0
        return np.array([thrust, _steer(diff), shoot])
    return np.array([1 if abs(diff) < 0.5 and d > 150 else 0, _steer(diff), 1 if abs(diff) < 0.2 else 0])


def evasion_action(env, agent_name, rng, ast_only=False, shooting=False, danger_zone=150.0):
    """Non-aggressive bot: flees threats (asteroids, bullets, ships), otherwise wanders."""
    ego = env.ships[agent_name]
    if not ego.active:
        return NOOP.copy()
    closest, tdx, tdy = float("inf"), 0.0, 0.0
    threats = [a.pos for a in env.asteroids if a.active]
    if not ast_only:
        threats += [b.pos for b in env.bullets if b.active and b.team_id != ego.team_id]
    threats += [s.pos for n, s in env.ships.items() if n != agent_name and s.active]
    for pos in threats:
        dx, dy, d = env.wrapped_delta(ego.pos, pos)
        if d < closest:
            closest, tdx, tdy = d, dx, dy
    for st in getattr(env, "stations", {}).values():
        if st.active:
            dx, dy, d = env.wrapped_delta(ego.pos, st.pos)
            if d - st.radius < closest:
                closest, tdx, tdy = d - st.radius, dx, dy
    if closest < danger_zone:
        diff = _angle_diff(math.atan2(-tdy, -tdx), ego.angle)
        return np.array([1 if abs(diff) < 1.5 else 0, _steer(diff), 0])
    u = rng.random(4)
    turn = 1 if u[1] < 0.1 else (2 if u[2] > 0.9 else 0)
    return np.array([int(u[0] < 0.1), turn, int(shooting and u[3] < 0.1)])


class ScriptedPolicy(Policy):
    """Wraps a per-agent scripted function ``fn(env, agent, rng, **kwargs)``."""

    def __init__(self, name, fn, seed: Optional[int] = None, **kwargs):
        self.name = name
        self.fn = fn
        self.kwargs = kwargs
        self.rng = np.random.default_rng(seed)

    def act(self, obs, contexts):
        if not contexts:
            return np.zeros((0, 3), dtype=np.int64)
        return np.stack([self.fn(env, agent, self.rng, **self.kwargs) for env, agent in contexts]).astype(np.int64)


class DummyPolicy(Policy):
    name = "dummy"

    def act(self, obs, contexts):
        return np.zeros((len(contexts), 3), dtype=np.int64)


def make_scripted(name: str, seed: Optional[int] = None) -> Policy:
    """Build a scripted policy by name: dummy, heuristic[_p], evasion, evasion_easy."""
    if name == "dummy":
        return DummyPolicy()
    if name == "heuristic":
        return ScriptedPolicy("heuristic_1.0", heuristic_action, seed, p=1.0)
    if name.startswith("heuristic_"):
        p = float(name.split("_")[1])
        return ScriptedPolicy(name, heuristic_action, seed, p=p)
    if name == "evasion":
        return ScriptedPolicy(name, evasion_action, seed)
    if name == "evasion_easy":
        return ScriptedPolicy(name, evasion_action, seed, ast_only=True)
    raise KeyError(f"Unknown scripted policy {name!r}")


SCRIPTED_NAMES = ["dummy", "evasion_easy", "evasion", "heuristic_0.25", "heuristic_0.5",
                  "heuristic_0.75", "heuristic_1.0"]


# ------------------------------------------------------------------ network


class NetworkPolicy(Policy):
    """Batched inference with an SB3 ActorCriticPolicy (live reference or frozen copy)."""

    def __init__(self, sb3_policy, name: str = "network", deterministic: bool = False):
        self.policy = sb3_policy
        self.name = name
        self.deterministic = deterministic

    @th.no_grad()
    def act(self, obs, contexts):
        if len(obs) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32), device=self.policy.device)
        actions = self.policy._predict(obs_t, deterministic=self.deterministic)
        return actions.cpu().numpy().astype(np.int64).reshape(len(obs), -1)

    @classmethod
    def load(cls, path: str, name: Optional[str] = None, deterministic: bool = False, device="cpu"):
        from stable_baselines3 import PPO
        model = PPO.load(path, device=device)
        model.policy.set_training_mode(False)
        return cls(model.policy, name=name or path, deterministic=deterministic)


class DelayedPolicy(Policy):
    """Handicap: act on an observation from ``delay`` decision steps ago (a reaction time).

    Wraps any observation-driven policy and needs no retraining. Note this does not affect scripted
    policies, which read the live environment state directly rather than the observation.
    """

    def __init__(self, inner: Policy, delay: int):
        self.inner = inner
        self.delay = int(delay)
        self.name = f"{inner.name}+delay{delay}"
        self._buf: dict = {}

    def act(self, obs, contexts):
        if self.delay <= 0 or len(obs) == 0:
            return self.inner.act(obs, contexts)
        obs = np.asarray(obs)
        delayed = np.empty_like(obs)
        for i, (env, agent) in enumerate(contexts):
            key = (id(env), agent)
            state = self._buf.get(key)
            if state is None or env.decision_step <= state[0]:  # new episode in this env slot
                state = [env.decision_step, deque([obs[i]] * (self.delay + 1), maxlen=self.delay + 1)]
                self._buf[key] = state
            state[0] = env.decision_step
            state[1].append(obs[i])
            delayed[i] = state[1][0]
        return self.inner.act(delayed, contexts)


class HumanPolicy(Policy):
    """Keyboard control (WASD/arrows + space). Requires a pygame display.

    Without one (headless, or ``rgb_array`` recording, which draws to an offscreen surface) there is
    no keyboard to read, so this idles instead of raising.
    """
    name = "human"
    _warned = False

    def act(self, obs, contexts):
        import pygame

        if not pygame.display.get_init() or pygame.display.get_surface() is None:
            if not HumanPolicy._warned:
                HumanPolicy._warned = True
                print("HumanPolicy: no pygame display available, idling (keyboard needs a window)")
            return np.zeros((len(contexts), 3), dtype=np.int64)
        keys = pygame.key.get_pressed()
        thrust = int(keys[pygame.K_UP] or keys[pygame.K_w])
        turn = 1 if (keys[pygame.K_LEFT] or keys[pygame.K_a]) else (2 if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) else 0)
        shoot = int(keys[pygame.K_SPACE])
        return np.tile(np.array([thrust, turn, shoot]), (len(contexts), 1))


def resolve_policy(spec: str, seed: Optional[int] = None, deterministic: bool = True) -> Policy:
    """Scripted policy name, or a path to an SB3 .zip checkpoint."""
    if spec.endswith(".zip") or "/" in spec:
        return NetworkPolicy.load(spec, name=spec, deterministic=deterministic)
    if spec == "human":
        return HumanPolicy()
    return make_scripted(spec, seed)
