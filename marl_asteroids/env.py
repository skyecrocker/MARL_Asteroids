"""Team Asteroids: red vs. blue multi-agent environment (PettingZoo ParallelEnv).

Key properties
- Toroidal map, discrete actions ``MultiDiscrete([2, 3, 2])`` = (thrust, turn, shoot).
- Each agent decision is repeated for ``config.action_repeat`` physics frames.
- Episode ends when a team is eliminated (termination) or at ``config.max_frames`` (truncation).
- Observations are ego-centric (rotated into the ship's heading frame), normalized to [-1, 1],
  with k-nearest slots for enemies, allies (excluding self), asteroids, and hazardous bullets.
- Rewards are computed for *all* possible agents each step (``self.rewards_all``) so that a
  training harness can deliver the team outcome to agents that died earlier. The PettingZoo
  ``step`` return only contains agents that were alive at the start of the step.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from .config import GameConfig
from .entities import Asteroid, Bullet, Ship

RED, BLUE = 0, 1
TEAM_NAMES = ("red", "blue")

# Per-slot feature widths (see _compute_obs for the layout)
EGO_DIM = 10          # + own hull and shield fractions
BASE_DIM = 8
ENEMY_DIM = BASE_DIM + 9   # + their hull and shield fractions
ALLY_DIM = BASE_DIM + 6
ASTEROID_DIM = BASE_DIM + 1
BULLET_DIM = BASE_DIM + 2

_EPS = 1e-8
TCA_HORIZON = 90.0   # frames; time-to-closest-approach normalization
MISS_SCALE = 200.0   # px; closest-approach miss distance normalization
SHOT_SCALE = 50.0    # px; predicted shot miss distance normalization


def obs_dim(config: GameConfig) -> int:
    return (EGO_DIM
            + config.obs_enemies * ENEMY_DIM
            + config.obs_allies * ALLY_DIM
            + config.obs_asteroids * ASTEROID_DIM
            + config.obs_bullets * BULLET_DIM)


class TeamAsteroidsEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "team_asteroids_v1", "render_fps": 60}

    def __init__(self, config: GameConfig, render_mode: Optional[str] = None, **render_options):
        self.config = config
        self.render_mode = render_mode
        self.render_options = render_options  # forwarded to render.Renderer
        self.renderer = None
        self.n_red = config.n_red
        self.m_blue = config.m_blue
        self.width = float(config.width)
        self.height = float(config.height)
        self.friendly_fire = config.friendly_fire
        self.asteroids_split = config.asteroids_split

        self.possible_agents = [f"red_{i}" for i in range(self.n_red)] + \
                               [f"blue_{i}" for i in range(self.m_blue)]
        self.agent_index = {a: i for i, a in enumerate(self.possible_agents)}
        self.team_of = {a: (RED if a.startswith("red") else BLUE) for a in self.possible_agents}
        self.team_agents = {
            RED: [a for a in self.possible_agents if self.team_of[a] == RED],
            BLUE: [a for a in self.possible_agents if self.team_of[a] == BLUE],
        }

        self._obs_dim = obs_dim(config)
        self._obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
        self._act_space = gym.spaces.MultiDiscrete([2, 3, 2])
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

        self.np_random = np.random.default_rng()
        self.agents: List[str] = []
        self.ships: Dict[str, Ship] = {}
        self.asteroids: List[Asteroid] = []
        self.bullets: List[Bullet] = []
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

    # ------------------------------------------------------------------ API

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.frame = 0
        self.decision_step = 0
        self.winner: Optional[int] = None  # RED, BLUE, or -1 for draw once the game ends
        self.game_over = False
        self.kill_log: List[Tuple[str, str]] = []
        self.damage_log: List[Tuple[str, str, int, bool]] = []
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

        self._spawn_ships()
        self._spawn_asteroids()
        self.bullets = []

        self._compute_obs()
        self._phi = self._potentials()
        self.rewards_all = {a: 0.0 for a in self.possible_agents}

        observations = {a: self._obs[self.agent_index[a]].copy() for a in self.agents}
        infos = {a: {} for a in self.agents}
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def step(self, actions):
        cfg = self.config
        rcfg = cfg.rewards
        alive_before = self.agents[:]
        ind = {a: 0.0 for a in self.possible_agents}

        for _ in range(cfg.action_repeat):
            for agent in alive_before:
                if agent in actions and self.ships[agent].active:
                    self._apply_action(agent, actions[agent])
            self._step_physics()
            events = self._resolve_collisions()
            self.frame += 1
            for source, victim, damage, killed in events:
                self.damage_log.append((source, victim, damage, killed))
                friendly = source in self.ships and self.team_of[source] == self.team_of[victim]
                ind[victim] += rcfg.damage_taken * damage
                if killed:
                    self.kill_log.append((source, victim))
                    ind[victim] += rcfg.death
                if source in self.ships:
                    if friendly:
                        ind[source] += rcfg.teamdamage * damage
                        if killed:
                            ind[source] += rcfg.teamkill
                    else:
                        ind[source] += rcfg.damage_dealt * damage
                        if killed:
                            ind[source] += rcfg.kill
            if self.render_mode == "human":
                self.render()
            if self._team_alive(RED) == 0 or self._team_alive(BLUE) == 0 or self._out_of_time():
                break
        self.decision_step += 1

        red_alive, blue_alive = self._team_alive(RED), self._team_alive(BLUE)
        terminated = red_alive == 0 or blue_alive == 0
        truncated = (not terminated) and self._out_of_time()
        self.game_over = terminated or truncated

        # Observations + potential-based shaping (phi of dead agents / terminal states is 0)
        self._compute_obs()
        phi_new = self._potentials() if not self.game_over else np.zeros(len(self.possible_agents))
        for agent in alive_before:
            i = self.agent_index[agent]
            new = phi_new[i] if self.ships[agent].active else 0.0
            ind[agent] += rcfg.shaping_scale * (rcfg.shaping_gamma * new - self._phi[i])
        self._phi = phi_new

        # Team-spirit mixing over the full roster, then the shared outcome
        rewards_all = {}
        for team in (RED, BLUE):
            members = self.team_agents[team]
            if not members:
                continue
            mean = sum(ind[a] for a in members) / len(members)
            for a in members:
                rewards_all[a] = (1 - rcfg.team_spirit) * ind[a] + rcfg.team_spirit * mean
        if self.game_over:
            if terminated and red_alive == 0 and blue_alive > 0:
                self.winner = BLUE
            elif terminated and blue_alive == 0 and red_alive > 0:
                self.winner = RED
            elif truncated:
                # Timeout tiebreak: the team that won the damage exchange takes it. Turns a flat
                # zero-reward draw into a real gradient.
                red_hp, blue_hp = self.team_hp(RED), self.team_hp(BLUE)
                self.winner = RED if red_hp > blue_hp else (BLUE if blue_hp > red_hp else -1)
            else:
                self.winner = -1
            for a in self.possible_agents:
                if self.winner == -1:
                    rewards_all[a] += rcfg.draw
                elif self.team_of[a] == self.winner:
                    rewards_all[a] += rcfg.win
                else:
                    rewards_all[a] += rcfg.loss
        self.rewards_all = rewards_all
        for a, r in rewards_all.items():
            self._cumulative_rewards[a] += r

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in alive_before:
            i = self.agent_index[agent]
            dead = not self.ships[agent].active
            observations[agent] = self._obs[i].copy()
            rewards[agent] = rewards_all[agent]
            terminations[agent] = dead or terminated
            truncations[agent] = (not dead) and truncated
            infos[agent] = {}
            if self.game_over:
                infos[agent]["winner"] = self.winner

        if self.game_over:
            self.agents = []
        else:
            self.agents = [a for a in alive_before if self.ships[a].active]
        return observations, rewards, terminations, truncations, infos

    def get_obs(self, agent: str) -> np.ndarray:
        """Latest observation for an agent (zeros if dead)."""
        return self._obs[self.agent_index[agent]].copy()

    # --------------------------------------------------------------- spawning

    def _spawn_ships(self):
        cfg, rng = self.config, self.np_random
        self.ships = {}
        for agent in self.possible_agents:
            team = self.team_of[agent]
            spawns = (cfg.red_spawns if team == RED else cfg.blue_spawns) or []
            idx = int(agent.split("_")[1])
            bp = spawns[idx] if idx < len(spawns) else None
            # A blueprint with no randomness always proposes the same spot, so retrying is pointless.
            deterministic = bp is not None and not bp.random_spawn and bp.x_noise == 0 and bp.y_noise == 0

            for _attempt in range(1 if deterministic else 100):
                if bp is None or bp.random_spawn:
                    x, y = rng.uniform(0, self.width), rng.uniform(0, self.height)
                    angle = rng.uniform(0, 2 * math.pi)
                else:
                    x = rng.normal(bp.x, bp.x_noise) % self.width
                    y = rng.normal(bp.y, bp.y_noise) % self.height
                    angle = rng.normal(bp.angle, bp.angle_noise)
                if all(self._wrapped_dist((x, y), s.pos) >= cfg.min_spawn_dist for s in self.ships.values()):
                    break

            ship = Ship(x, y, team, agent, cfg, fire_rate=bp.fire_rate if bp else None)
            ship.angle = angle
            if bp is not None:
                ship.vel[:] = (rng.normal(bp.vx, bp.vx_noise), rng.normal(bp.vy, bp.vy_noise))
            self.ships[agent] = ship

    def _spawn_asteroids(self):
        cfg, rng = self.config, self.np_random
        self.asteroids = []
        for bp in cfg.asteroid_spawns or []:
            x = rng.normal(bp.x, bp.x_noise) % self.width
            y = rng.normal(bp.y, bp.y_noise) % self.height
            vx, vy = rng.normal(bp.vx, bp.vx_noise), rng.normal(bp.vy, bp.vy_noise)
            self.asteroids.append(Asteroid(x, y, vx, vy, cfg, size=bp.size))
        for _ in range(max(0, cfg.num_asteroids - len(cfg.asteroid_spawns or []))):
            for _attempt in range(200):
                ax, ay = rng.uniform(0, self.width), rng.uniform(0, self.height)
                if all(self._wrapped_dist((ax, ay), s.pos) >= cfg.asteroid_clear_radius
                       for s in self.ships.values()):
                    break
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(*cfg.asteroid_speed_range)
            self.asteroids.append(Asteroid(ax, ay, math.cos(angle) * speed, math.sin(angle) * speed, cfg, size=3))

    # ---------------------------------------------------------------- physics

    def _apply_action(self, agent_name, action):
        cfg = self.config
        ship = self.ships[agent_name]
        thrust, turn, shoot = int(action[0]), int(action[1]), int(action[2])
        ship.thrusting = thrust == 1

        if turn == 1:
            ship.angle -= cfg.turn_speed
        elif turn == 2:
            ship.angle += cfg.turn_speed
        ship.angle = (ship.angle + math.pi) % (2 * math.pi) - math.pi

        if thrust == 1:
            ship.vel[0] += math.cos(ship.angle) * cfg.thrust_power
            ship.vel[1] += math.sin(ship.angle) * cfg.thrust_power
            speed = math.hypot(ship.vel[0], ship.vel[1])
            if speed > cfg.max_speed:
                ship.vel *= cfg.max_speed / speed

        if shoot == 1 and ship.cooldown <= 0:
            c, s = math.cos(ship.angle), math.sin(ship.angle)
            bx = ship.pos[0] + c * (ship.radius + 5)
            by = ship.pos[1] + s * (ship.radius + 5)
            bvx = ship.vel[0] + c * cfg.bullet_speed
            bvy = ship.vel[1] + s * cfg.bullet_speed
            self.bullets.append(Bullet(bx, by, bvx, bvy, agent_name, ship.team_id, cfg))
            ship.cooldown = ship.fire_rate

    def _step_physics(self):
        drag = self.config.drag_coefficient
        size = np.array([self.width, self.height])
        for ship in self.ships.values():
            if not ship.active:
                continue
            ship.pos += ship.vel
            ship.vel *= drag
            ship.pos %= size
            if ship.cooldown > 0:
                ship.cooldown -= 1
            ship.tick_shield(self.config.shield_regen_delay, self.config.shield_regen_frames)
        for ast in self.asteroids:
            if ast.active:
                ast.pos += ast.vel
                ast.pos %= size
        alive = []
        for b in self.bullets:
            b.pos += b.vel
            b.pos %= size
            b.lifespan -= 1
            if b.lifespan > 0 and b.active:
                alive.append(b)
        self.bullets = alive

    def _separate(self, ship, delta, dist, overlap, push_both: bool, other=None):
        """Push a contact apart along its normal and reflect the inbound velocity.

        Without this, bodies stay overlapped and the contact re-triggers every frame.
        """
        # n points from ``ship`` toward the thing it hit.
        n = np.array([delta[0] / dist, delta[1] / dist]) if dist > 1e-9 else np.array([1.0, 0.0])
        gap = (overlap - dist) + 1.0
        rest = self.config.bounce_restitution
        size = np.array([self.width, self.height])
        share = 0.5 if push_both else 1.0

        def kick(s, direction):
            """direction = -1 moves along -n (away, for ``ship``), +1 along +n (for ``other``)."""
            s.pos = (s.pos + n * gap * share * direction) % size
            vn = float(s.vel @ n)
            if vn * direction < 0:  # only reflect the component closing on the contact
                s.vel -= (1 + rest) * vn * n

        kick(ship, -1.0)
        if push_both and other is not None:
            kick(other, 1.0)

    def _resolve_collisions(self) -> List[Tuple[str, str, int, bool]]:
        """Apply damage from all collisions.

        Returns ``(source, victim, damage, killed)`` per hit. ``source`` is an agent name for
        weapon fire, or "asteroid"/"ram" for environmental damage.
        """
        events: List[Tuple[str, str, int, bool]] = []
        cfg = self.config
        new_asteroids = []

        def hit(source: str, victim: str, amount: int):
            ship = self.ships[victim]
            dealt = ship.apply_damage(amount)
            if dealt:
                events.append((source, victim, dealt, not ship.active))

        # Bullets vs ships
        for b in self.bullets:
            if not b.active:
                continue
            for name, ship in self.ships.items():
                if not ship.active or b.owner_name == name:
                    continue
                if not self.friendly_fire and b.team_id == ship.team_id:
                    continue
                if self._wrapped_dist(b.pos, ship.pos) < b.radius + ship.radius:
                    b.active = False
                    hit(b.owner_name, name, cfg.bullet_damage)
                    break

        # Bullets vs asteroids
        for b in self.bullets:
            if not b.active:
                continue
            for ast in self.asteroids:
                if not ast.active:
                    continue
                if self._wrapped_dist(b.pos, ast.pos) < b.radius + ast.radius:
                    b.active = False
                    if self.asteroids_split:
                        ast.active = False
                        if ast.size > 1:
                            heading = math.atan2(ast.vel[1], ast.vel[0])
                            speed = max(math.hypot(ast.vel[0], ast.vel[1]) * self.config.asteroid_split_speedup,
                                        self.config.asteroid_split_min_speed)
                            spread = self.config.asteroid_split_spread
                            for off in (spread, -spread):
                                new_asteroids.append(Asteroid(
                                    ast.pos[0], ast.pos[1],
                                    math.cos(heading + off) * speed, math.sin(heading + off) * speed,
                                    self.config, size=ast.size - 1))
                    break

        # Ships vs asteroids: bounce off and take damage, gated by the collision cooldown so a
        # lingering overlap cannot deal damage every frame.
        for name, ship in self.ships.items():
            if not ship.active:
                continue
            for ast in self.asteroids:
                if not ast.active:
                    continue
                dx, dy, dist = self.wrapped_delta(ship.pos, ast.pos)
                overlap = ship.radius + ast.radius
                if dist < overlap:
                    self._separate(ship, (dx, dy), dist, overlap, push_both=False)
                    if ship.collision_cooldown <= 0:
                        ship.collision_cooldown = cfg.collision_damage_cooldown
                        hit("asteroid", name, cfg.asteroid_damage)
                    break

        # Ships vs ships (ramming damages both, no kill credit to either)
        names = list(self.ships)
        for i, na in enumerate(names):
            sa = self.ships[na]
            if not sa.active:
                continue
            for nb in names[i + 1:]:
                sb = self.ships[nb]
                if not sb.active:
                    continue
                dx, dy, dist = self.wrapped_delta(sa.pos, sb.pos)
                overlap = sa.radius + sb.radius
                if dist < overlap:
                    self._separate(sa, (dx, dy), dist, overlap, push_both=True, other=sb)
                    if sa.collision_cooldown <= 0 and sb.collision_cooldown <= 0:
                        sa.collision_cooldown = cfg.collision_damage_cooldown
                        sb.collision_cooldown = cfg.collision_damage_cooldown
                        hit("ram", na, cfg.ram_damage)
                        hit("ram", nb, cfg.ram_damage)
                    break

        if self.asteroids_split:
            self.asteroids = [a for a in self.asteroids if a.active] + new_asteroids
        return events

    # --------------------------------------------------------------- geometry

    def _wrap(self, d: np.ndarray) -> np.ndarray:
        """Wrap displacement(s) (..., 2) to the shortest toroidal image."""
        size = np.array([self.width, self.height])
        return (d + size / 2) % size - size / 2

    def _wrapped_dist(self, a, b) -> float:
        dx = (b[0] - a[0] + self.width / 2) % self.width - self.width / 2
        dy = (b[1] - a[1] + self.height / 2) % self.height - self.height / 2
        return math.hypot(dx, dy)

    def wrapped_delta(self, ego_pos, target_pos):
        """Shortest (dx, dy, dist) from ego to target on the torus."""
        dx = (target_pos[0] - ego_pos[0] + self.width / 2) % self.width - self.width / 2
        dy = (target_pos[1] - ego_pos[1] + self.height / 2) % self.height - self.height / 2
        return dx, dy, math.hypot(dx, dy)

    # Backwards-compatible name used by the scripted policies
    _get_relative_wrapped_distance = wrapped_delta

    def _team_alive(self, team: int) -> int:
        return sum(1 for a in self.team_agents[team] if self.ships[a].active)

    def team_hp(self, team: int) -> int:
        """Total remaining hull + shield across a team (used for the timeout tiebreak)."""
        return sum(self.ships[a].hp for a in self.team_agents[team] if self.ships[a].active)

    def _out_of_time(self) -> bool:
        return self.config.max_frames is not None and self.frame >= self.config.max_frames

    # ----------------------------------------------------------- observations

    def _compute_obs(self):
        """Compute observations for every possible agent in one vectorized pass.

        Layout per agent:
          ego (8): alive, v_fwd, v_lat, cooldown_frac, can_shoot, allies_alive_frac,
                   enemies_alive_frac, time_left_frac
          enemies (k x 15), allies (k x 12), asteroids (k x 9), bullets (k x 10)
          base slot features (8): present, bearing_x, bearing_y (unit vector in the ego frame),
                   range (dist / (dist + sensor_range)), rel_vx, rel_vy (ego frame),
                   tca, closest_approach_miss
          enemy extras: lead-aim error (cos, sin), intercept time, predicted miss if I fire now,
                   their aim at me (cos), their predicted miss on me if they fire, their cooldown
          ally extras: their heading in my frame (cos, sin), my friendly-fire miss on them, cooldown
          asteroid extras: radius
          bullet extras: hostile flag, remaining life
        """
        cfg = self.config
        S = len(self.possible_agents)
        ships = [self.ships[a] for a in self.possible_agents]
        L = cfg.sensor_range
        vref = 2 * cfg.max_speed + cfg.bullet_speed

        P = np.array([s.pos for s in ships])                   # (S,2)
        V = np.array([s.vel for s in ships])
        A = np.array([s.angle for s in ships])
        team = np.array([s.team_id for s in ships])
        alive = np.array([s.active for s in ships])
        cd = np.array([s.cooldown / max(1, s.fire_rate) for s in ships])
        rad = np.array([s.radius for s in ships])
        hull_f = np.array([s.hull / max(1, s.max_hull) for s in ships])
        shield_f = np.array([s.shield / max(1, s.max_shield) if s.max_shield else 0.0 for s in ships])
        c, s_ = np.cos(A)[:, None], np.sin(A)[:, None]         # (S,1)
        H = np.stack([np.cos(A), np.sin(A)], axis=1)           # (S,2)

        def rot(vec):  # (S,N,2) world -> ego frame of each row's ego
            return np.stack([c * vec[..., 0] + s_ * vec[..., 1],
                             -s_ * vec[..., 0] + c * vec[..., 1]], axis=-1)

        def base(Pt, Vt, rt):
            D = self._wrap(Pt[None, :, :] - P[:, None, :])     # (S,N,2)
            RV = Vt[None, :, :] - V[:, None, :]
            dist = np.linalg.norm(D, axis=-1)
            rv2 = (RV ** 2).sum(-1)
            tca = np.clip(-(D * RV).sum(-1) / (rv2 + _EPS), 0.0, TCA_HORIZON)
            miss = np.linalg.norm(D + RV * tca[..., None], axis=-1) - (rad[:, None] + rt[None, :])
            Dr, RVr = rot(D), rot(RV)
            # Bearing as a unit vector plus a saturating range, rather than dx/L and dist/L clipped
            # to [-1, 1]. Clipping components independently destroys bearing past the normalization
            # scale (everything far reads as (1, 1)), which made large maps unnavigable. This form
            # keeps bearing exact at any distance, and range stays monotone: 0.5 at sensor_range,
            # approaching 1 far away, never fully saturated.
            inv = 1.0 / (dist + _EPS)
            feats = np.stack([
                np.ones_like(dist),
                Dr[..., 0] * inv, Dr[..., 1] * inv, dist / (dist + L),
                RVr[..., 0] / vref, RVr[..., 1] / vref,
                tca / TCA_HORIZON, np.clip(miss / MISS_SCALE, -1, 1),
            ], axis=-1)
            return D, RV, dist, tca, miss, feats

        def shot_miss(Pt, Vt, rt):
            """Predicted closest-approach miss (px) of a bullet fired now by each ego at each target."""
            b0 = P + H * (rad[:, None] + 5)
            vb = V + H * cfg.bullet_speed
            p = self._wrap(Pt[None] - b0[:, None])
            w = Vt[None] - vb[:, None]
            t = np.clip(-(p * w).sum(-1) / ((w ** 2).sum(-1) + _EPS), 0.0, cfg.bullet_lifespan)
            return np.linalg.norm(p + w * t[..., None], axis=-1) - (rt[None] + cfg.bullet_radius)

        rows = np.arange(S)[:, None]

        def select(key, k):
            """Indices of the k smallest keys per row (inf = invalid) and a presence mask."""
            n = key.shape[1]
            order = np.zeros((S, k), dtype=np.intp)
            present = np.zeros((S, k), dtype=bool)
            if n == 0:
                return order, present
            m = min(n, k)
            o = np.argsort(key, axis=1)[:, :m]
            order[:, :m] = o
            present[:, :m] = np.isfinite(key[rows, o])
            return order, present

        def gather(feats, order, present):
            out = feats[rows, order]
            out[~present] = 0.0
            return out.reshape(S, -1)

        blocks = []

        # Ego
        Vr = np.stack([c[:, 0] * V[:, 0] + s_[:, 0] * V[:, 1], -s_[:, 0] * V[:, 0] + c[:, 0] * V[:, 1]], 1)
        red_alive = alive[team == RED].sum()
        blue_alive = alive[team == BLUE].sum()
        own_alive = np.where(team == RED, red_alive, blue_alive)
        opp_alive = np.where(team == RED, blue_alive, red_alive)
        own_n = np.where(team == RED, self.n_red, self.m_blue)
        opp_n = np.where(team == RED, self.m_blue, self.n_red)
        # No time limit -> no time pressure to represent, so the clock feature is pinned at full.
        time_left = 1.0 if cfg.max_frames is None else 1.0 - self.frame / cfg.max_frames
        ego = np.stack([
            np.ones(S), Vr[:, 0] / cfg.max_speed, Vr[:, 1] / cfg.max_speed, cd,
            np.array([s.cooldown <= 0 for s in ships], dtype=float),
            (own_alive - 1) / np.maximum(1, own_n - 1), opp_alive / np.maximum(1, opp_n),
            np.full(S, time_left), hull_f, shield_f,
        ], axis=1)
        blocks.append(ego)

        # Ships (enemies and allies share pairwise computations)
        D, RV, dist, tca, miss, feats = base(P, V, rad)
        smiss = shot_miss(P, V, rad)                           # (S,S) ego i fires at j
        # Lead-aim intercept: |D + RV t| = bullet_speed * t
        a = (RV ** 2).sum(-1) - cfg.bullet_speed ** 2
        b = 2 * (D * RV).sum(-1)
        cc = (D ** 2).sum(-1)
        disc = b * b - 4 * a * cc
        sq = np.sqrt(np.maximum(disc, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(np.abs(a) > _EPS, (-b - sq) / (2 * a), -cc / np.where(np.abs(b) > _EPS, b, -_EPS))
            t2 = np.where(np.abs(a) > _EPS, (-b + sq) / (2 * a), t1)
        t1 = np.where(t1 > 0, t1, np.inf)
        t2 = np.where(t2 > 0, t2, np.inf)
        t_int = np.minimum(t1, t2)
        valid = (disc >= 0) & np.isfinite(t_int)
        aim_vec = np.where(valid[..., None], D + RV * np.where(valid, t_int, 0)[..., None], D)
        aim_r = rot(aim_vec)
        aim_n = np.linalg.norm(aim_r, axis=-1) + _EPS
        aim_cos, aim_sin = aim_r[..., 0] / aim_n, aim_r[..., 1] / aim_n
        t_int_f = np.where(valid, np.clip(t_int / cfg.bullet_lifespan, 0, 1), 1.0)
        dir_cos = (H[:, None, :] * D).sum(-1) / (dist + _EPS)  # my heading vs direction to j
        their_aim = dir_cos.T                                  # j's heading vs direction to me
        their_smiss = smiss.T
        Hr = rot(np.broadcast_to(H[None], (S, S, 2)))

        # Target durability is broadcast along the target axis, so slot k reports that ship's health.
        # Visible health is what makes focusing fire on a weakened enemy learnable.
        hull_t = np.broadcast_to(hull_f[None], (S, S))
        shield_t = np.broadcast_to(shield_f[None], (S, S))
        enemy_feats = np.concatenate([feats, np.stack([
            aim_cos, aim_sin, t_int_f, np.clip(smiss / SHOT_SCALE, -1, 1),
            their_aim, np.clip(their_smiss / SHOT_SCALE, -1, 1), np.broadcast_to(cd[None], (S, S)),
            hull_t, shield_t,
        ], -1)], -1)
        ally_feats = np.concatenate([feats, np.stack([
            Hr[..., 0], Hr[..., 1], np.clip(smiss / SHOT_SCALE, -1, 1), np.broadcast_to(cd[None], (S, S)),
            hull_t, shield_t,
        ], -1)], -1)

        eye = np.eye(S, dtype=bool)
        both_alive = alive[:, None] & alive[None, :]
        same_team = team[:, None] == team[None, :]
        enemy_key = np.where(both_alive & ~same_team, dist, np.inf)
        ally_key = np.where(both_alive & same_team & ~eye, dist, np.inf)
        e_ord, e_pres = select(enemy_key, cfg.obs_enemies)
        a_ord, a_pres = select(ally_key, cfg.obs_allies)
        blocks.append(gather(enemy_feats, e_ord, e_pres))
        blocks.append(gather(ally_feats, a_ord, a_pres))

        # Asteroids
        asts = [x for x in self.asteroids if x.active]
        if asts:
            Pa = np.array([x.pos for x in asts]); Va = np.array([x.vel for x in asts])
            ra = np.array([x.radius for x in asts])
            _, _, adist, atca, amiss, afeats = base(Pa, Va, ra)
            afeats = np.concatenate([afeats, np.broadcast_to((ra / cfg.asteroid_large_radius)[None, :, None],
                                                             (S, len(asts), 1))], -1)
            akey = np.where(alive[:, None], adist, np.inf)
        else:
            adist = atca = amiss = np.zeros((S, 0))
            afeats = np.zeros((S, 0, ASTEROID_DIM)); akey = np.zeros((S, 0))
        o, p = select(akey, cfg.obs_asteroids)
        blocks.append(gather(afeats, o, p) if asts else np.zeros((S, cfg.obs_asteroids * ASTEROID_DIM)))

        # Bullets that can hurt each ego (never its own)
        bl = [x for x in self.bullets if x.active]
        if bl:
            Pb = np.array([x.pos for x in bl]); Vb = np.array([x.vel for x in bl])
            rb = np.array([x.radius for x in bl])
            bteam = np.array([x.team_id for x in bl])
            owner = np.array([self.agent_index.get(x.owner_name, -1) for x in bl])
            _, _, bdist, btca, bmiss, bfeats = base(Pb, Vb, rb)
            hostile = (bteam[None, :] != team[:, None]).astype(float)
            life = np.broadcast_to(np.array([x.lifespan / cfg.bullet_lifespan for x in bl])[None], hostile.shape)
            bfeats = np.concatenate([bfeats, np.stack([hostile, life], -1)], -1)
            can_hurt = (hostile > 0) | self.friendly_fire
            can_hurt &= owner[None, :] != np.arange(S)[:, None]
            bkey = np.where(alive[:, None] & can_hurt, bdist, np.inf)
            o, p = select(bkey, cfg.obs_bullets)
            blocks.append(gather(bfeats, o, p))
        else:
            bmiss = btca = np.zeros((S, 0)); bkey = np.zeros((S, 0))
            blocks.append(np.zeros((S, cfg.obs_bullets * BULLET_DIM)))

        obs = np.clip(np.concatenate(blocks, axis=1), -1.0, 1.0).astype(np.float32)
        obs[~alive] = 0.0
        self._obs = obs

        # Cache shaping primitives
        self._cache = dict(
            alive=alive, enemy_key=enemy_key, aim_cos=aim_cos, dist=dist,
            ship_tca=tca, ship_miss=miss,
            ast_key=akey, ast_tca=atca, ast_miss=amiss,
            bul_key=bkey, bul_tca=btca, bul_miss=bmiss,
        )

    def _potentials(self) -> np.ndarray:
        """Shaping potential per agent: aim at + stay in range of nearest enemy, avoid imminent hits."""
        rc = self.config.rewards
        k = self._cache
        S = len(self.possible_agents)
        phi = np.zeros(S)
        for i in range(S):
            if not k["alive"][i]:
                continue
            row = k["enemy_key"][i]
            if np.isfinite(row).any():
                j = int(np.argmin(row))
                aim_q = (1.0 + k["aim_cos"][i, j]) / 2.0
                range_q = float(np.clip(
                    1.0 - max(0.0, k["dist"][i, j] - rc.engage_range) / rc.engage_falloff, 0.0, 1.0))
                phi[i] += rc.aim_weight * aim_q ** 2 + rc.approach_weight * range_q
            threat = 0.0
            for key, tca, miss in (("ast_key", "ast_tca", "ast_miss"), ("bul_key", "bul_tca", "bul_miss")):
                valid = np.isfinite(k[key][i])
                if valid.any():
                    t = (1.0 - np.clip(k[miss][i][valid], 0, 60) / 60.0) * (1.0 - k[tca][i][valid] / TCA_HORIZON)
                    threat = max(threat, float(t.max()))
            phi[i] -= rc.hazard_weight * threat
        return phi

    # -------------------------------------------------------------- rendering

    def render(self):
        """Delegate to marl_asteroids.render.Renderer (kept out of the env; purely cosmetic)."""
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from .render import Renderer
            self.renderer = Renderer(self, **self.render_options)
        return self.renderer.render(self.render_mode, fps=self.metadata["render_fps"])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
