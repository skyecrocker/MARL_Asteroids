"""Pygame renderer for TeamAsteroidsEnv.

Kept separate from the environment so visual polish never affects simulation or training. The env
holds no drawing state; it just forwards render() here. Purely cosmetic effects (thruster flames,
muzzle flashes, explosions, starfield, trails) live in the renderer.

The view scales the world to fit the window, so large maps still fit on screen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .env import BLUE, RED

# Palette
BG_TOP = (6, 8, 20)
BG_BOTTOM = (16, 12, 32)
TEAM_COLORS = {
    RED: {"hull": (255, 96, 88), "glow": (255, 40, 40), "trail": (120, 30, 30)},
    BLUE: {"hull": (96, 176, 255), "glow": (40, 140, 255), "trail": (26, 70, 120)},
}
ROCK_FILL = (74, 70, 86)
ROCK_EDGE = (150, 146, 168)
BULLET_CORE = (255, 255, 235)
HUD_TEXT = (226, 232, 245)
HUD_DIM = (120, 132, 156)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    radius: float
    color: Tuple[int, int, int]
    kind: str = "spark"


@dataclass
class RenderState:
    """Cosmetic state that persists across frames (owned by the renderer, not the env)."""
    particles: List[Particle] = field(default_factory=list)
    trails: dict = field(default_factory=dict)
    stars: Optional[np.ndarray] = None
    shake: float = 0.0
    last_frame: int = -1
    prev_bullets: int = 0
    dead_seen: set = field(default_factory=set)


class Renderer:
    def __init__(self, env, window: int = 1000, show_hud: bool = True, effects: bool = True,
                 trail_length: int = 18):
        self.env = env
        self.window = window
        self.show_hud = show_hud
        self.effects = effects
        self.trail_length = trail_length
        self.scale = window / max(env.width, env.height)
        self.surface = None
        self.screen = None
        self.clock = None
        self.state = RenderState()
        self.rng = np.random.default_rng(0)
        self._rock_shapes: dict = {}

    # ------------------------------------------------------------------ setup

    def _ensure_init(self, mode: str):
        import pygame

        if self.surface is not None:
            return
        pygame.init()
        pygame.font.init()
        size = (int(self.env.width * self.scale), int(self.env.height * self.scale))
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("MARL Asteroids")
            self.screen = pygame.display.set_mode(size)
            self.surface = self.screen
        else:
            self.surface = pygame.Surface(size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 15)
        self.font_big = pygame.font.SysFont("dejavusansmono,consolas,monospace", 30, bold=True)
        self.bg = self._make_background(size)

    def _make_background(self, size):
        import pygame

        w, h = size
        bg = pygame.Surface(size)
        for y in range(h):  # vertical gradient
            t = y / max(1, h - 1)
            bg.fill(tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)), (0, y, w, 1))
        n = int(w * h / 4200)
        pts = self.rng.random((n, 2)) * (w, h)
        mag = self.rng.random(n)
        for (x, y), m in zip(pts, mag):
            v = int(40 + 150 * m ** 3)
            r = 2 if m > 0.97 else 1
            pygame.draw.circle(bg, (v, v, min(255, int(v * 1.1))), (int(x), int(y)), r)
        return bg

    def _rock_shape(self, rock):
        """Stable irregular polygon per asteroid (cached by identity)."""
        key = id(rock)
        shape = self._rock_shapes.get(key)
        if shape is None or shape[0] != rock.radius:
            rng = np.random.default_rng(key % (2 ** 31))
            n = 11
            ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
            rad = rock.radius * (0.78 + 0.32 * rng.random(n))
            craters = [(rock.radius * (rng.random() - 0.5) * 1.0,
                        rock.radius * (rng.random() - 0.5) * 1.0,
                        rock.radius * (0.1 + 0.16 * rng.random())) for _ in range(3)]
            shape = (rock.radius, np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1),
                     (rng.random() - 0.5) * 0.02, craters)
            self._rock_shapes[key] = shape
        return shape

    # ----------------------------------------------------------------- effects

    def _spawn_burst(self, x, y, n, speed, color, life, radius, kind="spark"):
        ang = self.rng.random(n) * 2 * math.pi
        mag = speed * (0.25 + 0.75 * self.rng.random(n))
        for a, m in zip(ang, mag):
            self.state.particles.append(Particle(
                x, y, math.cos(a) * m, math.sin(a) * m,
                life, life * (0.6 + 0.4 * self.rng.random()), radius, color, kind))

    def _detect_events(self):
        """Infer cosmetic events from env state (deaths, new bullets) without touching the env."""
        env, st = self.env, self.state
        for name, ship in env.ships.items():
            if not ship.active and name not in st.dead_seen:
                st.dead_seen.add(name)
                c = TEAM_COLORS[ship.team_id]
                self._spawn_burst(ship.pos[0], ship.pos[1], 34, 3.4, c["glow"], 26, 3.0, "spark")
                self._spawn_burst(ship.pos[0], ship.pos[1], 16, 1.6, (255, 226, 150), 20, 5.0, "smoke")
                st.shake = min(16.0, st.shake + 9.0)
            elif ship.active and name in st.dead_seen:
                st.dead_seen.discard(name)  # new episode
        n = len(env.bullets)
        if n > st.prev_bullets:
            for b in env.bullets[st.prev_bullets:n]:
                c = TEAM_COLORS[b.team_id]
                self._spawn_burst(b.pos[0], b.pos[1], 5, 1.1, c["glow"], 9, 2.0)
        st.prev_bullets = n

    def _update_particles(self):
        st = self.state
        alive = []
        for p in st.particles:
            p.x += p.vx
            p.y += p.vy
            p.vx *= 0.94
            p.vy *= 0.94
            p.life -= 1
            if p.life > 0:
                alive.append(p)
        st.particles = alive
        st.shake *= 0.85

    def _update_trails(self):
        st = self.env, self.state
        env, st = st
        for name, ship in env.ships.items():
            t = st.trails.setdefault(name, [])
            if ship.active:
                t.append((float(ship.pos[0]), float(ship.pos[1])))
                if len(t) > self.trail_length:
                    del t[0]
            elif t:
                del t[0]

    # ------------------------------------------------------------------- draw

    def render(self, mode: str = "human", fps: int = 60):
        import pygame

        self._ensure_init(mode)
        env, st = self.env, self.state
        if env.frame < st.last_frame:  # episode restarted
            st.particles.clear()
            st.trails.clear()
            st.dead_seen.clear()
            st.prev_bullets = 0
            st.shake = 0.0
        st.last_frame = env.frame

        if self.effects:
            self._detect_events()
            self._update_particles()
            self._update_trails()

        s = self.scale
        ox = oy = 0.0
        if self.effects and st.shake > 0.3:
            ox, oy = (self.rng.random(2) - 0.5) * st.shake
        surf = self.surface
        surf.blit(self.bg, (int(ox * 0.3), int(oy * 0.3)))

        def to_px(pos):
            return int(pos[0] * s + ox), int(pos[1] * s + oy)

        # Ship trails
        if self.effects:
            for name, pts in st.trails.items():
                if len(pts) < 2:
                    continue
                col = TEAM_COLORS[env.ships[name].team_id]["trail"]
                for i in range(1, len(pts)):
                    if abs(pts[i][0] - pts[i - 1][0]) > env.width / 2 or \
                       abs(pts[i][1] - pts[i - 1][1]) > env.height / 2:
                        continue  # skip the segment that wraps across the map edge
                    f = i / len(pts)
                    pygame.draw.line(surf, tuple(int(c * f) for c in col),
                                     to_px(pts[i - 1]), to_px(pts[i]), max(1, int(2.6 * f * s)))

        # Asteroids
        for rock in env.asteroids:
            if not rock.active:
                continue
            _, poly, spin, craters = self._rock_shape(rock)
            th = env.frame * spin
            rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            pts = [to_px(rock.pos + p) for p in poly @ rot.T]
            pygame.draw.polygon(surf, ROCK_FILL, pts)
            for cx, cy, cr in craters:  # craters for texture, rotating with the rock
                rx = cx * math.cos(th) - cy * math.sin(th)
                ry = cx * math.sin(th) + cy * math.cos(th)
                pygame.draw.circle(surf, (58, 55, 68), to_px((rock.pos[0] + rx, rock.pos[1] + ry)),
                                   max(1, int(cr * s)))
            pygame.draw.polygon(surf, ROCK_EDGE, pts, max(1, int(1.8 * s)))

        # Particles (under ships, over rocks)
        if self.effects:
            for p in st.particles:
                f = max(0.0, p.life / p.max_life)
                r = max(1, int(p.radius * s * (1.6 - f if p.kind == "smoke" else f)))
                col = tuple(min(255, int(c * (f ** 0.6))) for c in p.color)
                pygame.draw.circle(surf, col, to_px((p.x, p.y)), r)

        # Bullets: bright core with a tapering glow tail so incoming fire is easy to read
        for b in env.bullets:
            if not b.active:
                continue
            glow = TEAM_COLORS[b.team_id]["glow"]
            px = to_px(b.pos)
            for k, (mult, fade) in enumerate(((4.0, 0.25), (2.2, 0.55))):
                tail = (b.pos[0] - b.vel[0] * mult, b.pos[1] - b.vel[1] * mult)
                pygame.draw.line(surf, tuple(int(c * fade) for c in glow), to_px(tail), px,
                                 max(1, int((4 - k) * s)))
            pygame.draw.circle(surf, glow, px, max(3, int(b.radius * s * 2.0)))
            pygame.draw.circle(surf, BULLET_CORE, px, max(2, int(b.radius * s * 1.1)))

        # Ships
        for name, ship in env.ships.items():
            if not ship.active:
                continue
            col = TEAM_COLORS[ship.team_id]
            a, r = ship.angle, ship.radius
            hull = [to_px(ship.pos + (math.cos(a + o) * r * m, math.sin(a + o) * r * m))
                    for o, m in ((0.0, 1.45), (2.45, 1.0), (math.pi, 0.45), (-2.45, 1.0))]
            if self.effects and ship.thrusting:  # thruster flame opposite the heading
                flick = 0.75 + 0.5 * self.rng.random()
                base = ship.pos + (math.cos(a + math.pi) * r * 0.5, math.sin(a + math.pi) * r * 0.5)
                tip = ship.pos + (math.cos(a + math.pi) * r * (1.0 + 1.5 * flick),
                                  math.sin(a + math.pi) * r * (1.0 + 1.5 * flick))
                side = [ship.pos + (math.cos(a + math.pi + o) * r * 0.5, math.sin(a + math.pi + o) * r * 0.5)
                        for o in (0.85, -0.85)]
                pygame.draw.polygon(surf, (255, int(150 + 80 * flick), 60),
                                    [to_px(side[0]), to_px(tip), to_px(side[1]), to_px(base)])
            pygame.draw.polygon(surf, col["hull"], hull)
            # Bright outline while armed, dim while reloading: readable weapon state at a glance.
            armed = ship.cooldown <= 0
            edge = (250, 250, 255) if armed else tuple(int(c * 0.45) for c in col["hull"])
            pygame.draw.polygon(surf, edge, hull, max(1, int(2.0 * s)))
            if not armed:  # reload arc shrinking toward the nose
                frac = 1.0 - ship.cooldown / max(1, ship.fire_rate)
                nose = ship.pos + (math.cos(a) * r * 1.45, math.sin(a) * r * 1.45)
                pygame.draw.circle(surf, tuple(int(c * frac) for c in col["glow"]),
                                   to_px(nose), max(1, int(2.6 * s * frac)))

        if self.show_hud:
            self._draw_hud(surf)

        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

    def _draw_hud(self, surf):
        env = self.env
        red, blue = env._team_alive(RED), env._team_alive(BLUE)
        surf.blit(self.font.render(f"RED {red}", True, TEAM_COLORS[RED]["hull"]), (14, 12))
        label = self.font.render(f"BLUE {blue}", True, TEAM_COLORS[BLUE]["hull"])
        surf.blit(label, (surf.get_width() - label.get_width() - 14, 12))
        t = self.font.render(f"{env.frame}/{env.config.max_frames}", True, HUD_DIM)
        surf.blit(t, ((surf.get_width() - t.get_width()) // 2, 12))
        bar_w = surf.get_width() - 28
        frac = min(1.0, env.frame / max(1, env.config.max_frames))
        import pygame
        pygame.draw.rect(surf, (40, 44, 60), (14, 34, bar_w, 3))
        pygame.draw.rect(surf, HUD_DIM, (14, 34, int(bar_w * frac), 3))
        if env.game_over:
            msg = {BLUE: "BLUE WINS", RED: "RED WINS"}.get(env.winner, "DRAW")
            color = TEAM_COLORS.get(env.winner, {}).get("hull", HUD_TEXT)
            surface = self.font_big.render(msg, True, color)
            surf.blit(surface, ((surf.get_width() - surface.get_width()) // 2,
                                (surf.get_height() - surface.get_height()) // 2))

    def close(self):
        import pygame
        if self.surface is not None:
            pygame.quit()
            self.surface = None
            self.screen = None
