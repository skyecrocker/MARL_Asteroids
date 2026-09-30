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
                 trail_length: int = 18, follow: Optional[str] = None, view_size: float = 1200.0,
                 minimap: bool = True):
        """
        Args:
            follow: agent name to keep centered. None renders the whole map scaled to the window.
            view_size: world units across the viewport in follow mode (clamped to the map size).
            minimap: draw a whole-map inset in follow mode.
        """
        self.env = env
        self.window = window
        self.show_hud = show_hud
        self.effects = effects
        self.trail_length = trail_length
        self.follow = follow
        self.minimap = minimap
        map_span = max(env.width, env.height)
        if follow is None:
            self.view_size = map_span
            self.scale = window / map_span
        else:
            # Showing more than the map would draw empty space: each object appears once.
            self.view_size = min(view_size, map_span)
            self.scale = window / self.view_size
        self.surface = None
        self.screen = None
        self.clock = None
        self.state = RenderState()
        self.rng = np.random.default_rng(0)
        self._rock_shapes: dict = {}
        self._cam = None  # last camera center, kept when the followed ship dies

    # ------------------------------------------------------------------ setup

    def _ensure_init(self, mode: str):
        import pygame

        if self.surface is not None:
            return
        pygame.init()
        pygame.font.init()
        if self.follow is None:
            size = (int(self.env.width * self.scale), int(self.env.height * self.scale))
        else:
            size = (self.window, self.window)
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
        self.font_small = pygame.font.SysFont("dejavusansmono,consolas,monospace", 11)
        self.bg = self._make_background(size)
        self.stars = self._make_stars(size)  # tileable layer for the scrolling camera

    def _make_background(self, size):
        import pygame

        w, h = size
        bg = pygame.Surface(size)
        for y in range(h):  # vertical gradient
            t = y / max(1, h - 1)
            bg.fill(tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)), (0, y, w, 1))
        bg.blit(self._make_stars(size), (0, 0))
        return bg

    def _make_stars(self, size):
        """Star layer with black as the transparent key, so it can be tiled while scrolling."""
        import pygame

        w, h = size
        surf = pygame.Surface(size)
        surf.fill((0, 0, 0))
        rng = np.random.default_rng(12345)
        n = int(w * h / 4200)
        for (x, y), m in zip(rng.random((n, 2)) * (w, h), rng.random(n)):
            v = int(40 + 150 * m ** 3)
            pygame.draw.circle(surf, (v, v, min(255, int(v * 1.1))), (int(x), int(y)),
                               2 if m > 0.97 else 1)
        surf.set_colorkey((0, 0, 0))
        return surf

    # ------------------------------------------------------------------ camera

    def _camera_center(self):
        """World point to keep centered; holds the last position once the followed ship dies."""
        if self.follow is None:
            return np.array([self.env.width / 2, self.env.height / 2])
        ship = self.env.ships.get(self.follow)
        if ship is not None and ship.active:
            self._cam = ship.pos.copy()
        elif self._cam is None:
            self._cam = np.array([self.env.width / 2, self.env.height / 2])
        return self._cam

    def _projector(self, cam, size, shake):
        """World -> pixel mapping. In follow mode offsets are wrapped, so the torus seam is seamless."""
        s = self.scale
        cx, cy = size[0] / 2, size[1] / 2
        ox, oy = shake
        if self.follow is None:
            def to_px(pos):
                return int(pos[0] * s + ox), int(pos[1] * s + oy)
        else:
            w, h = self.env.width, self.env.height

            def to_px(pos):
                dx = (pos[0] - cam[0] + w / 2) % w - w / 2
                dy = (pos[1] - cam[1] + h / 2) % h - h / 2
                return int(cx + dx * s + ox), int(cy + dy * s + oy)
        return to_px

    def _visible(self, pos, cam, margin=0.0):
        if self.follow is None:
            return True
        w, h = self.env.width, self.env.height
        dx = abs((pos[0] - cam[0] + w / 2) % w - w / 2)
        dy = abs((pos[1] - cam[1] + h / 2) % h - h / 2)
        lim = self.view_size / 2 + margin
        return dx <= lim and dy <= lim

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
        size = surf.get_size()
        cam = self._camera_center()
        to_px = self._projector(cam, size, (ox, oy))

        if self.follow is None:
            surf.blit(self.bg, (int(ox * 0.3), int(oy * 0.3)))
        else:
            # Scroll the star layer with the camera and tile it, so motion is readable and the
            # torus seam is invisible.
            surf.fill(BG_BOTTOM)
            sw, sh = self.stars.get_size()
            bx = int(-cam[0] * s + ox) % sw - sw
            by = int(-cam[1] * s + oy) % sh - sh
            for gx in (bx, bx + sw):
                for gy in (by, by + sh):
                    surf.blit(self.stars, (gx, gy))

        # Ship trails. Segments are skipped when the projected jump is huge, which covers both a
        # ship wrapping in whole-map view and any leftover seam crossing.
        if self.effects:
            jump = max(size) * 0.5
            for name, pts in st.trails.items():
                if len(pts) < 2:
                    continue
                col = TEAM_COLORS[env.ships[name].team_id]["trail"]
                proj = [to_px(p) for p in pts]
                for i in range(1, len(proj)):
                    a, b = proj[i - 1], proj[i]
                    if abs(a[0] - b[0]) > jump or abs(a[1] - b[1]) > jump:
                        continue
                    f = i / len(proj)
                    pygame.draw.line(surf, tuple(int(c * f) for c in col), a, b,
                                     max(1, int(2.6 * f * s)))

        # Asteroids
        for rock in env.asteroids:
            if not rock.active or not self._visible(rock.pos, cam, rock.radius):
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
            if not b.active or not self._visible(b.pos, cam, 40.0):
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
            if not self._visible(ship.pos, cam, ship.radius * 2):
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

        if self.follow is not None:
            self._highlight_followed(surf, to_px)
            if self.minimap:
                self._draw_minimap(surf, cam)
            # Drawn last so edge contacts are never clipped by the minimap inset
            self._draw_offscreen_markers(surf, cam)
        if self.show_hud:
            self._draw_hud(surf)

        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

    def _wrapped_delta(self, a, b):
        w, h = self.env.width, self.env.height
        return ((b[0] - a[0] + w / 2) % w - w / 2, (b[1] - a[1] + h / 2) % h - h / 2)

    def _highlight_followed(self, surf, to_px):
        """Ring the ship under the player's control so it is unmistakable in a crowd."""
        import pygame

        ship = self.env.ships.get(self.follow)
        if ship is None or not ship.active:
            return
        r = max(8, int(ship.radius * self.scale * 2.2))
        pygame.draw.circle(surf, (235, 240, 255), to_px(ship.pos), r, 1)

    def _draw_offscreen_markers(self, surf, cam):
        """Edge chevrons for ships outside the viewport, with distance in world units."""
        import pygame

        w, h = surf.get_size()
        cx, cy = w / 2, h / 2
        margin = 46  # keeps chevrons and their labels clear of the HUD strip along the top
        for name, ship in self.env.ships.items():
            if not ship.active or name == self.follow or self._visible(ship.pos, cam):
                continue
            dx, dy = self._wrapped_delta(cam, ship.pos)
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            px, py = dx * self.scale, dy * self.scale
            if abs(px) < 1e-6 and abs(py) < 1e-6:
                continue
            # Push out to the window edge along the bearing
            t = min((cx - margin) / max(abs(px), 1e-6), (cy - margin) / max(abs(py), 1e-6))
            ex, ey = cx + px * t, cy + py * t
            ang = math.atan2(py, px)
            col = TEAM_COLORS[ship.team_id]["hull"]
            tip = (ex + math.cos(ang) * 9, ey + math.sin(ang) * 9)
            left = (ex + math.cos(ang + 2.5) * 9, ey + math.sin(ang + 2.5) * 9)
            right = (ex + math.cos(ang - 2.5) * 9, ey + math.sin(ang - 2.5) * 9)
            pygame.draw.polygon(surf, col, [tip, left, right])
            label = self.font_small.render(f"{int(dist)}", True, col)
            lx = min(max(ex - label.get_width() / 2, 2), w - label.get_width() - 2)
            ly = min(max(ey - math.sin(ang) * 16 - 6, 2), h - label.get_height() - 2)
            surf.blit(label, (int(lx), int(ly)))

    def _draw_minimap(self, surf, cam):
        """Whole-map inset: rocks, ships, the followed ship, and the current viewport."""
        import pygame

        env = self.env
        side = max(110, int(min(surf.get_size()) * 0.17))
        pad = 12
        x0, y0 = surf.get_width() - side - pad, surf.get_height() - side - pad
        panel = pygame.Surface((side, side))
        panel.set_alpha(165)
        panel.fill((10, 12, 24))
        surf.blit(panel, (x0, y0))
        pygame.draw.rect(surf, (70, 78, 100), (x0, y0, side, side), 1)
        k = side / max(env.width, env.height)

        def mp(pos):
            return int(x0 + pos[0] * k), int(y0 + pos[1] * k)

        for rock in env.asteroids:
            if rock.active:
                pygame.draw.circle(surf, (96, 92, 108), mp(rock.pos), max(1, int(rock.radius * k)))
        for name, ship in env.ships.items():
            if not ship.active:
                continue
            col = TEAM_COLORS[ship.team_id]["hull"]
            pygame.draw.circle(surf, col, mp(ship.pos), 3 if name == self.follow else 2)
            if name == self.follow:
                pygame.draw.circle(surf, (255, 255, 255), mp(ship.pos), 5, 1)
        # Viewport box, drawn in up to four pieces so it wraps with the torus
        vs = self.view_size * k
        bx, by = cam[0] * k - vs / 2, cam[1] * k - vs / 2
        for gx in (bx, bx + side, bx - side):
            for gy in (by, by + side, by - side):
                r = pygame.Rect(int(x0 + gx), int(y0 + gy), int(vs), int(vs))
                clipped = r.clip(pygame.Rect(x0, y0, side, side))
                if clipped.width > 1 and clipped.height > 1:
                    pygame.draw.rect(surf, (200, 210, 235), clipped, 1)

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
