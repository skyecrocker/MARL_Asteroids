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
    prev_hp: dict = field(default_factory=dict)
    flash: dict = field(default_factory=dict)  # agent -> frames of damage flash remaining


class Renderer:
    def __init__(self, env, window: int = 1000, show_hud: bool = True, effects: bool = True,
                 trail_length: int = 18, follow: Optional[str] = None, view_size: float = 1200.0,
                 minimap: bool = True, show_obs: bool = False, obs_agent: Optional[str] = None):
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
        self.show_obs = show_obs      # toggled live with Tab in watch.py
        self.obs_agent = obs_agent    # whose observation to draw; defaults to the followed ship
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

    def _image_finder(self, to_px, size):
        """``images(pos, radius)`` -> pixel centers of every toroidal copy that overlaps the surface.

        The map repeats every (width, height) world units, i.e. every period * scale pixels. When
        the view spans (nearly) the whole map, an object crossing the edge must be drawn on both
        sides at once, half on each, instead of popping across.
        """
        px_w, px_h = self.env.width * self.scale, self.env.height * self.scale
        sw, sh = size

        def images(pos, radius):
            cx, cy = to_px(pos)
            r = radius * self.scale + 2
            for dx in (0.0, -px_w, px_w):
                x = cx + dx
                if x < -r or x > sw + r:
                    continue
                for dy in (0.0, -px_h, px_h):
                    y = cy + dy
                    if -r <= y <= sh + r:
                        yield (x, y)
        return images

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
        """Infer cosmetic events from env state (hits, deaths, new bullets) without touching it."""
        env, st = self.env, self.state
        for name, ship in env.ships.items():
            prev = st.prev_hp.get(name)
            if prev is not None and ship.hp < prev and ship.active:
                st.flash[name] = 6
                c = TEAM_COLORS[ship.team_id]
                self._spawn_burst(ship.pos[0], ship.pos[1], 10, 2.0, (255, 240, 200), 12, 2.0)
                st.shake = min(10.0, st.shake + 3.0)
            st.prev_hp[name] = ship.hp
            st.flash[name] = max(0, st.flash.get(name, 0) - 1)
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
        images = self._image_finder(to_px, size)

        def at(center, offset):
            """Pixel point ``offset`` world units from a projected center. Shapes are built from
            offsets, never by projecting each vertex: per-vertex projection wraps vertices on
            opposite sides of the seam independently and smears the shape across the screen."""
            return int(center[0] + offset[0] * s), int(center[1] + offset[1] * s)

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
            if not rock.active:
                continue
            _, poly, spin, craters = self._rock_shape(rock)
            th = env.frame * spin
            c_, s_ = math.cos(th), math.sin(th)
            rot = np.array([[c_, -s_], [s_, c_]])
            offsets = poly @ rot.T
            crater_offs = [((cx * c_ - cy * s_, cx * s_ + cy * c_), cr) for cx, cy, cr in craters]
            for center in images(rock.pos, rock.radius * 1.15):
                pts = [at(center, o) for o in offsets]
                pygame.draw.polygon(surf, ROCK_FILL, pts)
                for off, cr in crater_offs:  # craters for texture, rotating with the rock
                    pygame.draw.circle(surf, (58, 55, 68), at(center, off), max(1, int(cr * s)))
                pygame.draw.polygon(surf, ROCK_EDGE, pts, max(1, int(1.8 * s)))

        # Particles (under ships, over rocks)
        if self.effects:
            for p in st.particles:
                f = max(0.0, p.life / p.max_life)
                r = max(1, int(p.radius * s * (1.6 - f if p.kind == "smoke" else f)))
                col = tuple(min(255, int(c * (f ** 0.6))) for c in p.color)
                for center in images((p.x, p.y), p.radius * 2):
                    pygame.draw.circle(surf, col, (int(center[0]), int(center[1])), r)

        # Bullets: bright core with a tapering glow tail so incoming fire is easy to read
        for b in env.bullets:
            if not b.active:
                continue
            glow = TEAM_COLORS[b.team_id]["glow"]
            tail_len = 4.0 * math.hypot(b.vel[0], b.vel[1])
            for center in images(b.pos, b.radius + tail_len):
                px = (int(center[0]), int(center[1]))
                for k, (mult, fade) in enumerate(((4.0, 0.25), (2.2, 0.55))):
                    tail = at(center, (-b.vel[0] * mult, -b.vel[1] * mult))
                    pygame.draw.line(surf, tuple(int(c * fade) for c in glow), tail, px,
                                     max(1, int((4 - k) * s)))
                pygame.draw.circle(surf, glow, px, max(3, int(b.radius * s * 2.0)))
                pygame.draw.circle(surf, BULLET_CORE, px, max(2, int(b.radius * s * 1.1)))

        # Ships
        for name, ship in env.ships.items():
            if not ship.active:
                continue
            col = TEAM_COLORS[ship.team_id]
            a, r = ship.angle, ship.radius
            hull_offs = [(math.cos(a + o) * r * m, math.sin(a + o) * r * m)
                         for o, m in ((0.0, 1.45), (2.45, 1.0), (math.pi, 0.45), (-2.45, 1.0))]
            flame_offs = None
            if self.effects and ship.thrusting:  # thruster flame opposite the heading
                flick = 0.75 + 0.5 * self.rng.random()
                back = a + math.pi
                flame_offs = [(math.cos(back + 0.85) * r * 0.5, math.sin(back + 0.85) * r * 0.5),
                              (math.cos(back) * r * (1.0 + 1.5 * flick),
                               math.sin(back) * r * (1.0 + 1.5 * flick)),
                              (math.cos(back - 0.85) * r * 0.5, math.sin(back - 0.85) * r * 0.5),
                              (math.cos(back) * r * 0.5, math.sin(back) * r * 0.5)]
                flame_col = (255, int(150 + 80 * flick), 60)
            body = col["hull"]
            if self.effects and st.flash.get(name, 0) > 0:  # white flash on taking damage
                body = (255, 255, 255)
            # Bright outline while armed, dim while reloading: readable weapon state at a glance.
            armed = ship.cooldown <= 0
            edge = (250, 250, 255) if armed else tuple(int(c * 0.45) for c in col["hull"])
            # Radius covers the flame, the health bar above the ship, and the reload dot
            for center in images(ship.pos, r * 3.0):
                if flame_offs is not None:
                    pygame.draw.polygon(surf, flame_col, [at(center, o) for o in flame_offs])
                hull = [at(center, o) for o in hull_offs]
                pygame.draw.polygon(surf, body, hull)
                pygame.draw.polygon(surf, edge, hull, max(1, int(2.0 * s)))
                if not armed:  # reload dot at the nose, growing as the shot comes back
                    frac = 1.0 - ship.cooldown / max(1, ship.fire_rate)
                    nose = at(center, (math.cos(a) * r * 1.45, math.sin(a) * r * 1.45))
                    pygame.draw.circle(surf, tuple(int(c * frac) for c in col["glow"]),
                                       nose, max(1, int(2.6 * s * frac)))
                self._draw_health_bar(surf, ship, center)

        if self.show_obs:
            agent = self.obs_agent or self.follow or next(iter(env.ships), None)
            if agent is not None:
                self._draw_obs_overlay(surf, to_px, agent)
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
        if mode == "surface":  # caller blits it into a window it owns
            return surf
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

    def _draw_health_bar(self, surf, ship, center):
        """Segmented bar above the ship (at pixel ``center``): hull blocks plus shield blocks."""
        import pygame

        if ship.max_hp <= 0:
            return
        px, py = int(center[0]), int(center[1])
        seg = max(3, int(5 * self.scale))
        gap = max(1, int(1.5 * self.scale))
        total = ship.max_hp * seg + (ship.max_hp - 1) * gap
        x0 = px - total // 2
        y0 = py - int(ship.radius * self.scale) - max(7, int(9 * self.scale))
        h = max(2, int(3 * self.scale))
        for i in range(ship.max_hp):
            filled_hull = i < ship.hull
            is_shield_slot = i >= ship.max_hull
            if is_shield_slot:
                filled = (i - ship.max_hull) < ship.shield
                color = (110, 220, 255) if filled else (34, 48, 62)
            else:
                color = ((120, 230, 140) if ship.hull > 1 else (255, 120, 110)) if filled_hull \
                    else (56, 40, 44)
            pygame.draw.rect(surf, color, (x0 + i * (seg + gap), y0, seg, h))

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
        # Viewport box, tiled so it wraps with the torus. Uses a clip region rather than clipping the
        # rects themselves, which would draw false edges along the panel border.
        if self.view_size >= max(env.width, env.height) - 1:
            return  # viewport covers the whole map: a box would just be stray lines
        vs = self.view_size * k
        bx, by = cam[0] * k - vs / 2, cam[1] * k - vs / 2
        prev_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(x0, y0, side, side))
        for gx in (bx, bx + side, bx - side):
            for gy in (by, by + side, by - side):
                pygame.draw.rect(surf, (200, 210, 235),
                                 pygame.Rect(int(x0 + gx), int(y0 + gy), int(vs), int(vs)), 1)
        surf.set_clip(prev_clip)

    def _draw_obs_overlay(self, surf, to_px, agent: str):
        """Draw the observation vector decoded back into world space.

        This is the agent's own view, reconstructed from the numbers the policy receives, so it both
        documents exactly what the agent knows and doubles as a debugging tool. Range is recovered by
        inverting ``dist / (dist + sensor_range)``, and bearing is rotated out of the ego frame.
        """
        import pygame

        from .env import (ALLY_DIM, ASTEROID_DIM, BASE_DIM, BULLET_DIM, EGO_DIM, ENEMY_DIM)

        env = self.env
        ship = env.ships.get(agent)
        if ship is None or not ship.active:
            return
        obs = env.get_obs(agent)
        cfg = env.config
        L = cfg.sensor_range

        # Sensor scale ring: where the range feature reads 0.5
        pygame.draw.circle(surf, (58, 74, 96), to_px(ship.pos), int(L * self.scale), 1)

        blocks = [("enemy", cfg.obs_enemies, ENEMY_DIM, (255, 110, 100)),
                  ("ally", cfg.obs_allies, ALLY_DIM, (110, 190, 255)),
                  ("rock", cfg.obs_asteroids, ASTEROID_DIM, (170, 166, 186)),
                  ("bullet", cfg.obs_bullets, BULLET_DIM, (255, 226, 120))]
        off = EGO_DIM
        for kind, count, dim, color in blocks:
            for k in range(count):
                slot = obs[off + k * dim: off + (k + 1) * dim]
                if slot.size < BASE_DIM or slot[0] <= 0.5:
                    continue
                bx, by, r = float(slot[1]), float(slot[2]), float(min(slot[3], 0.999))
                dist = r * L / max(1e-6, 1.0 - r)
                world_ang = ship.angle + math.atan2(by, bx)
                # Wrap into the map so a contact decoded across the edge lands on screen
                pos = ((ship.pos[0] + math.cos(world_ang) * dist) % env.width,
                       (ship.pos[1] + math.sin(world_ang) * dist) % env.height)
                px = to_px(pos)
                size = 7 if kind in ("enemy", "ally") else 4
                pygame.draw.circle(surf, color, px, size, 1)
                # Slot rank, so it is obvious which contact occupies which observation slot
                if kind in ("enemy", "ally"):
                    surf.blit(self.font_small.render(str(k), True, color), (px[0] + 8, px[1] - 6))
                    # Relative velocity in the ego frame, rotated back to world
                    vref = 2 * cfg.max_speed + cfg.bullet_speed
                    rvx, rvy = float(slot[4]) * vref, float(slot[5]) * vref
                    ca, sa = math.cos(ship.angle), math.sin(ship.angle)
                    wvx, wvy = ca * rvx - sa * rvy, sa * rvx + ca * rvy
                    # End point offset from the projected marker, not projected on its own,
                    # so the line cannot wrap away from its start
                    end = (int(px[0] + wvx * 5 * self.scale), int(px[1] + wvy * 5 * self.scale))
                    pygame.draw.line(surf, tuple(int(c * 0.7) for c in color), px, end, 1)
            off += count * dim

        # Ego readouts, in the same units the policy sees
        vfwd, vlat = float(obs[1]), float(obs[2])
        lines = [f"obs view: {agent}",
                 f"v_fwd {vfwd:+.2f}  v_lat {vlat:+.2f}",
                 f"reload {float(obs[3]):.2f}  armed {int(obs[4] > 0.5)}",
                 f"allies {float(obs[5]):.2f}  enemies {float(obs[6]):.2f}",
                 f"time_left {float(obs[7]):.2f}",
                 f"hull {float(obs[8]):.2f}  shield {float(obs[9]):.2f}"]
        panel = pygame.Surface((196, 16 * len(lines) + 10))
        panel.set_alpha(170)
        panel.fill((8, 10, 20))
        surf.blit(panel, (12, 52))
        for i, line in enumerate(lines):
            surf.blit(self.font_small.render(line, True, (198, 210, 230)), (20, 58 + i * 16))

    def _draw_hud(self, surf):
        env = self.env
        import pygame

        red, blue = env._team_alive(RED), env._team_alive(BLUE)
        surf.blit(self.font.render(f"RED {red}  hp {env.team_hp(RED)}", True,
                                   TEAM_COLORS[RED]["hull"]), (14, 12))
        label = self.font.render(f"hp {env.team_hp(BLUE)}  BLUE {blue}", True, TEAM_COLORS[BLUE]["hull"])
        surf.blit(label, (surf.get_width() - label.get_width() - 14, 12))
        limit = env.config.max_frames
        t = self.font.render(f"{env.frame}" if limit is None else f"{env.frame}/{limit}", True, HUD_DIM)
        surf.blit(t, ((surf.get_width() - t.get_width()) // 2, 12))
        if limit is not None:  # no clock bar when the game has no time limit
            bar_w = surf.get_width() - 28
            frac = min(1.0, env.frame / max(1, limit))
            pygame.draw.rect(surf, (40, 44, 60), (14, 34, bar_w, 3))
            pygame.draw.rect(surf, HUD_DIM, (14, 34, int(bar_w * frac), 3))
        if env.game_over:
            msg = {BLUE: "BLUE WINS", RED: "RED WINS"}.get(env.winner, "DRAW")
            color = TEAM_COLORS.get(env.winner, {}).get("hull", HUD_TEXT)
            surface = self.font_big.render(msg, True, color)
            surf.blit(surface, ((surf.get_width() - surface.get_width()) // 2,
                                (surf.get_height() - surface.get_height()) // 2))

    def close(self):
        """Release drawing state. Only shuts pygame down if this renderer opened the display;
        in ``surface``/``rgb_array`` mode the window (if any) belongs to someone else."""
        import pygame
        if self.screen is not None:
            pygame.quit()
        self.surface = None
        self.screen = None
