"""Pygame renderer for TeamAsteroidsEnv.

Kept separate from the environment so visual polish never affects simulation or training. The env
holds no drawing state; it just forwards render() here. Purely cosmetic effects (thruster flames,
muzzle flashes, explosions, starfield, trails) live in the renderer.

The view scales the world to fit the window, so large maps still fit on screen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .entities import Convoy
from .env import BLUE, RED

# Palette
BG_TOP = (6, 8, 20)
BG_BOTTOM = (16, 12, 32)
TEAM_COLORS = {
    RED: {"hull": (255, 96, 88), "glow": (255, 40, 40), "trail": (120, 30, 30)},
    BLUE: {"hull": (96, 176, 255), "glow": (40, 140, 255), "trail": (26, 70, 120)},
}
# --- Roguelite looks. Variants differ by silhouette, intelligence tiers by shade, pips and a ring.
# Shapes are (angle offset from the heading, radius multiple) vertex lists. Hitboxes stay circles of
# ship.radius whatever the silhouette.
SHAPES = {
    "standard": ((0.0, 1.45), (2.45, 1.0), (math.pi, 0.45), (-2.45, 1.0)),
    # slim dart with a forked tail
    "scout": ((0.0, 1.6), (2.2, 0.75), (2.75, 1.35), (math.pi, 0.35), (-2.75, 1.35), (-2.2, 0.75)),
    "heavy": ((0.0, 1.3), (1.2, 1.2), (2.3, 1.25), (math.pi, 0.9), (-2.3, 1.25), (-1.2, 1.2)),
    # long barrel on a small cross-winged body
    "sniper": ((0.0, 2.2), (0.12, 1.0), (1.4, 0.75), (2.5, 1.1), (math.pi, 0.5), (-2.5, 1.1),
               (-1.4, 0.75), (-0.12, 1.0)),
    "wingman": ((0.0, 1.35), (2.0, 1.15), (2.7, 0.6), (math.pi, 0.5), (-2.7, 0.6), (-2.0, 1.15)),
}
DEFAULT_SHAPE = SHAPES["standard"]
# tier -> (body, pips, ring) for enemies; bodies stay red-dominant
ENEMY_TIER_STYLE = {
    "basic": ((255, 112, 100), 0, False),
    "veteran": ((228, 62, 52), 1, False),
    "elite": ((176, 22, 70), 2, True),
}
WINGMAN_BODY = (120, 220, 235)
WINGMAN_TIER_PIPS = {"basic": (0, False), "veteran": (1, False), "elite": (2, True)}
ENEMY_PIP = (255, 206, 84)
WINGMAN_PIP = (245, 250, 255)
ELITE_RING = (255, 190, 70)


@dataclass(frozen=True)
class ShipLook:
    shape: tuple
    body: Tuple[int, int, int]
    pips: int = 0
    pip_color: Tuple[int, int, int] = (255, 255, 255)
    ring: Optional[Tuple[int, int, int]] = None


def ship_look(ship) -> ShipLook:
    """How a ship is drawn, from its team, roguelite variant and intelligence tier. Ships without a
    variant (every existing mode, and your own ship) keep the classic arrowhead in team colors."""
    variant = getattr(ship, "variant", None)
    tier = getattr(ship, "tier", None)
    if variant is None:
        return ShipLook(DEFAULT_SHAPE, TEAM_COLORS[ship.team_id]["hull"])
    if variant == "wingman":
        pips, ring = WINGMAN_TIER_PIPS.get(tier, (0, False))
        return ShipLook(SHAPES["wingman"], WINGMAN_BODY, pips, WINGMAN_PIP,
                        WINGMAN_PIP if ring else None)
    body, pips, ring = ENEMY_TIER_STYLE.get(tier, ENEMY_TIER_STYLE["basic"])
    if ship.team_id != RED:  # never happens in play; keep team colors honest anyway
        body = TEAM_COLORS[ship.team_id]["hull"]
    return ShipLook(SHAPES.get(variant, DEFAULT_SHAPE), body, pips, ENEMY_PIP,
                    ELITE_RING if ring else None)


def draw_look(surface, look: ShipLook, center, angle: float, radius_px: float,
              edge=(250, 250, 255), edge_width: int = 2):
    """Draw a ship look at a pixel ``center`` (used by the renderer and the legend)."""
    import pygame

    cx, cy = center
    pts = [(int(cx + math.cos(angle + o) * radius_px * m), int(cy + math.sin(angle + o) * radius_px * m))
           for o, m in look.shape]
    pygame.draw.polygon(surface, look.body, pts)
    pygame.draw.polygon(surface, edge, pts, max(1, edge_width))
    draw_marks(surface, look, center, angle, radius_px)


def draw_marks(surface, look: ShipLook, center, angle: float, radius_px: float):
    """Intelligence markings: pips across the hull and, for elites, a ring."""
    import pygame

    cx, cy = center
    if look.pips:
        # Pips sit side by side across the hull, just behind the center
        px, py = -math.sin(angle), math.cos(angle)
        bx, by = cx - math.cos(angle) * radius_px * 0.15, cy - math.sin(angle) * radius_px * 0.15
        pr = max(2, int(round(radius_px * 0.3)))
        for i in range(look.pips):
            off = (i - (look.pips - 1) / 2) * radius_px * 0.7
            c = (int(bx + px * off), int(by + py * off))
            pygame.draw.circle(surface, (20, 14, 24), c, pr + 1)  # dark rim: readable on any body
            pygame.draw.circle(surface, look.pip_color, c, pr)
    if look.ring is not None:
        pygame.draw.circle(surface, look.ring, (int(cx), int(cy)), max(6, int(radius_px * 2.0)), 1)


LEGEND_VARIANTS = (("standard", "Standard", "balanced"), ("scout", "Scout", "fast, fragile"),
                   ("heavy", "Heavy", "slow, tough"), ("sniper", "Sniper", "long range"))
LEGEND_TIERS = (("basic", "Basic", "slow to react"), ("veteran", "Veteran", "sharper"),
                ("elite", "Elite", "fast and accurate"))


class _LegendShip:
    def __init__(self, variant, tier, team=RED):
        self.variant, self.tier, self.team_id = variant, tier, team


def legend_height(row: int = 26) -> int:
    return 3 * 24 + 2 * 8 + row * (len(LEGEND_VARIANTS) + len(LEGEND_TIERS) + 1) + 8


def draw_legend(surface, rect, font, row: int = 26, title_color=(226, 232, 245),
                hints: Optional[bool] = None):
    """Key for enemy variants (silhouette), intelligence tiers (shade, pips, ring) and wingmen.
    ``hints`` adds a short description per row (default: when ``rect`` is wide enough)."""
    x, y = rect[0], rect[1]
    if hints is None:
        hints = len(rect) < 3 or rect[2] >= 270
    dim = (130, 142, 166)
    icon_r = max(6.0, row * 0.33)
    first = True

    def heading(text):
        nonlocal y, first
        if not first:
            y += 8
        first = False
        surface.blit(font.render(text, True, title_color), (x, y))
        y += 24

    def entry(look, name, hint):
        nonlocal y
        draw_look(surface, look, (x + 16, y + row // 2), -math.pi / 2, icon_r, edge_width=1)
        surface.blit(font.render(name, True, title_color), (x + 40, y + row // 2 - 9))
        if hints:
            surface.blit(font.render(hint, True, dim), (x + 48 + max(font.size(name)[0], 70),
                                                        y + row // 2 - 9))
        y += row

    heading("Enemy types")
    for v, name, hint in LEGEND_VARIANTS:
        entry(ship_look(_LegendShip(v, "basic")), name, hint)
    heading("Enemy intelligence")
    for t, name, hint in LEGEND_TIERS:
        entry(ship_look(_LegendShip("standard", t)), name, hint)
    heading("Your fleet")
    entry(ship_look(_LegendShip("wingman", "basic", BLUE)), "Wingman", "AI ally")
    return y - rect[1]


ROCK_FILL = (74, 70, 86)
ROCK_EDGE = (150, 146, 168)
BULLET_CORE = (255, 255, 235)
HUD_TEXT = (226, 232, 245)
THRUST_COLOR = (255, 190, 90)  # engine exhaust, fades to black with age
THRUST_PER_FRAME = 2           # sparks per thrusting ship per rendered frame
THRUST_MAX_PARTICLES = 400     # cap on live exhaust sparks across all ships
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
                 minimap: bool = True, show_obs: bool = False, obs_agent: Optional[str] = None,
                 hud_text: Optional[Callable[[], str]] = None, thruster_particles: bool = True):
        """
        Args:
            follow: agent name to keep centered. None renders the whole map scaled to the window.
            view_size: world units across the viewport in follow mode (clamped to the map size).
            minimap: draw a whole-map inset in follow mode.
            hud_text: optional callback for the line at the top center (roguelite: wave, enemies
                left, credits), replacing the clock.
        """
        self.hud_text = hud_text
        self.thruster_particles = thruster_particles  # off: the classic polygon flame
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
            if ship.reserve:  # survival ships waiting for their wave were never in play
                continue
            if not ship.active and name not in st.dead_seen:
                st.dead_seen.add(name)
                c = TEAM_COLORS[ship.team_id]
                self._spawn_burst(ship.pos[0], ship.pos[1], 34, 3.4, c["glow"], 26, 3.0, "spark")
                self._spawn_burst(ship.pos[0], ship.pos[1], 16, 1.6, (255, 226, 150), 20, 5.0, "smoke")
                st.shake = min(16.0, st.shake + 9.0)
            elif ship.active and name in st.dead_seen:
                st.dead_seen.discard(name)  # respawned (or a new episode)
                st.trails[name] = []        # don't draw a trail from the wreck to the spawn point
        for team, station in env.stations.items():
            key = ("station", team)
            prev = st.prev_hp.get(key)
            if prev is not None and station.hp < prev:
                st.flash[key] = 5
                self._spawn_burst(station.pos[0], station.pos[1], 6, 2.5, (255, 240, 200), 10, 2.0)
            if not station.active and key not in st.dead_seen:
                st.dead_seen.add(key)
                c = TEAM_COLORS[team]
                self._spawn_burst(station.pos[0], station.pos[1], 90, 6.0, c["glow"], 50, 4.0, "spark")
                self._spawn_burst(station.pos[0], station.pos[1], 40, 3.0, (255, 226, 150), 45, 9.0, "smoke")
                st.shake = 24.0
            st.prev_hp[key] = station.hp
            st.flash[key] = max(0, st.flash.get(key, 0) - 1)
        n = len(env.bullets)
        if n > st.prev_bullets:
            for b in env.bullets[st.prev_bullets:n]:
                c = TEAM_COLORS[b.team_id]
                self._spawn_burst(b.pos[0], b.pos[1], 5, 1.1, c["glow"], 9, 2.0)
        st.prev_bullets = n

    def _spawn_thrust(self):
        """Engine exhaust: a few short-lived sparks per thrusting ship each frame, behind the ship.

        Only ships on screen spawn them, and the total is capped, so the cost stays flat as ships
        are added. Exhaust is only spawned here (rendering), never in the env, so training is
        unaffected."""
        if not self.thruster_particles:
            return
        env, st = self.env, self.state
        cam = self._camera_center()
        live = sum(1 for p in st.particles if p.kind == "thrust")
        budget = THRUST_MAX_PARTICLES - live
        for ship in env.ships.values():
            if budget <= 0:
                break
            if not ship.active or ship.reserve or not ship.thrusting:
                continue
            if not self._visible(ship.pos, cam, margin=ship.radius * 4):
                continue
            back = ship.angle + math.pi
            ex = ship.pos[0] + math.cos(back) * ship.radius
            ey = ship.pos[1] + math.sin(back) * ship.radius
            for _ in range(THRUST_PER_FRAME):
                spread = (self.rng.random() - 0.5) * 0.5
                speed = 1.0 + 1.2 * self.rng.random()
                vx = math.cos(back + spread) * speed + ship.vel[0] * 0.5
                vy = math.sin(back + spread) * speed + ship.vel[1] * 0.5
                life = 9.0 + 6.0 * self.rng.random()
                st.particles.append(Particle(ex, ey, vx, vy, life, life, 2.2,
                                             THRUST_COLOR, "thrust"))
            budget -= THRUST_PER_FRAME

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
            self._spawn_thrust()
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

        # Escort route: a dashed line from the convoy to its destination, and the destination gate
        if env.convoy is not None and env.convoy.active:
            self._draw_route(surf, images, at)
        # Capture the flag bases (and flags sitting at them or dropped in space)
        for team, flag in env.flags.items():
            for center in images(flag.base, env.config.flag_radius * 2):
                self._draw_base(surf, team, center)
            if flag.carrier is None:
                for center in images(flag.pos, 40):
                    self._draw_flag(surf, team, center, dropped=flag.drop_timer is not None)

        # Stations (siege) and the escort convoy
        for team, station in env.stations.items():
            for center in images(station.pos, station.radius * 1.7):
                self._draw_station(surf, station, center, at)

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
            look = ship_look(ship)
            a, r = ship.angle, ship.radius
            hull_offs = [(math.cos(a + o) * r * m, math.sin(a + o) * r * m) for o, m in look.shape]
            flame_offs = None
            if self.effects and ship.thrusting and not self.thruster_particles:  # classic flame
                flick = 0.75 + 0.5 * self.rng.random()
                back = a + math.pi
                flame_offs = [(math.cos(back + 0.85) * r * 0.5, math.sin(back + 0.85) * r * 0.5),
                              (math.cos(back) * r * (1.0 + 1.5 * flick),
                               math.sin(back) * r * (1.0 + 1.5 * flick)),
                              (math.cos(back - 0.85) * r * 0.5, math.sin(back - 0.85) * r * 0.5),
                              (math.cos(back) * r * 0.5, math.sin(back) * r * 0.5)]
                flame_col = (255, int(150 + 80 * flick), 60)
            body = look.body
            if self.effects and st.flash.get(name, 0) > 0:  # white flash on taking damage
                body = (255, 255, 255)
            # Bright outline while armed, dim while reloading: readable weapon state at a glance.
            armed = ship.cooldown <= 0
            edge = (250, 250, 255) if armed else tuple(int(c * 0.45) for c in look.body)
            # Radius covers the flame, the health bar above the ship, and the reload dot
            for center in images(ship.pos, r * 3.0):
                if flame_offs is not None:
                    pygame.draw.polygon(surf, flame_col, [at(center, o) for o in flame_offs])
                hull = [at(center, o) for o in hull_offs]
                pygame.draw.polygon(surf, body, hull)
                pygame.draw.polygon(surf, edge, hull, max(1, int(2.0 * s)))
                if look.pips or look.ring is not None:  # roguelite intelligence markings
                    draw_marks(surf, look, center, a, r * s)
                if ship.protection > 0 and (env.frame // 5) % 2 == 0:  # spawn protection: pulsing bubble
                    pygame.draw.circle(surf, (190, 225, 255), (int(center[0]), int(center[1])),
                                       max(6, int(r * 1.9 * s)), 1)
                if not armed:  # reload dot at the nose, growing as the shot comes back
                    frac = 1.0 - ship.cooldown / max(1, ship.fire_rate)
                    nose = at(center, (math.cos(a) * r * 1.45, math.sin(a) * r * 1.45))
                    pygame.draw.circle(surf, tuple(int(c * frac) for c in col["glow"]),
                                       nose, max(1, int(2.6 * s * frac)))
                self._draw_health_bar(surf, ship, center)

        for team, flag in env.flags.items():  # flags trail behind their carriers
            if flag.carrier is not None:
                ship = env.ships[flag.carrier]
                back = ship.angle + math.pi
                for center in images(ship.pos, 40):
                    self._draw_flag(surf, team, at(center, (math.cos(back) * ship.radius * 1.6,
                                                            math.sin(back) * ship.radius * 1.6)),
                                    small=True)

        if self.show_obs:
            agent = self.obs_agent or self.follow or next(iter(env.ships), None)
            if agent is not None:
                self._draw_obs_overlay(surf, to_px, agent)
        if self.follow is not None:
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

    def _draw_route(self, surf, images, at):
        import pygame
        env = self.env
        convoy = env.convoy
        remaining = env.route_length - convoy.distance
        col = TEAM_COLORS[BLUE]["trail"]
        for center in images(convoy.pos, remaining + 60):
            x = 0.0
            while x < remaining:  # dashes toward +x
                a = at(center, (x, 0.0))
                b = at(center, (min(x + 18, remaining), 0.0))
                pygame.draw.line(surf, col, a, b, max(1, int(2 * self.scale)))
                x += 40
            gate = at(center, (remaining, 0.0))
            g = max(10, int(46 * self.scale))
            pygame.draw.line(surf, TEAM_COLORS[BLUE]["hull"], (gate[0], gate[1] - g), (gate[0], gate[1] + g),
                             max(2, int(3 * self.scale)))
            for dy in (-g, g):
                pygame.draw.circle(surf, TEAM_COLORS[BLUE]["glow"], (gate[0], gate[1] + dy), max(3, int(5 * self.scale)))
            # Escort ring: blue ships inside it keep the convoy moving
            rr = int(env.config.convoy_escort_radius * self.scale)
            ring = (90, 150, 210) if convoy.moving else (150, 90, 90)
            pygame.draw.circle(surf, ring, (int(center[0]), int(center[1])), rr, 1)

    def _draw_base(self, surf, team, center):
        import pygame
        col = TEAM_COLORS[team]
        r = int(self.env.config.flag_radius * self.scale)
        c = (int(center[0]), int(center[1]))
        pygame.draw.circle(surf, tuple(int(v * 0.35) for v in col["trail"]), c, int(r * 1.6))
        pygame.draw.circle(surf, col["hull"], c, int(r * 1.6), max(1, int(2 * self.scale)))
        pygame.draw.circle(surf, col["glow"], c, max(2, int(r * 0.25)))

    def _draw_flag(self, surf, team, center, dropped=False, small=False):
        import pygame
        s = self.scale * (0.7 if small else 1.0)
        col = TEAM_COLORS[team]["hull"]
        if dropped and (self.env.frame // 10) % 2 == 0:
            col = (255, 255, 255)  # blinking: it can be grabbed or returned
        x, y = int(center[0]), int(center[1])
        pole = int(30 * s)
        pygame.draw.line(surf, (220, 225, 235), (x, y + pole // 3), (x, y - pole), max(1, int(2 * s)))
        pygame.draw.polygon(surf, col, [(x, y - pole), (x + int(20 * s), y - pole + int(7 * s)),
                                        (x, y - pole + int(14 * s))])

    def _draw_station(self, surf, station, center, at):
        """Octagonal hull, rotating ring, glowing core, shield bubble, and shield/hull bars."""
        import pygame

        s = self.scale
        cx, cy = int(center[0]), int(center[1])
        R = station.radius
        col = TEAM_COLORS[station.team_id]
        if isinstance(station, Convoy):
            return self._draw_convoy(surf, station, center, at)
        octagon = [(math.cos(k * math.pi / 4 + math.pi / 8) * R, math.sin(k * math.pi / 4 + math.pi / 8) * R)
                   for k in range(8)]
        if not station.active:  # wreck
            pygame.draw.polygon(surf, (44, 42, 52), [at(center, o) for o in octagon])
            pygame.draw.polygon(surf, (78, 74, 88), [at(center, o) for o in octagon], max(1, int(2 * s)))
            return
        flash = self.effects and self.state.flash.get(("station", station.team_id), 0) > 0
        fill = (255, 255, 255) if flash else tuple(int(c * 0.55) for c in col["trail"])
        pts = [at(center, o) for o in octagon]
        pygame.draw.polygon(surf, fill, pts)
        pygame.draw.polygon(surf, col["hull"], pts, max(2, int(3 * s)))
        # Rotating ring of docking lights
        spin = self.env.frame * 0.01
        for k in range(6):
            a = spin + k * math.pi / 3
            pygame.draw.circle(surf, col["glow"], at(center, (math.cos(a) * R * 0.68, math.sin(a) * R * 0.68)),
                               max(1, int(2.5 * s)))
        pygame.draw.circle(surf, col["glow"], (cx, cy), max(3, int(R * 0.3 * s)))
        pygame.draw.circle(surf, (255, 255, 255), (cx, cy), max(1, int(R * 0.12 * s)))
        # Shield bubble, brighter the fuller it is
        frac = station.shield / max(1, station.max_shield)
        if frac > 0:
            rr = int((R + 9) * s)
            bubble = pygame.Surface((2 * rr + 4, 2 * rr + 4), pygame.SRCALPHA)
            alpha = int(40 + 130 * frac)
            pygame.draw.circle(bubble, (110, 220, 255, alpha), (rr + 2, rr + 2), rr, max(1, int(2 * s)))
            surf.blit(bubble, (cx - rr - 2, cy - rr - 2))
        # Bars above: shield on top, hull below
        bw = int(R * 2.4 * s)
        bh = max(3, int(4 * s))
        x0 = cx - bw // 2
        y0 = cy - int((R + 22) * s)
        hull_frac = station.hull / max(1, station.max_hull)
        hull_col = (120, 230, 140) if hull_frac > 0.35 else (255, 120, 110)
        for i, (f, c) in enumerate(((frac, (110, 220, 255)), (hull_frac, hull_col))):
            y = y0 + i * (bh + 2)
            pygame.draw.rect(surf, (30, 34, 48), (x0, y, bw, bh))
            pygame.draw.rect(surf, c, (x0, y, int(bw * f), bh))

    def _draw_convoy(self, surf, convoy, center, at):
        """A long freighter hull pointing +x, with engine glow while moving and the same bars."""
        import pygame
        s, R = self.scale, convoy.radius
        col = TEAM_COLORS[BLUE]
        cx, cy = int(center[0]), int(center[1])
        hull = [(1.5 * R, 0), (0.9 * R, 0.55 * R), (-1.3 * R, 0.6 * R), (-1.5 * R, 0.3 * R),
                (-1.5 * R, -0.3 * R), (-1.3 * R, -0.6 * R), (0.9 * R, -0.55 * R)]
        if not convoy.active:
            pygame.draw.polygon(surf, (44, 42, 52), [at(center, o) for o in hull])
            return
        if convoy.moving and self.effects:
            flick = 0.7 + 0.6 * self.rng.random()
            for dy in (-0.3 * R, 0.3 * R):
                pygame.draw.polygon(surf, (255, int(160 + 70 * flick), 70),
                                    [at(center, (-1.5 * R, dy - 6)), at(center, (-1.5 * R - 22 * flick, dy)),
                                     at(center, (-1.5 * R, dy + 6))])
        flash = self.effects and self.state.flash.get(("station", BLUE), 0) > 0
        fill = (255, 255, 255) if flash else tuple(int(c * 0.6) for c in col["trail"])
        pts = [at(center, o) for o in hull]
        pygame.draw.polygon(surf, fill, pts)
        pygame.draw.polygon(surf, col["hull"], pts, max(2, int(3 * s)))
        for k in range(3):  # cargo pods
            pygame.draw.rect(surf, col["glow"], (*at(center, (-0.9 * R + k * 0.7 * R, -0.25 * R)),
                                                 max(3, int(0.45 * R * s)), max(3, int(0.5 * R * s))), 1)
        frac = convoy.shield / max(1, convoy.max_shield)
        if frac > 0:
            rr = int((R * 1.6 + 6) * s)
            bubble = pygame.Surface((2 * rr + 4, 2 * rr + 4), pygame.SRCALPHA)
            pygame.draw.circle(bubble, (110, 220, 255, int(40 + 130 * frac)), (rr + 2, rr + 2), rr,
                               max(1, int(2 * s)))
            surf.blit(bubble, (cx - rr - 2, cy - rr - 2))
        bw, bh = int(R * 2.8 * s), max(3, int(4 * s))
        x0, y0 = cx - bw // 2, cy - int((R * 1.6 + 16) * s)
        hull_frac = convoy.hull / max(1, convoy.max_hull)
        for i, (f, c) in enumerate(((frac, (110, 220, 255)),
                                    (hull_frac, (120, 230, 140) if hull_frac > 0.35 else (255, 120, 110)))):
            pygame.draw.rect(surf, (30, 34, 48), (x0, y0 + i * (bh + 2), bw, bh))
            pygame.draw.rect(surf, c, (x0, y0 + i * (bh + 2), int(bw * f), bh))

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
        # Flags: small triangle markers at the edge, so the objective is always findable
        for team, flag in self.env.flags.items():
            if self._visible(flag.pos, cam, 20):
                continue
            dx, dy = self._wrapped_delta(cam, flag.pos)
            px, py = dx * self.scale, dy * self.scale
            if abs(px) < 1e-6 and abs(py) < 1e-6:
                continue
            t = min((cx - margin) / max(abs(px), 1e-6), (cy - margin) / max(abs(py), 1e-6))
            ex, ey = int(cx + px * t), int(cy + py * t)
            col = TEAM_COLORS[team]["hull"]
            pygame.draw.line(surf, (220, 225, 235), (ex, ey + 6), (ex, ey - 12), 2)
            pygame.draw.polygon(surf, col, [(ex, ey - 12), (ex + 11, ey - 8), (ex, ey - 4)])
        # Stations: square markers, so the objective is always findable
        for team, station in self.env.stations.items():
            if not station.active or self._visible(station.pos, cam, station.radius):
                continue
            dx, dy = self._wrapped_delta(cam, station.pos)
            px, py = dx * self.scale, dy * self.scale
            if abs(px) < 1e-6 and abs(py) < 1e-6:
                continue
            t = min((cx - margin) / max(abs(px), 1e-6), (cy - margin) / max(abs(py), 1e-6))
            ex, ey = int(cx + px * t), int(cy + py * t)
            col = TEAM_COLORS[team]["hull"]
            pygame.draw.rect(surf, col, (ex - 7, ey - 7, 14, 14), 2)
            pygame.draw.rect(surf, col, (ex - 3, ey - 3, 6, 6))

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
        for team, flag in env.flags.items():
            x, y = mp(flag.base)
            pygame.draw.circle(surf, TEAM_COLORS[team]["trail"], (x, y), 5, 1)
            fx, fy = mp(flag.pos)
            pygame.draw.polygon(surf, TEAM_COLORS[team]["hull"], [(fx, fy - 6), (fx + 6, fy - 3), (fx, fy)])
        for team, station in env.stations.items():
            half = max(3, int(station.radius * k))
            x, y = mp(station.pos)
            color = TEAM_COLORS[team]["hull"] if station.active else (80, 80, 92)
            pygame.draw.rect(surf, color, (x - half, y - half, 2 * half, 2 * half), 0 if station.active else 1)
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
        extra = {RED: f"hp {env.team_hp(RED)}", BLUE: f"hp {env.team_hp(BLUE)}"}
        for team, station in env.stations.items():
            extra[team] = (f"station {station.health_fraction:.0%}" if station.active
                           else "station lost")
        if env.mode == "deathmatch":
            extra = {team: f"kills {env.team_kills[team]}" for team in (RED, BLUE)}
        elif env.mode == "ctf":
            extra = {team: f"captures {env.captures[team]}/{env.config.captures_to_win}"
                     for team in (RED, BLUE)}
        elif env.mode == "escort":
            c = env.convoy
            extra = {RED: "attacking", BLUE: (f"convoy {env.convoy_progress:.0%}"
                                              + ("" if c.moving or not c.active else " HALTED"))}
        surf.blit(self.font.render(f"RED {red}  {extra[RED]}", True,
                                   TEAM_COLORS[RED]["hull"]), (14, 12))
        label = self.font.render(f"{extra[BLUE]}  BLUE {blue}", True, TEAM_COLORS[BLUE]["hull"])
        surf.blit(label, (surf.get_width() - label.get_width() - 14, 12))

        def clock(frames):
            secs = int(frames // 60)
            return f"{secs // 60}:{secs % 60:02d}"

        limit = env.config.max_frames
        center = clock(env.frame) if limit is None else f"{clock(env.frame)} / {clock(limit)}"
        if env.mode == "survival":
            if env.wave_cooldown is not None:
                center = f"Wave {env.wave + 1} in {env.wave_cooldown / 60:.0f}s   {center}"
            else:
                center = f"Wave {env.wave}  ·  {red} left   {center}"
        if self.hud_text is not None:
            center = self.hud_text()
        bright = env.mode in ("survival", "roguelite") or self.hud_text is not None
        t = self.font.render(center, True, HUD_TEXT if bright else HUD_DIM)
        surf.blit(t, ((surf.get_width() - t.get_width()) // 2, 12))
        if env.mode == "roguelite" and not env.wave_active and not env.game_over and env.frame > 0:
            m = self.font_big.render(f"WAVE {env.wave} CLEARED", True, (150, 235, 170))
            surf.blit(m, ((surf.get_width() - m.get_width()) // 2,
                          (surf.get_height() - m.get_height()) // 2))
        # The followed ship is down: say when (or whether) it comes back
        followed = env.ships.get(self.follow) if self.follow else None
        if followed is not None and not followed.active and not env.game_over:
            if followed.respawn_timer is not None:
                msg = f"Respawning in {followed.respawn_timer / 60:.1f}s"
            elif env.mode == "siege":
                msg = "Station destroyed: no respawn"
            elif env.mode == "survival":
                msg = "Down. You return when the wave is cleared"
            else:
                msg = "Destroyed"
            m = self.font_big.render(msg, True, HUD_TEXT)
            surf.blit(m, ((surf.get_width() - m.get_width()) // 2, int(surf.get_height() * 0.62)))
        if limit is not None:  # no clock bar when the game has no time limit
            bar_w = surf.get_width() - 28
            frac = min(1.0, env.frame / max(1, limit))
            pygame.draw.rect(surf, (40, 44, 60), (14, 34, bar_w, 3))
            pygame.draw.rect(surf, HUD_DIM, (14, 34, int(bar_w * frac), 3))
        if env.game_over:
            msg = {BLUE: "BLUE WINS", RED: "RED WINS"}.get(env.winner, "DRAW")
            if env.mode == "survival":
                n = env.waves_cleared
                msg = f"SURVIVED {n} WAVE{'S' if n != 1 else ''}"
            elif env.mode == "roguelite":
                msg = f"RUN OVER · WAVE {env.wave}"
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
