"""App screens: main menu, match setup, game (+ pause), results, settings, gallery."""
from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pygame
import pygame_gui
from pygame_gui.elements import (UIButton, UIDropDownMenu, UIHorizontalSlider, UILabel,
                                 UITextBox)

from ..checkpoints import discover
from ..config import ARENA_SIZES, MODE_NAMES, MODE_TIME_LIMIT, MODES
from ..render import TEAM_COLORS, Renderer
from ..env import BLUE, RED
from ..session import DIFFICULTY, HUMAN, KEEP, PHYSICS_FPS, MatchSpec, agent_names, build_config, build_match
from .main import BG, Screen
from .settings import WINDOW_SIZES

# ----------------------------------------------------------------------------- vocabulary

MODE_OPTIONS = [(MODE_NAMES[m], m) for m in MODES]
ARENA_OPTIONS = [(f"{name.capitalize()} ({size:.0f})", name) for name, size in ARENA_SIZES.items()]
SPAWN_OPTIONS = [("Fixed (facing off)", True), ("Random", False)]
BOT = "heuristic_1.0"  # default AI in modes the trained agents haven't learned
# Modes with an objective the trained agents can't see; fighting modes use them as-is
OBJECTIVE_MODES = ("siege", "survival")
COMBAT_MODES = ("deathmatch", "sudden_death")

SCRIPTED = [
    ("Bot: heuristic", "heuristic_1.0"),
    ("Bot: heuristic 75%", "heuristic_0.75"),
    ("Bot: heuristic 50%", "heuristic_0.5"),
    ("Bot: heuristic 25%", "heuristic_0.25"),
    ("Bot: evasive", "evasion"),
    ("Bot: target dummy", "dummy"),
]
DIFFICULTY_OPTIONS = [("Easy", "easy"), ("Normal", "normal"), ("Hard", "hard"),
                      ("Brutal (no handicap)", "brutal")]
VIEW_OPTIONS = [("Close", 800.0), ("Medium", 1200.0), ("Wide", 1800.0)]
TEAM_SIZE_OPTIONS = [(str(n), n) for n in range(1, 5)]
TIME_OPTIONS = [("No limit", None), ("1 minute", 60.0), ("2 minutes", 120.0),
                ("3 minutes", 180.0), ("5 minutes", 300.0)]
ROCK_OPTIONS = [("Split when shot", True), ("Solid (cover)", False)]
MAX_ASTEROIDS = 60


def ship_label(agent: str) -> str:
    team, idx = agent.split("_")
    return f"{team.capitalize()} {int(idx) + 1}"


def format_time(max_frames: Optional[int]) -> str:
    if max_frames is None:
        return "no limit"
    secs = max_frames / PHYSICS_FPS
    return f"{secs / 60:.0f} min" if secs >= 60 and secs % 60 == 0 else f"{secs:.0f} s"


_MODELS_CACHE: Optional[list] = None


def models(refresh: bool = False) -> list:
    """Compatible checkpoints, scanned once and re-scanned on request (menu / setup entry)."""
    global _MODELS_CACHE
    if refresh or _MODELS_CACHE is None:
        _MODELS_CACHE = discover()
    return _MODELS_CACHE


def controller_options() -> List[Tuple[str, str]]:
    """(label, controller) for every choice: you, compatible trained models, scripted bots."""
    options = [("You (keyboard)", HUMAN)]
    options += [(f"AI: {c.label}", c.path) for c in models()]
    options += SCRIPTED
    return options


def best_model(rank: int = 0) -> str:
    found = models()
    if not found:
        return "heuristic_1.0"
    return found[min(rank, len(found) - 1)].path


def controller_label(controller: str) -> str:
    for label, value in controller_options():
        if value == controller:
            return label
    return os.path.basename(controller)


def valid_controller(controller: str) -> bool:
    return any(controller == value for _, value in controller_options())


# --------------------------------------------------------------------------- widgets


def draw_card(surface: pygame.Surface, rect: pygame.Rect, alpha: int = 225):
    card = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(card, (11, 15, 28, alpha), card.get_rect(), border_radius=10)
    pygame.draw.rect(card, (43, 57, 86, 255), card.get_rect(), width=1, border_radius=10)
    surface.blit(card, rect.topleft)


class Choice:
    """Dropdown over (label, value) pairs; maps the selected label back to its value."""

    def __init__(self, element: UIDropDownMenu, options: Sequence[Tuple[str, Any]],
                 on_change: Callable[[Any], None]):
        self.element = element
        self.options = list(options)
        self.on_change = on_change

    @property
    def value(self):
        selected = self.element.selected_option
        text = selected[0] if isinstance(selected, tuple) else selected
        for label, value in self.options:
            if label == text:
                return value
        return self.options[0][1]


class MenuScreen(Screen):
    """Screen with helpers for widgets that dispatch to callbacks."""

    def enter(self):
        self._buttons: Dict[Any, Callable[[], None]] = {}
        self._choices: Dict[Any, Choice] = {}
        self._sliders: Dict[Any, Callable[[int], None]] = {}
        self.cards: List[pygame.Rect] = []
        super().enter()

    # widget helpers --------------------------------------------------------
    @property
    def manager(self):
        return self.app.manager

    def label(self, rect, text, oid="#field_label"):
        return UILabel(pygame.Rect(rect), text, self.manager, container=self.root, object_id=oid)

    def button(self, rect, text, callback, oid=None):
        b = UIButton(pygame.Rect(rect), text, self.manager, container=self.root, object_id=oid)
        self._buttons[b] = callback
        return b

    def choice(self, rect, options, value, on_change):
        labels = [text for text, _ in options]
        start = next((text for text, v in options if v == value), labels[0])
        element = UIDropDownMenu(labels, start, pygame.Rect(rect), self.manager, container=self.root)
        c = Choice(element, options, on_change)
        self._choices[element] = c
        return c

    def slider(self, rect, value, value_range, on_change):
        s = UIHorizontalSlider(pygame.Rect(rect), value, value_range, self.manager,
                               container=self.root, click_increment=1)
        self._sliders[s] = on_change
        return s

    def text(self, rect, html, oid=None):
        return UITextBox(html, pygame.Rect(rect), self.manager, container=self.root, object_id=oid)

    def card(self, rect) -> pygame.Rect:
        r = pygame.Rect(rect)
        self.cards.append(r)
        return r

    # dispatch ----------------------------------------------------------------
    def handle_event(self, event):
        if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element in self._buttons:
            self._buttons[event.ui_element]()
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element in self._choices:
            c = self._choices[event.ui_element]
            c.on_change(c.value)
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED and event.ui_element in self._sliders:
            self._sliders[event.ui_element](int(event.value))
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.back()

    def back(self):
        self.app.pop()

    def draw(self, surface):
        super().draw(surface)
        for r in self.cards:
            draw_card(surface, r)

    def center_x(self, width: int) -> int:
        return (self.app.size[0] - width) // 2


# ------------------------------------------------------------------------- main menu


class MainMenuScreen(MenuScreen):
    def build(self):
        w, h = self.app.size
        col = 320
        x = self.center_x(col)
        y = int(h * 0.2)
        self.label((0, y - 6, w, 84), "MARL ASTEROIDS", "#title")
        self.label((0, y + 74, w, 30), "Multi-agent reinforcement learning, in space", "#subtitle")
        y += 140
        self.card((x - 30, y - 20, col + 60, 4 * 62 + 30))
        for text, cb in (("Play", lambda: self.app.push(SetupScreen(self.app))),
                         ("Gallery", lambda: self.app.push(GalleryScreen(self.app))),
                         ("Settings", lambda: self.app.push(SettingsScreen(self.app))),
                         ("Quit", self.app.quit)):
            self.button((x, y, col, 50), text, cb, "#menu_button")
            y += 62
        n = len(models(refresh=True))
        note = (f"{n} trained agents available" if n else
                "No compatible trained agents found in models/ (bots only)")
        self.label((0, h - 40, w, 24), note, "#note")

    def back(self):
        self.app.quit()


# ------------------------------------------------------------------------- match setup


def is_trained(controller: str) -> bool:
    return controller.endswith(".zip")


def default_spec() -> MatchSpec:
    """You plus the strongest AI wingman against two of the strongest AI, 2v2 deathmatch."""
    ai = best_model(0)
    return MatchSpec(mode="deathmatch", arena="small", m_blue=2, n_red=2, difficulty="normal",
                     default_controller=ai,
                     controllers={"blue_0": HUMAN, "blue_1": ai, "red_0": ai, "red_1": ai})


def sanitize(spec: MatchSpec) -> MatchSpec:
    """Make a saved setup safe to use: unknown modes/sizes or missing models fall back."""
    if spec.arena not in ARENA_SIZES or spec.mode not in MODES:
        return default_spec()
    for side in ("m_blue", "n_red"):
        value = getattr(spec, side)
        if not isinstance(value, int) or not 1 <= value <= 4:
            setattr(spec, side, 2)
    if not valid_controller(spec.default_controller):
        spec.default_controller = best_model(0) if spec.mode in COMBAT_MODES else BOT
    for agent, controller in list(spec.controllers.items()):
        if not valid_controller(controller):
            spec.controllers[agent] = spec.default_controller
    if spec.difficulty is not None and spec.difficulty not in DIFFICULTY:
        spec.difficulty = "normal"
    if spec.mode == "deathmatch" and spec.time_limit is None:
        spec.time_limit = KEEP  # respawning deathmatch only ends on the clock
    return spec


class SetupScreen(MenuScreen):
    ROW = 44

    def __init__(self, app):
        super().__init__(app)
        models(refresh=True)  # pick up checkpoints trained since the app started
        saved = app.settings.last_setup
        self.spec = sanitize(MatchSpec.from_dict(saved)) if saved else default_spec()
        self._sync_controllers()
        self.status = ""
        self.status_is_error = False
        self._pending_start = 0

    # spec helpers --------------------------------------------------------------
    def _cfg(self):
        return build_config(self.spec)

    def _sync_controllers(self):
        """Keep exactly one controller per ship; new ships get the default AI."""
        agents = agent_names(self._cfg())
        self.spec.controllers = {a: self.spec.controllers.get(a, self.spec.default_controller)
                                 for a in agents}

    def _set(self, field, value, rebuild=False):
        setattr(self.spec, field, value)
        if field in ("n_red", "m_blue"):
            self._sync_controllers()
        self.status = ""
        if rebuild:
            self.rebuild()
        else:
            self._refresh_preview()

    def _set_mode(self, mode):
        self.spec.mode = mode
        self.spec.time_limit = KEEP  # each mode has its own sensible default clock
        self.status, self.status_is_error = "", False
        if mode in OBJECTIVE_MODES:
            # The trained agents only know combat: they ignore stations and waves entirely.
            # Swap them for objective-aware bots so the mode actually gets played.
            swapped = [a for a, c in self.spec.controllers.items() if is_trained(c)]
            for a in swapped:
                self.spec.controllers[a] = BOT
            if is_trained(self.spec.default_controller):
                self.spec.default_controller = BOT
            if swapped:
                self.status = ("Trained agents only know combat modes, so AI ships now use "
                               "bots. You can still pick them.")
        self.rebuild()

    def _set_arena(self, arena):
        self.spec.num_asteroids = None  # new arena, its own default rock count
        self._set("arena", arena, rebuild=True)

    def _set_controller(self, agent, controller):
        if controller == HUMAN:  # one keyboard: picking yourself here frees any other ship
            for other, c in self.spec.controllers.items():
                if c == HUMAN and other != agent:
                    self.spec.controllers[other] = self.spec.default_controller
        self.spec.controllers[agent] = controller
        self.status = ""
        self.rebuild()

    # layout --------------------------------------------------------------------
    def build(self):
        w, h = self.app.size
        cfg = self._cfg()
        mode = self.spec.mode
        self.label((60, 24, w - 120, 40), "Match setup", "#heading")

        gap = 30
        card_w = (w - 120 - gap) // 2
        left = pygame.Rect(60, 80, card_w, h - 190)
        right = pygame.Rect(60 + card_w + gap, 80, card_w, h - 190)
        self.card(left)
        self.card(right)
        lw = 150
        cw = card_w - lw - 50

        # Left: match
        x, y = left.x + 20, left.y + 16
        self.label((x, y, 300, 30), "Match", "#heading")
        y += 42

        def row(name):
            nonlocal y
            self.label((x, y, lw, 34), name)
            r = (x + lw, y, cw, 34)
            y += self.ROW
            return r

        self.choice(row("Mode"), MODE_OPTIONS, mode, self._set_mode)
        # Tall enough for two lines at the narrowest window, so it never needs a scrollbar
        self.text((x - 4, y - 10, card_w - 32, 66),
                  f"<font color='#8a98b8'>{MODES[mode]}</font>", "#mode_blurb")
        y += 50
        self.choice(row("Arena size"), ARENA_OPTIONS, self.spec.arena, self._set_arena)
        self.choice(row("Blue ships"), TEAM_SIZE_OPTIONS, cfg.m_blue,
                    lambda v: self._set("m_blue", v, rebuild=True))
        red_label = "Max per wave" if mode == "survival" else "Red ships"
        self.choice(row(red_label), TEAM_SIZE_OPTIONS, cfg.n_red,
                    lambda v: self._set("n_red", v, rebuild=True))
        if mode in COMBAT_MODES:
            self.choice(row("Spawns"), SPAWN_OPTIONS, self.spec.fixed_spawns,
                        lambda v: self._set("fixed_spawns", v))

        r = row("Asteroids")
        slider_w = r[2] - 56
        count = min(MAX_ASTEROIDS, cfg.num_asteroids)
        self.slider((r[0], r[1] + 4, slider_w, 26), count, (0, MAX_ASTEROIDS), self._set_asteroids)
        self.asteroid_value = self.label((r[0] + slider_w + 10, r[1], 46, 34), str(count))

        self.choice(row("Rocks"), ROCK_OPTIONS, cfg.asteroids_split,
                    lambda v: self._set("asteroids_split", v))
        default_frames = None if MODE_TIME_LIMIT[mode] is None else int(MODE_TIME_LIMIT[mode] * 60)
        time_opts = [(f"Default ({format_time(default_frames)})", KEEP)] + [
            o for o in TIME_OPTIONS if not (mode == "deathmatch" and o[1] is None)]  # needs a clock
        self.choice(row("Time limit"), time_opts, self.spec.time_limit,
                    lambda v: self._set("time_limit", v))
        self.preview = self.label((x, y + 4, card_w - 40, 26), "", "#hint")
        self._refresh_preview()

        # Right: ships. Rows tighten so up to 4v4 plus the 3 settings rows and hint fit the card.
        x, y = right.x + 20, right.y + 16
        self.label((x, y, 300, 30), "Ships", "#heading")
        y += 42
        options = controller_options()
        ships = agent_names(cfg)
        room = right.bottom - y - 3 * self.ROW - 56
        ship_row = max(36, min(self.ROW, room // max(1, len(ships))))
        for agent in ships:
            oid = "#blue_label" if agent.startswith("blue") else "#red_label"
            self.label((x, y, lw, 32), ship_label(agent), oid)
            self.choice((x + lw, y, cw, 32), options, self.spec.controllers[agent],
                        lambda v, a=agent: self._set_controller(a, v))
            y += ship_row
        y += 8
        humans = [a for a, c in self.spec.controllers.items() if c == HUMAN]
        diff = self.spec.difficulty or "normal"
        self.label((x, y, lw, 34), "Difficulty")
        self.choice((x + lw, y, cw, 34), DIFFICULTY_OPTIONS, diff,
                    lambda v: self._set("difficulty", v))
        y += self.ROW
        cam_opts = [("Auto" + (" (your ship)" if humans else " (whole map)"), "auto"),
                    ("Whole map", "none")] + [(f"Follow {ship_label(a)}", a) for a in ships]
        follow = self.spec.follow if any(v == self.spec.follow for _, v in cam_opts) else "auto"
        self.label((x, y, lw, 34), "Camera")
        self.choice((x + lw, y, cw, 34), cam_opts, follow, lambda v: self._set("follow", v))
        y += self.ROW
        self.label((x, y, lw, 34), "Zoom")
        self.choice((x + lw, y, cw, 34), VIEW_OPTIONS, self.spec.view, lambda v: self._set("view", v))
        y += self.ROW
        if mode in OBJECTIVE_MODES and any(is_trained(c) for c in self.spec.controllers.values()):
            hint = "Trained agents ignore the objective in this mode (they only know combat)."
        elif humans:
            hint = "Difficulty slows AI reactions and reloads while you play."
        else:
            hint = "No ship is yours: AI plays unhandicapped. Pick 'You' to play."
        self.label((x, y, card_w - 40, 26), hint, "#hint")

        # Footer
        self.button((60, h - 84, 160, 46), "Back", self.back)
        self.button((w - 60 - 220, h - 84, 220, 46), "Start", self.start, "#primary_button")
        self.status_label = self.label((240, h - 80, w - 540, 36), self.status,
                                       "#error" if self.status_is_error else "#hint")

    def _set_asteroids(self, n):
        self.spec.num_asteroids = n
        self.asteroid_value.set_text(str(n))
        self._refresh_preview()

    def _refresh_preview(self):
        try:
            cfg = self._cfg()
        except ValueError as e:
            self.preview.set_text(str(e))
            return
        rocks = f"{cfg.num_asteroids} asteroid{'s' if cfg.num_asteroids != 1 else ''}"
        if cfg.num_asteroids:
            rocks += ", splitting" if cfg.asteroids_split else ", solid"
        self.preview.set_text(f"{cfg.width:.0f} x {cfg.height:.0f} · {rocks} · "
                              f"{format_time(cfg.max_frames)}")

    # start ---------------------------------------------------------------------
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            self.start()
            return
        super().handle_event(event)

    def start(self):
        self.status, self.status_is_error = "Loading agents...", False
        self.status_label.set_text(self.status)
        self._pending_start = 2  # let one frame draw "Loading" before blocking on model loads

    def update(self, dt):
        if not self._pending_start:
            return
        self._pending_start -= 1
        if self._pending_start:
            return
        spec = copy.deepcopy(self.spec)
        if spec.difficulty is None:
            spec.difficulty = "normal"
        try:
            match = build_match(spec)
        except Exception as e:  # bad setup or unreadable checkpoint: report, don't crash
            self.status, self.status_is_error = f"Can't start: {e}", True
            self.rebuild()
            return
        self.app.settings.last_setup = self.spec.to_dict()
        self.app.save_settings()
        self.status = ""
        self.app.push(GameScreen(self.app, match))


# -------------------------------------------------------------------------------- game


class GameScreen(MenuScreen):
    END_HOLD_SECONDS = 1.4

    def __init__(self, app, match, game_index: int = 0):
        super().__init__(app)
        self.match = match
        self.game_index = game_index
        self.env = None
        self.renderer: Optional[Renderer] = None
        self.paused = False
        self.end_hold = None  # physics frames left to show the result banner
        self.frame_surface: Optional[pygame.Surface] = None
        self.font = None

    # lifecycle -----------------------------------------------------------------
    def enter(self):
        super().enter()
        if self.env is None:
            self._start_game()

    def _start_game(self):
        w, h = self.app.size
        s = self.app.settings
        self.side = min(w, h)
        self.env = self.match.make_env()
        self.renderer = Renderer(self.env, window=self.side, follow=self.match.follow,
                                 view_size=self.match.spec.view, effects=s.effects,
                                 minimap=s.minimap, show_obs=s.show_obs)
        self.env.on_frame = self._on_frame
        self.who = {}  # roster labels, computed once rather than every frame
        for agent, controller in self.match.assignment.items():
            who = "You" if controller == HUMAN else controller_label(controller)
            self.who[agent] = who.replace("AI: ", "").replace("Bot: ", "bot: ")
        self.match.reset(self.game_index)
        self.paused = False
        self.end_hold = None
        self.font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 15)
        self.font_big = pygame.font.SysFont("dejavusansmono,consolas,monospace", 20, bold=True)

    def exit(self):
        super().exit()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.match.close()
        self.env = None

    # pause -----------------------------------------------------------------------
    def build(self):
        if not self.paused:
            return
        w, h = self.app.size
        cw = 300
        x = self.center_x(cw)
        y = h // 2 - 150
        self.card((x - 30, y - 60, cw + 60, 330))
        self.label((x, y - 48, cw, 40), "Paused", "#heading")
        for text, cb in (("Resume", self.resume), ("Restart", self.restart),
                         ("Change setup", self.to_setup), ("Main menu", self.app.home)):
            self.button((x, y, cw, 46), text, cb, "#menu_button")
            y += 58

    def pause(self):
        self.paused = True
        self.rebuild()

    def resume(self):
        self.paused = False
        self.rebuild()

    def restart(self):
        self.game_index += 1
        self.match.reset(self.game_index)
        self.end_hold = None
        self.resume()

    def to_setup(self):
        self.app.pop()

    def back(self):
        self.resume() if self.paused else self.pause()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB and self.renderer:
            self.renderer.show_obs = not self.renderer.show_obs
            return
        super().handle_event(event)

    # frames ----------------------------------------------------------------------
    def drives_frames(self) -> bool:
        return not self.paused

    def advance(self):
        env = self.env
        if self.end_hold is not None:  # game over: keep effects running under the banner
            self.app.present(self._draw)
            self.end_hold -= 1
            if self.end_hold <= 0:
                self._finish()
            return
        if env.frame == 0 and self.frame_surface is None:
            self.app.present(self._draw)  # first frame, before anyone moves
        self.match.step()  # draws and paces through env.on_frame
        if env.game_over:
            self.end_hold = int(self.END_HOLD_SECONDS * PHYSICS_FPS)

    def _finish(self):
        stats = self.match.stats()
        human = self.match.human_agents[0] if self.match.human_agents else None
        results = ResultsScreen(self.app, self.match, stats, self.env.winner, self.env.frame,
                                human, self.game_index, self.env.summary())
        self.app.replace(results)

    def _on_frame(self):
        self.app.present(self._draw)

    def _draw(self, surface):
        self.frame_surface = self.renderer.render("surface")
        self._compose(surface, self.frame_surface)

    def _compose(self, surface, frame):
        w, h = surface.get_size()
        surface.fill(BG)
        ox, oy = (w - frame.get_width()) // 2, (h - frame.get_height()) // 2
        surface.blit(frame, (ox, oy))
        margin = ox
        if margin >= 170:
            self._draw_roster(surface, BLUE, pygame.Rect(16, 20, margin - 32, h - 40))
            self._draw_roster(surface, RED, pygame.Rect(ox + frame.get_width() + 16, 20,
                                                        margin - 32, h - 40))

    def _draw_roster(self, surface, team, rect):
        env = self.env
        col = TEAM_COLORS[team]["hull"]
        name = "BLUE" if team == BLUE else "RED"
        surface.blit(self.font_big.render(name, True, col), (rect.x, rect.y))
        y = rect.y + 34
        station = env.stations.get(team)
        if station is not None:  # station health under the team name
            label = (f"Station {station.health_fraction:.0%}" if station.active else "Station lost")
            surface.blit(self.font.render(label, True, col if station.active else (150, 90, 90)),
                         (rect.x, y))
            bw = min(rect.width, 140)
            for i, (f, c) in enumerate(((station.shield / max(1, station.max_shield), (110, 220, 255)),
                                        (station.hull / max(1, station.max_hull), (120, 230, 140)))):
                pygame.draw.rect(surface, (30, 34, 48), (rect.x, y + 22 + i * 8, bw, 5))
                pygame.draw.rect(surface, c, (rect.x, y + 22 + i * 8, int(bw * f), 5))
            y += 52
        if env.mode == "deathmatch":  # the score
            n = env.team_kills[team]
            surface.blit(self.font_big.render(f"{n} kill{'s' if n != 1 else ''}", True, col),
                         (rect.x, y))
            y += 34
        if env.mode == "survival" and team == RED:
            wave = (f"Wave {env.wave + 1} in {env.wave_cooldown / 60:.0f}s"
                    if env.wave_cooldown is not None else f"Wave {env.wave}")
            surface.blit(self.font.render(f"{wave}  ·  cleared {env.waves_cleared}", True, col),
                         (rect.x, y))
            y += 28
        for agent in env.team_agents[team]:
            ship = env.ships[agent]
            if ship.reserve:  # survival ships not in the current wave
                continue
            who = self.who[agent]
            if not ship.active and ship.respawn_timer is not None:
                who = f"respawn in {ship.respawn_timer / 60:.1f}s"
            color = col if ship.active else (90, 96, 112)
            surface.blit(self.font.render(ship_label(agent), True, color), (rect.x, y))
            surface.blit(self.font.render(who[:max(8, rect.width // 9)], True, (150, 160, 182)),
                         (rect.x, y + 18))
            # hull + shield pips
            px = rect.x
            for i in range(ship.max_hp):
                if i < ship.max_hull:
                    on = i < ship.hull
                    c = ((120, 230, 140) if ship.hull > 1 else (255, 120, 110)) if on else (56, 40, 44)
                else:
                    c = (110, 220, 255) if (i - ship.max_hull) < ship.shield else (34, 48, 62)
                pygame.draw.rect(surface, c if ship.active else (40, 42, 50), (px, y + 40, 14, 6))
                px += 18
            y += 62
        hints = ["Esc  pause", "Tab  sensor view"]
        if team == BLUE and self.match.human_agents:
            hints = ["W / Up     thrust", "A D / arrows turn", "Space      shoot"] + hints
        for i, line in enumerate(reversed(hints)):
            surface.blit(self.font.render(line, True, (110, 120, 145)),
                         (rect.x, rect.bottom - 20 - i * 20))

    def draw(self, surface):  # paused: frozen frame, dimmed, under the pause card
        if self.frame_surface is not None:
            self._compose(surface, self.frame_surface)
        else:
            surface.fill(BG)
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((4, 6, 16, 160))
        surface.blit(dim, (0, 0))
        for r in self.cards:
            draw_card(surface, r)


# ------------------------------------------------------------------------------ results


class ResultsScreen(MenuScreen):
    def __init__(self, app, match, stats, winner, frames, human, game_index, summary=None):
        super().__init__(app)
        self.match, self.stats = match, stats
        self.summary = summary or {"mode": match.config.mode}
        self.winner, self.frames, self.human, self.game_index = winner, frames, human, game_index

    def build(self):
        w, h = self.app.size
        cw = 560
        x = self.center_x(cw)
        y = int(h * 0.16)
        mode = self.summary.get("mode", "sudden_death")
        rows = [("Kills", "kills"), ("Damage dealt", "damage"),
                ("Team kills", "teamkills"), ("Crashes", "env_deaths")]
        if mode == "siege":
            rows.insert(1, ("Station damage", "station_damage"))
        # Card hugs the content: title, subtitle, header row, stat rows, buttons
        self.card((x - 30, y - 20, cw + 60, 20 + 118 + 30 * len(rows) + 56 + 46 + 30))
        if mode == "survival":
            n = self.summary.get("waves_cleared", 0)
            title = f"Survived {n} wave{'s' if n != 1 else ''}"
        elif self.human:
            my_team = BLUE if self.human.startswith("blue") else RED
            title = ("Victory" if self.winner == my_team else
                     "Draw" if self.winner == -1 else "Defeat")
        else:
            title = {BLUE: "Blue wins", RED: "Red wins"}.get(self.winner, "Draw")
        self.label((x, y - 6, cw, 80), title, "#title")
        how = self._how()
        # Deathmatch always runs to the clock, so its duration says nothing; the score does
        sub = how if mode == "deathmatch" else f"{how} after {self.frames / PHYSICS_FPS:.1f} s"
        self.label((x, y + 70, cw, 28), sub, "#subtitle")

        # Stats as a label grid: name column left-aligned, numbers right-aligned under each team
        s = self.stats
        ty = y + 118
        name_w, num_w, pad = 220, 90, 60
        tx = x + (cw - (name_w + 2 * num_w + pad)) // 2
        self.label((tx + name_w, ty, num_w, 26), "Blue", "#blue_head")
        self.label((tx + name_w + num_w + pad, ty, num_w, 26), "Red", "#red_head")
        for name, key in rows:
            ty += 30
            if key == "station_damage":
                dealt = self.summary.get("station_damage", {})
                d = {"blue": dealt.get(BLUE, 0), "red": dealt.get(RED, 0)}
            else:
                d = getattr(s, key)
            self.label((tx, ty, name_w, 26), name, "#stat_name")
            self.label((tx + name_w, ty, num_w, 26), str(d["blue"]), "#stat")
            self.label((tx + name_w + num_w + pad, ty, num_w, 26), str(d["red"]), "#stat")

        y = ty + 56
        bw = (cw - 20) // 3
        self.button((x, y, bw, 46), "Play again", self.again, "#primary_button")
        self.button((x + bw + 10, y, bw, 46), "Change setup", self.app.pop)
        self.button((x + 2 * (bw + 10), y, bw, 46), "Main menu", self.app.home)

    def _how(self) -> str:
        cfg = self.match.config
        mode = self.summary.get("mode", "sudden_death")
        timed_out = cfg.max_frames is not None and self.frames >= cfg.max_frames
        if mode == "deathmatch":
            kills = self.summary.get("kills", {})
            b, r = kills.get(BLUE, 0), kills.get(RED, 0)
            if self.winner == -1:
                return f"Level on kills when time ran out: {b} to {r}"
            return f"Most kills when time ran out: {max(b, r)} to {min(b, r)}"
        if mode == "survival":
            if timed_out:
                return "Held out until the time limit"
            return f"Overrun during wave {self.summary.get('wave', 1)}"
        if self.winner == -1:
            return "Out of time, even health" if timed_out else "Mutual destruction"
        if timed_out:
            return ("Won on station health at the time limit" if mode == "siege"
                    else "Won on health at the time limit")
        if mode == "siege":
            return "Station destroyed and fleet wiped out"
        return "Fleet destroyed"

    def again(self):
        self.app.replace(GameScreen(self.app, self.match, self.game_index + 1))

    def back(self):
        self.app.home()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.again()
            return
        super().handle_event(event)


# ----------------------------------------------------------------------------- settings


class SettingsScreen(MenuScreen):
    def build(self):
        w, h = self.app.size
        s = self.app.settings
        cw = 520
        x = self.center_x(cw)
        y = int(h * 0.15)
        self.card((x - 30, y - 20, cw + 60, 540))
        self.label((x, y, cw, 40), "Settings", "#heading")
        y += 56
        lw, fw = 220, cw - 220

        sizes = [(f"{a} x {b}", (a, b)) for a, b in WINDOW_SIZES]
        current = tuple(self.app.size)
        if current not in [v for _, v in sizes]:
            sizes.append((f"{current[0]} x {current[1]}", current))
        self.label((x, y, lw, 34), "Window size")
        self.choice((x + lw, y, fw, 34), sizes, current, self.app.set_window_size)
        y += 50
        on_off = [("On", True), ("Off", False)]
        for label, key in (("Visual effects", "effects"), ("Minimap", "minimap"),
                           ("Sensor view by default", "show_obs"), ("Menu background match", "attract_mode")):
            self.label((x, y, lw, 34), label)
            self.choice((x + lw, y, fw, 34), on_off, getattr(s, key),
                        lambda v, k=key: self._toggle(k, v))
            y += 50
        self.button((x, y + 10, cw, 40), "Forget saved match setup", self._forget)
        path = str(self._path()).replace(os.path.expanduser("~"), "~", 1)
        if len(path) > 60:  # keep the file name visible
            path = path[:22] + "..." + path[-35:]
        self.label((x, y + 56, cw, 30), f"Saved to {path}", "#hint")
        self.button((x, y + 100, 160, 46), "Back", self.back)

    def _path(self):
        from .settings import config_path
        return self.app.settings_path or config_path()

    def _toggle(self, key, value):
        setattr(self.app.settings, key, value)
        if key in ("effects", "attract_mode"):
            self.app._background = None
        self.app.save_settings()

    def _forget(self):
        self.app.settings.last_setup = None
        self.app.save_settings()


# ------------------------------------------------------------------------------ gallery


GALLERY_EXHIBITS = [
    ("Exploiting a fixed opponent",
     "An agent trained only against a scripted bot beats it every time, then loses to a self-play "
     "agent that scores lower against the same bot."),
    ("Self-play progression", "The same agent after 250k, 1M, 2.5M and 20M steps of self-play."),
    ("Reward design", "What an agent learns when one penalty dominates its reward."),
    ("Credit assignment", "Teams trained with individual rewards vs. a shared team reward."),
    ("Why deathmatch doesn't produce teamwork",
     "Trained teams share kills no more often than bots that never coordinate."),
    ("Partial observability", "Planned: limited sensors and what agents do when they lose contact."),
    ("Human-AI coordination", "Planned: a wingman trained to cooperate with people, not just copies "
                              "of itself."),
]


class GalleryScreen(MenuScreen):
    def build(self):
        w, h = self.app.size
        cw = min(820, w - 120)
        x = self.center_x(cw)
        y = 60
        self.card((x - 30, y - 20, cw + 60, h - 100))
        self.label((x, y, cw, 40), "MARL Gallery", "#heading")
        self.label((x, y + 40, cw, 26), "Interactive exhibits on multi-agent RL are coming. "
                                        "Planned exhibits:", "#hint")
        html = "".join(f"<b>{t}</b><br><font color='#9aa7c4'>{d}</font><br><br>"
                       for t, d in GALLERY_EXHIBITS)
        self.text((x, y + 76, cw, h - 260), html)
        self.button((x, h - 120, 160, 46), "Back", self.back)
