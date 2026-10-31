"""App screens: main menu, experimental menu, match setup, game (+ pause), results, roguelite run
(game, shop, run over), settings, gallery."""
from __future__ import annotations

import copy
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pygame
import pygame_gui
from pygame_gui.elements import (UIButton, UIDropDownMenu, UIHorizontalSlider, UILabel,
                                 UITextBox)

from ..checkpoints import discover
from ..config import ARENA_SIZES, MAX_TEAM_SIZE, MODE_NAMES, MODE_TIME_LIMIT, MODES
from ..render import TEAM_COLORS, Renderer, draw_legend, draw_look, legend_height, ship_look
from ..env import BLUE, PLAYER, RED
from ..roguelite import RogueliteRun, shop_rows, wingman_tier
from ..session import DIFFICULTY, HUMAN, KEEP, PHYSICS_FPS, MatchSpec, agent_names, build_config, build_match
from .main import BG, Screen
from .settings import WINDOW_SIZES

# ----------------------------------------------------------------------------- vocabulary

MODE_OPTIONS = [(MODE_NAMES[m], m) for m in MODES]
ARENA_OPTIONS = [(f"{name.capitalize()} ({size:.0f})", name) for name, size in ARENA_SIZES.items()]
SPAWN_OPTIONS = [("Fixed (facing off)", True), ("Random", False)]
BOT = "heuristic_1.0"  # default AI in modes the trained agents haven't learned
# Modes with an objective the trained agents can't see; fighting modes use them as-is
OBJECTIVE_MODES = ("siege", "survival", "ctf", "escort")
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
TEAM_SIZE_OPTIONS = [(str(n), n) for n in range(1, MAX_TEAM_SIZE + 1)]
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
        entries = (("Roguelite survival", self.new_run),
                   ("Classic survival", lambda: self.app.push(SetupScreen(self.app, mode="survival"))),
                   ("Experimental", lambda: self.app.push(ExperimentalMenuScreen(self.app))),
                   ("Settings", lambda: self.app.push(SettingsScreen(self.app))),
                   ("Quit", self.app.quit))
        self.card((x - 30, y - 20, col + 60, len(entries) * 62 + 30))
        for text, cb in entries:
            self.button((x, y, col, 50), text, cb, "#menu_button")
            y += 62
        self.error = getattr(self, "error", "")
        if self.error:
            note = self.error
        else:
            n = len(models(refresh=True))
            note = (f"{n} trained agents available" if n else
                    "No compatible trained agents found in models/ (bots only)")
        self.label((0, h - 40, w, 24), note, "#error" if self.error else "#note")

    def new_run(self):
        try:
            run = RogueliteRun()
        except Exception as e:  # broken tuning tables: report, don't crash
            self.error = f"Can't start a run: {e}"
            self.rebuild()
            return
        self.error = ""
        self.app.push(RogueliteGameScreen(self.app, run))

    def back(self):
        self.app.quit()


class ExperimentalMenuScreen(MenuScreen):
    """Every other mode (through the full match setup) and the MARL gallery."""

    def build(self):
        w, h = self.app.size
        col = 320
        x = self.center_x(col)
        y = int(h * 0.2)
        self.label((0, y - 6, w, 60), "Experimental", "#heading_center")
        self.label((0, y + 50, w, 30), "Every mode, any teams and AI, plus the MARL gallery. "
                                       "Less polished than the survival modes.", "#subtitle")
        y += 120
        entries = (("Custom match", lambda: self.app.push(SetupScreen(self.app))),
                   ("Gallery", lambda: self.app.push(GalleryScreen(self.app))),
                   ("Back", self.back))
        self.card((x - 30, y - 20, col + 60, len(entries) * 62 + 30))
        for text, cb in entries:
            self.button((x, y, col, 50), text, cb, "#menu_button")
            y += 62


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
        if not isinstance(value, int) or not 1 <= value <= MAX_TEAM_SIZE:
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

    def __init__(self, app, mode: Optional[str] = None):
        super().__init__(app)
        models(refresh=True)  # pick up checkpoints trained since the app started
        saved = app.settings.last_setup
        self.spec = sanitize(MatchSpec.from_dict(saved)) if saved else default_spec()
        self._sync_controllers()
        self.status = ""
        self.status_is_error = False
        self._pending_start = 0
        if mode is not None and mode != self.spec.mode:
            self._apply_mode(mode)  # e.g. "Classic survival" on the main menu

    # spec helpers --------------------------------------------------------------
    def _cfg(self):
        return build_config(self.spec)

    def _sync_controllers(self):
        """Keep exactly one controller per ship; new ships get the default AI. In survival every
        enemy shares one controller (set with the Enemies dropdown), and you can't be one."""
        agents = agent_names(self._cfg())
        self.spec.controllers = {a: self.spec.controllers.get(a, self.spec.default_controller)
                                 for a in agents}
        if self.spec.mode == "survival":
            enemy = self.spec.controllers.get("red_0", BOT)
            if enemy == HUMAN:
                enemy = BOT
            for a in agents:
                if a.startswith("red"):
                    self.spec.controllers[a] = enemy

    def _set_enemies(self, controller):
        for a in self.spec.controllers:
            if a.startswith("red"):
                self.spec.controllers[a] = controller
        self.status = ""
        self.rebuild()

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
        self._apply_mode(mode)
        self.rebuild()

    def _apply_mode(self, mode):
        """Switch modes without touching widgets (usable before the screen is shown)."""
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
        self._sync_controllers()  # survival brings its enemy pool; other modes trim back

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
        if mode != "survival":  # survival waves size themselves
            self.choice(row("Red ships"), TEAM_SIZE_OPTIONS, cfg.n_red,
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

        # Right: ships. Up to 8 rows in one column (labels left, dropdowns right); beyond that,
        # blue and red side by side. Survival enemies share a single dropdown.
        x, y = right.x + 20, right.y + 16
        self.label((x, y, 300, 30), "Ships", "#heading")
        y += 42
        options = controller_options()
        ships = agent_names(cfg)
        blues = [a for a in ships if a.startswith("blue")]
        reds = [] if mode == "survival" else [a for a in ships if a.startswith("red")]
        # (label, agent or None for the survival Enemies row)
        left_rows = [(ship_label(a), a) for a in blues]
        right_rows = [(ship_label(a), a) for a in reds]
        if mode == "survival":
            right_rows.append(("Enemies", None))
        room = right.bottom - y - 3 * self.ROW - 56
        enemy_opts = [o for o in options if o[1] != HUMAN]

        def ship_choice(rect, agent):
            if agent is None:
                self.choice(rect, enemy_opts, self.spec.controllers["red_0"], self._set_enemies)
            else:
                self.choice(rect, options, self.spec.controllers[agent],
                            lambda v, a=agent: self._set_controller(a, v))

        if len(left_rows) + len(right_rows) <= 8:
            rows = left_rows + right_rows
            ship_row = max(36, min(self.ROW, room // max(1, len(rows))))
            for text, agent in rows:
                oid = "#blue_label" if agent and agent.startswith("blue") else "#red_label"
                self.label((x, y, lw, 32), text, oid)
                ship_choice((x + lw, y, cw, 32), agent)
                y += ship_row
        else:
            col_w = (card_w - 40 - 20) // 2
            slw = 74 if right_rows and right_rows[-1][1] is None else 64
            n = max(len(left_rows), len(right_rows))
            ship_row = max(36, min(self.ROW, room // n))
            for col, rows in enumerate((left_rows, right_rows)):
                cx = x + col * (col_w + 20)
                for i, (text, agent) in enumerate(rows):
                    oid = "#blue_label" if agent and agent.startswith("blue") else "#red_label"
                    self.label((cx, y + i * ship_row, slw, 32), text, oid)
                    ship_choice((cx + slw, y + i * ship_row, col_w - slw, 32), agent)
            y += n * ship_row
        y += 8
        humans = [a for a, c in self.spec.controllers.items() if c == HUMAN]
        diff = self.spec.difficulty or "normal"
        self.label((x, y, lw, 34), "Difficulty")
        self.choice((x + lw, y, cw, 34), DIFFICULTY_OPTIONS, diff,
                    lambda v: self._set("difficulty", v))
        y += self.ROW
        cam_opts = [("Auto" + (" (your ship)" if humans else " (whole map)"), "auto"),
                    ("Whole map", "none")] + [(f"Follow {ship_label(a)}", a) for a in blues + reds]
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
                                 minimap=s.minimap, show_obs=s.show_obs,
                                 thruster_particles=s.thruster_particles)
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
        if env.mode == "ctf":
            n, goal = env.captures[team], env.config.captures_to_win
            surface.blit(self.font_big.render(f"{n}/{goal} captures", True, col), (rect.x, y))
            flag = env.flags[team]
            where = ("at base" if flag.at_base else
                     f"taken by {ship_label(flag.carrier)}" if flag.carrier else
                     f"dropped, home in {flag.drop_timer / 60:.0f}s")
            surface.blit(self.font.render(f"Flag {where}", True, (150, 160, 182)), (rect.x, y + 28))
            y += 56
        if env.mode == "escort":
            if team == BLUE:
                c = env.convoy
                label = (f"Convoy {env.convoy_progress:.0%}" + ("" if c.moving else "  halted")
                         if c.active else "Convoy destroyed")
                surface.blit(self.font_big.render(label, True, col), (rect.x, y))
                bw = min(rect.width, 160)
                for i, (f, cc) in enumerate(((env.convoy_progress, (200, 210, 235)),
                                             (c.hp / max(1, c.max_hp), (120, 230, 140)))):
                    pygame.draw.rect(surface, (30, 34, 48), (rect.x, y + 30 + i * 8, bw, 5))
                    pygame.draw.rect(surface, cc, (rect.x, y + 30 + i * 8, int(bw * f), 5))
                y += 52
            else:
                surface.blit(self.font_big.render("Stop the convoy", True, col), (rect.x, y))
                y += 34
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
        hints = ["Esc  pause", "Tab  sensor view"]
        if team == BLUE and self.match.human_agents:
            hints = ["W / Up     thrust", "A D / arrows turn", "Space      shoot"] + hints
        shown = [a for a in env.team_agents[team] if not env.ships[a].reserve]  # skip idle wave slots
        avail = rect.bottom - 20 * len(hints) - 10 - y
        # Full rows (name, controller, pips) while they fit; then one line per ship; then truncate
        row_h = 62 if len(shown) * 62 <= avail else 22
        fit = max(1, avail // row_h)
        overflow = len(shown) - fit if len(shown) > fit else 0
        if overflow:
            shown = shown[:fit - 1]
        for agent in shown:
            ship = env.ships[agent]
            who = self.who[agent]
            if not ship.active and ship.respawn_timer is not None:
                who = f"respawn in {ship.respawn_timer / 60:.1f}s"
            color = col if ship.active else (90, 96, 112)
            surface.blit(self.font.render(ship_label(agent), True, color), (rect.x, y))
            if row_h == 62:
                surface.blit(self.font.render(who[:max(8, rect.width // 9)], True, (150, 160, 182)),
                             (rect.x, y + 18))
                pip_x, pip_y = rect.x, y + 40
            else:  # compact: pips on the name line
                pip_x, pip_y = rect.x + 72, y + 6
            for i in range(ship.max_hp):  # hull + shield pips
                if i < ship.max_hull:
                    on = i < ship.hull
                    c = ((120, 230, 140) if ship.hull > 1 else (255, 120, 110)) if on else (56, 40, 44)
                else:
                    c = (110, 220, 255) if (i - ship.max_hull) < ship.shield else (34, 48, 62)
                pygame.draw.rect(surface, c if ship.active else (40, 42, 50), (pip_x, pip_y, 14, 6))
                pip_x += 18
            y += row_h
        if overflow:
            surface.blit(self.font.render(f"+ {overflow + 1} more", True, (150, 160, 182)), (rect.x, y))
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
        if mode == "deathmatch":  # the score counts every enemy death, not just your shots
            rows.insert(0, ("Score", "score"))
        if mode == "ctf":
            rows.insert(0, ("Captures", "captures"))
        if mode == "escort":
            rows.insert(1, ("Convoy damage", "station_damage"))
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
        limit = self.match.config.max_frames
        timed_out = limit is not None and self.frames >= limit
        # A game that ran to the clock: its duration says nothing, the score does
        sub = how if mode == "deathmatch" or timed_out else f"{how} after {self.frames / PHYSICS_FPS:.1f} s"
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
            if key == "captures":
                caps = self.summary.get("captures", {})
                d = {"blue": caps.get(BLUE, 0), "red": caps.get(RED, 0)}
            elif key == "score":
                score = self.summary.get("kills", {})
                d = {"blue": score.get(BLUE, 0), "red": score.get(RED, 0)}
            elif key == "station_damage":
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
        if mode == "ctf":
            caps = self.summary.get("captures", {})
            b, r = caps.get(BLUE, 0), caps.get(RED, 0)
            if timed_out:
                return (f"Level on captures when time ran out: {b} to {r}" if self.winner == -1
                        else f"Most captures when time ran out: {max(b, r)} to {min(b, r)}")
            return f"Reached {cfg.captures_to_win} captures, {max(b, r)} to {min(b, r)}"
        if mode == "escort":
            pct = self.summary.get("progress", 0.0)
            if self.winner == BLUE:
                return "Convoy delivered"
            if timed_out:
                return f"Convoy stopped short at {pct:.0%} when time ran out"
            return f"Convoy destroyed at {pct:.0%} of the route"
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


# ---------------------------------------------------------------------------- roguelite

ROGUELITE_VIEW = 1000.0  # world units across the screen: a bit closer than matches, ships read better
CLEARED_HOLD_SECONDS = 1.2


def draw_pips(surface, ship, x, y, active=True):
    """Hull and shield pips in a row (the roster style)."""
    for i in range(ship.max_hp):
        if i < ship.max_hull:
            on = i < ship.hull
            c = ((120, 230, 140) if ship.hull > 1 else (255, 120, 110)) if on else (56, 40, 44)
        else:
            c = (110, 220, 255) if (i - ship.max_hull) < ship.shield else (34, 48, 62)
        pygame.draw.rect(surface, c if active else (40, 42, 50), (x, y, 12, 6))
        x += 15


class RogueliteGameScreen(GameScreen):
    """A roguelite run: waves with a shop in between. Pushes the shop when a wave is cleared and
    resumes the same env when it pops; replaces itself with the run-over screen at the end."""

    def __init__(self, app, run: RogueliteRun):
        super().__init__(app, run)
        self.run = run
        self.show_legend = True
        self.shop_hold: Optional[int] = None

    def _start_game(self):
        w, h = self.app.size
        s = self.app.settings
        self.side = min(w, h)
        self.env = self.run.make_env()
        self.renderer = Renderer(self.env, window=self.side, follow=PLAYER, view_size=ROGUELITE_VIEW,
                                 effects=s.effects, minimap=s.minimap, show_obs=s.show_obs,
                                 hud_text=self.run.hud_text, thruster_particles=s.thruster_particles)
        self.env.on_frame = self._on_frame
        self.run.start()
        self.paused = False
        self.end_hold = None
        self.shop_hold = None
        self.font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 15)
        self.font_big = pygame.font.SysFont("dejavusansmono,consolas,monospace", 20, bold=True)

    def build(self):  # pause card
        if not self.paused:
            return
        w, h = self.app.size
        cw = 300
        x = self.center_x(cw)
        y = h // 2 - 90
        self.card((x - 30, y - 60, cw + 60, 210))
        self.label((x, y - 48, cw, 40), "Paused", "#heading")
        for text, cb in (("Resume", self.resume), ("Quit run", self.app.home)):
            self.button((x, y, cw, 46), text, cb, "#menu_button")
            y += 58

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
            self.show_legend = not self.show_legend
            return
        super().handle_event(event)

    def advance(self):
        if self.end_hold is not None:  # run over: keep effects running under the banner
            self.app.present(self._draw)
            self.end_hold -= 1
            if self.end_hold <= 0:
                self._finish()
            return
        if self.shop_hold is not None:  # "wave cleared" for a moment, then the shop
            self.app.present(self._draw)
            self.shop_hold -= 1
            if self.shop_hold <= 0:
                self.shop_hold = None
                self.app.push(ShopScreen(self.app, self.run))
            return
        if self.run.phase != "wave":
            return
        if self.env.frame == 0 and self.frame_surface is None:
            self.app.present(self._draw)
        phase = self.run.step()  # draws and paces through env.on_frame
        if phase == "over":
            self.end_hold = int(self.END_HOLD_SECONDS * PHYSICS_FPS)
        elif phase == "shop":
            self.shop_hold = int(CLEARED_HOLD_SECONDS * PHYSICS_FPS)

    def _finish(self):
        self.app.replace(RunOverScreen(self.app, self.run.summary(), self.run.cfg))

    def _compose(self, surface, frame):
        w, h = surface.get_size()
        surface.fill(BG)
        ox, oy = (w - frame.get_width()) // 2, (h - frame.get_height()) // 2
        surface.blit(frame, (ox, oy))
        margin = ox
        if margin >= 170:
            self._draw_fleet(surface, pygame.Rect(16, 20, margin - 32, h - 40))
            self._draw_wave_panel(surface, pygame.Rect(ox + frame.get_width() + 16, 20,
                                                       margin - 32, h - 40))
        elif self.show_legend:  # no side margins: overlay the key on the play area
            r = pygame.Rect(ox + 12, oy + 50, 230, legend_height(22) + 16)
            panel = pygame.Surface(r.size, pygame.SRCALPHA)
            panel.fill((8, 10, 20, 170))
            surface.blit(panel, r.topleft)
            draw_legend(surface, (r.x + 8, r.y + 8, r.width - 16), self.font, row=22, hints=False)

    def _draw_fleet(self, surface, rect):
        env, state = self.env, self.run.state
        col = TEAM_COLORS[BLUE]["hull"]
        surface.blit(self.font_big.render("YOUR FLEET", True, col), (rect.x, rect.y))
        y = rect.y + 34
        tier = wingman_tier(self.run.cfg, state).label
        for agent in env.team_agents[BLUE]:
            ship = env.ships[agent]
            if ship.reserve:
                continue
            look = ship_look(ship)
            draw_look(surface, look, (rect.x + 10, y + 10), -math.pi / 2, 7.0, edge_width=1)
            name = "You" if agent == PLAYER else f"Wingman {agent.split('_')[1]}"
            sub = "" if agent == PLAYER else f"{tier} AI"
            if not ship.active:
                sub = "down: back next wave"
            surface.blit(self.font.render(name, True, col if ship.active else (90, 96, 112)),
                         (rect.x + 28, y + 2))
            if sub:
                surface.blit(self.font.render(sub, True, (150, 160, 182)), (rect.x + 28, y + 20))
            draw_pips(surface, ship, rect.x + 28, y + 40, ship.active)
            y += 62
        hints = ["W / Up     thrust", "A D / arrows turn", "Space      shoot", "L  legend",
                 "Esc  pause"]
        for i, line in enumerate(reversed(hints)):
            surface.blit(self.font.render(line, True, (110, 120, 145)),
                         (rect.x, rect.bottom - 20 - i * 20))

    def _draw_wave_panel(self, surface, rect):
        state = self.run.state
        col = TEAM_COLORS[RED]["hull"]
        surface.blit(self.font_big.render(f"WAVE {state.wave}", True, col), (rect.x, rect.y))
        y = rect.y + 34
        left = self.run.enemies_left()
        for text, c in ((f"{left} enem{'y' if left == 1 else 'ies'} left", col),
                        (f"{state.credits} credits", (255, 210, 122)),
                        (f"{state.waves_cleared} cleared", (150, 160, 182))):
            surface.blit(self.font.render(text, True, c), (rect.x, y))
            y += 22
        if self.show_legend:
            draw_legend(surface, (rect.x, y + 16, rect.width), self.font, row=24)
        else:
            surface.blit(self.font.render("L  show legend", True, (110, 120, 145)), (rect.x, y + 16))


class ShopScreen(MenuScreen):
    """Between waves: spend credits on your ship and your wingmen, then Continue."""

    ROW = 52

    def __init__(self, app, run: RogueliteRun):
        super().__init__(app)
        self.run = run
        self.font = None

    def build(self):
        w, h = self.app.size
        state = self.run.state
        pad, gap = 40, 24
        legend_w = 330 if w >= 1400 else 240  # wide enough for row descriptions on big windows
        col_w = (w - 2 * pad - 2 * gap - legend_w) // 2
        top = 110
        self.label((pad, 24, w - 2 * pad, 44), f"Wave {state.wave} cleared", "#heading")
        self.credits_label = self.label((pad, 64, w - 2 * pad, 30),
                                        f"{state.credits} credits  ·  {self.run.state.waves_cleared} "
                                        f"wave{'s' if state.waves_cleared != 1 else ''} cleared",
                                        "#cost_left")
        rows = shop_rows(state, self.run.cfg)
        self.rows = {r.id: r for r in rows}
        self.buy_buttons = {}
        card_h = h - top - 120
        for c, (column, title) in enumerate((("player", "Your Ship"), ("wingmen", "Wingmen"))):
            x = pad + c * (col_w + gap)
            self.card((x, top, col_w, card_h))
            self.label((x + 20, top + 14, col_w - 40, 30), title, "#heading")
            y = top + 58
            name_w = col_w - 40 - 110 - 80 - 84
            for r in (r for r in rows if r.column == column):
                self.label((x + 20, y, name_w, 24), r.name)
                self.label((x + 20, y + 22, name_w + 110, 20), r.blurb, "#hint")
                self.label((x + 20 + name_w, y, 110, 24), r.level_text, "#stat")
                self.label((x + 20 + name_w + 110, y, 80, 24),
                           "Maxed" if r.maxed else f"{r.cost} cr", "#cost")
                b = self.button((x + col_w - 20 - 76, y, 76, 36), "Buy",
                                lambda i=r.id: self.purchase(i))
                if not r.enabled:
                    b.disable()
                self.buy_buttons[r.id] = b
                y += self.ROW
            if column == "wingmen" and state.wingmen == 0:
                self.label((x + 20, y + 6, col_w - 40, 22), "Buy a wingman to unlock their upgrades.",
                           "#hint")
        self.legend_rect = pygame.Rect(w - pad - legend_w, top, legend_w, card_h)
        self.card(self.legend_rect)
        self.button((w // 2 - 110, h - 90, 220, 50), "Continue", self.continue_run,
                    "#primary_button")
        self.label((w // 2 + 130, h - 80, 220, 30), "Enter  continue", "#hint")
        self.button((pad, h - 90, 160, 50), "Quit run", self.app.home)

    def purchase(self, item_id: str):
        if not self.rows[item_id].enabled:
            return
        self.run.buy(item_id)
        self.rebuild()  # credits, levels, costs and every Buy button update together

    def continue_run(self):
        self.run.next_wave()
        self.app.pop()

    def back(self):
        pass  # Esc must not skip the shop without starting the next wave

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.continue_run()
            return
        super().handle_event(event)

    def draw(self, surface):
        super().draw(surface)
        if self.font is None:
            self.font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 15)
        r = self.legend_rect
        draw_legend(surface, (r.x + 18, r.y + 16, r.width - 36), self.font, row=28)


class RunOverScreen(MenuScreen):
    def __init__(self, app, summary: dict, cfg):
        super().__init__(app)
        self.summary = summary
        self.cfg = cfg

    def build(self):
        w, h = self.app.size
        cw = 520
        x = self.center_x(cw)
        y = int(h * 0.18)
        s = self.summary
        rows = [("Wave reached", s.get("wave", 1)), ("Waves cleared", s.get("waves_cleared", 0)),
                ("Kills by you", s.get("player_kills", 0)),
                ("Kills by the fleet", s.get("fleet_kills", 0)),
                ("Wingmen", s.get("wingmen", 0)),
                ("Credits earned", s.get("credits_earned", 0))]
        self.card((x - 30, y - 20, cw + 60, 140 + 32 * len(rows) + 90))
        self.label((x, y - 6, cw, 80), "Run over", "#title")
        self.label((x, y + 70, cw, 28), f"Destroyed during wave {s.get('wave', 1)}", "#subtitle")
        ty = y + 120
        for name, value in rows:
            self.label((x + 80, ty, 240, 26), name, "#stat_name")
            self.label((x + 320, ty, 120, 26), str(value), "#stat")
            ty += 32
        bw = (cw - 10) // 2
        self.button((x, ty + 24, bw, 46), "New run", self.new_run, "#primary_button")
        self.button((x + bw + 10, ty + 24, bw, 46), "Main menu", self.app.home)

    def new_run(self):
        self.app.replace(RogueliteGameScreen(self.app, RogueliteRun(self.cfg)))

    def back(self):
        self.app.home()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.new_run()
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
        for label, key in (("Visual effects", "effects"), ("Engine particles", "thruster_particles"),
                           ("Minimap", "minimap"),
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
