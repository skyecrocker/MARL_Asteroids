"""Desktop app: one window, a stack of screens, pygame_gui widgets.

Run with ``uv run marl-asteroids`` (or ``uv run python scripts/app.py``).

Loop: each iteration drains events, then either lets the active screen drive its own frames (the
game screen draws and paces once per physics frame through ``env.on_frame``), or does a normal
UI frame: update, draw the screen's background, draw widgets, flip.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, List, Optional

import pygame
import pygame_gui

from .settings import AppSettings

THEME = Path(__file__).with_name("theme.json")
FPS = 60
BG = (6, 8, 20)


class Screen:
    """Base screen. Widgets go in ``self.root`` so ``teardown`` removes them all at once."""

    def __init__(self, app: "App"):
        self.app = app
        self.root: Optional[pygame_gui.elements.UIPanel] = None

    # lifecycle -------------------------------------------------------------
    def enter(self):
        w, h = self.app.size
        self.root = pygame_gui.elements.UIPanel(pygame.Rect(0, 0, w, h), manager=self.app.manager,
                                                object_id="#screen")
        self.build()

    def build(self):
        """Create widgets (parented to ``self.root``)."""

    def teardown(self):
        if self.root is not None:
            self.root.kill()
            self.root = None

    def rebuild(self):
        self.teardown()
        self.enter()

    def exit(self):
        """Called once when the screen leaves the stack for good."""
        self.teardown()

    # frame ---------------------------------------------------------------------
    def handle_event(self, event):
        pass

    def update(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        """Background drawn under the widgets."""
        self.app.draw_menu_background(surface)

    def drives_frames(self) -> bool:
        return False

    def advance(self):
        pass


class App:
    def __init__(self, settings_path: Optional[Path] = None):
        self.settings_path = settings_path
        self.settings = AppSettings.load(settings_path)
        pygame.init()
        pygame.display.set_caption("MARL Asteroids")
        self.size = tuple(self.settings.window)
        self.window = pygame.display.set_mode(self.size)
        self.manager = pygame_gui.UIManager(self.size, theme_path=str(THEME))
        # The theme preloads its own fonts; bold 14 is only used by <b> in small text boxes.
        self.manager.preload_fonts([{"name": "noto_sans", "point_size": 14, "style": "bold"}])
        self.clock = pygame.time.Clock()
        self.stack: List[Screen] = []
        self.running = True
        self.frames = 0
        self._background = None  # lazily created attract-mode match

    # screens -----------------------------------------------------------------
    @property
    def screen(self) -> Screen:
        return self.stack[-1]

    def push(self, screen: Screen):
        if self.stack:
            self.stack[-1].teardown()
        self.stack.append(screen)
        screen.enter()

    def pop(self):
        self.stack.pop().exit()
        if self.stack:
            self.stack[-1].enter()

    def replace(self, screen: Screen):
        if self.stack:
            self.stack.pop().exit()
        self.stack.append(screen)
        screen.enter()

    def home(self):
        while len(self.stack) > 1:
            self.stack.pop().exit()
        self.stack[-1].enter()

    # settings ------------------------------------------------------------------
    def save_settings(self):
        try:
            self.settings.save(self.settings_path)
        except OSError as e:  # settings are a convenience; never crash over them
            print(f"Could not save settings: {e}")

    def set_window_size(self, size):
        size = (int(size[0]), int(size[1]))
        if size == self.size:
            return
        self.size = size
        self.settings.window = list(size)
        self.window = pygame.display.set_mode(size)
        self.manager.set_window_resolution(size)
        self._background = None  # re-sized on next draw
        self.screen.rebuild()
        self.save_settings()

    # drawing -------------------------------------------------------------------
    def draw_menu_background(self, surface: pygame.Surface):
        if self.settings.attract_mode:
            if self._background is None:
                from .background import MenuBackground
                self._background = MenuBackground(self.size, effects=self.settings.effects)
            self._background.draw(surface)
        else:
            surface.fill(BG)

    def present(self, draw: Callable[[pygame.Surface], None]):
        """One paced frame drawn by ``draw`` plus widgets. Used by screens that drive frames."""
        dt = self.clock.tick(FPS) / 1000.0
        self.manager.update(dt)
        draw(self.window)
        self.manager.draw_ui(self.window)
        pygame.display.flip()
        self.frames += 1

    # loop ------------------------------------------------------------------------
    def quit(self):
        self.running = False

    def step(self):
        """One loop iteration (exposed for tests)."""
        screen = self.screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
                return
            self.manager.process_events(event)
            screen.handle_event(event)
            if self.screen is not screen or not self.running:
                return  # a handler switched screens; drop the rest of this batch
        if screen.drives_frames():
            screen.advance()
        else:
            dt = self.clock.tick(FPS) / 1000.0
            self.manager.update(dt)
            screen.update(dt)
            if self.screen is screen:
                screen.draw(self.window)
                self.manager.draw_ui(self.window)
                pygame.display.flip()
                self.frames += 1

    def run(self, start: Optional[Screen] = None):
        from .screens import MainMenuScreen

        self.push(start or MainMenuScreen(self))
        try:
            while self.running:
                self.step()
        finally:
            while self.stack:
                self.stack.pop().exit()
            self.save_settings()
            pygame.quit()


def main(argv=None):
    ap = argparse.ArgumentParser(description="MARL Asteroids desktop app")
    ap.add_argument("--settings", default=None, help="Settings file (default: ~/.config/marl-asteroids)")
    args = ap.parse_args(argv)
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Note: SDL_VIDEODRIVER=dummy is set, so no window will appear.")
    App(Path(args.settings) if args.settings else None).run()


if __name__ == "__main__":
    main()
