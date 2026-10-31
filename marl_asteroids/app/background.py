"""Attract mode: a bot-vs-bot station siege running behind the menus.

Scripted bots only, so it costs almost nothing and never depends on checkpoints being present.
The map is drawn large enough to cover the window and dimmed so widgets stay readable.
"""
from __future__ import annotations

import pygame

from ..config import make_game
from ..env import TeamAsteroidsEnv
from ..policies import make_scripted
from ..render import Renderer

DIM_ALPHA = 150


class MenuBackground:
    def __init__(self, size, effects: bool = True, seed: int = 0):
        self.size = size
        # A 3v3 station siege between bots: shows off the objective mode and restarts at the end
        cfg = make_game("siege", "medium", 3, 3, time_limit=180.0, use_mode_time_limit=False)
        self.env = TeamAsteroidsEnv(cfg)
        self.renderer = Renderer(self.env, window=max(size), effects=effects, show_hud=False)
        self.blue = make_scripted("heuristic_1.0", seed=seed)
        self.red = make_scripted("heuristic_0.75", seed=seed + 1)
        self.seed = seed
        self.games = 0
        self.tick = 0
        self.env.reset(seed=seed)
        self.dim = pygame.Surface(size, pygame.SRCALPHA)
        self.dim.fill((4, 6, 16, DIM_ALPHA))

    def _step(self):
        env = self.env
        if env.game_over or not env.agents:
            self.games += 1
            env.reset(seed=self.seed + self.games)
            return
        actions = {}
        for team, pol in (("blue", self.blue), ("red", self.red)):
            ags = [a for a in env.agents if a.startswith(team)]
            if ags:
                acts = pol.act(None, [(env, a) for a in ags])
                actions.update(zip(ags, acts))
        env.step(actions)

    def draw(self, surface: pygame.Surface):
        # One decision covers action_repeat physics frames; step every other UI frame to stay at
        # real-time speed.
        self.tick += 1
        if self.tick % self.env.config.action_repeat == 0:
            self._step()
        frame = self.renderer.render("surface")
        w, h = surface.get_size()
        surface.blit(frame, ((w - frame.get_width()) // 2, (h - frame.get_height()) // 2))
        surface.blit(self.dim, (0, 0))
