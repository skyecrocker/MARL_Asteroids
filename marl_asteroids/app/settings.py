"""Persisted app preferences and the last match setup.

Stored as JSON in ``$MARL_ASTEROIDS_CONFIG`` if set, otherwise
``$XDG_CONFIG_HOME/marl-asteroids/app.json`` (``~/.config/...`` by default). Loading never fails:
a missing or corrupt file just gives defaults, and unknown keys are ignored so old files keep
working after upgrades.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

WINDOW_SIZES: List[tuple] = [(1280, 800), (1440, 900), (1600, 1000), (1920, 1080)]


def config_path() -> Path:
    override = os.environ.get("MARL_ASTEROIDS_CONFIG")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "marl-asteroids" / "app.json"


@dataclass
class AppSettings:
    window: List[int] = field(default_factory=lambda: list(WINDOW_SIZES[0]))
    effects: bool = True          # particles, trails, flames, screen shake
    thruster_particles: bool = True  # engine exhaust as particles (off: the classic flame)
    minimap: bool = True
    show_obs: bool = False        # observation overlay on by default (Tab toggles in game)
    attract_mode: bool = True     # live AI match playing behind the main menu
    last_setup: Optional[dict] = None  # MatchSpec.to_dict() of the last match started

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppSettings":
        path = path or config_path()
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        names = {f.name for f in dataclasses.fields(cls)}
        settings = cls(**{k: v for k, v in data.items() if k in names})
        if (not isinstance(settings.window, (list, tuple)) or len(settings.window) != 2
                or not all(isinstance(v, int) and 320 <= v <= 10000 for v in settings.window)):
            settings.window = list(WINDOW_SIZES[0])
        return settings

    def save(self, path: Optional[Path] = None):
        """Atomic write, so a crash mid-save cannot corrupt the file."""
        path = Path(path or config_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=1))
        os.replace(tmp, path)
