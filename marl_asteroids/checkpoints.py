"""Find trained checkpoints that can actually play the current environment.

Each SB3 ``.zip`` stores its observation shape as plain JSON, so compatibility is checked without
loading PyTorch. Checkpoints built for an older observation layout (e.g. ``models/_stale_obs191``)
are reported as incompatible instead of crashing when picked.

A folder may contain ``manifest.json`` with display names and ratings::

    {"files": {"selfplay_final.zip": {"name": "Self-play (final)", "elo": 1718,
                                      "description": "20M steps of league self-play"}}}
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .config import GameConfig
from .env import obs_dim

MANIFEST = "manifest.json"


@dataclass(frozen=True)
class Checkpoint:
    path: str                  # relative to the working directory when possible
    name: str                  # display name
    obs_dim: Optional[int]     # None if the file could not be read
    elo: Optional[float] = None
    description: str = ""

    @property
    def compatible(self) -> bool:
        return self.obs_dim == current_obs_dim()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.elo:.0f})" if self.elo is not None else self.name


def current_obs_dim() -> int:
    return obs_dim(GameConfig())


def models_root() -> Path:
    """``./models`` if it exists (normal use from the repo root), else the repo's models folder."""
    cwd = Path.cwd() / "models"
    if cwd.is_dir():
        return cwd
    return Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=1024)
def _read_obs_dim(path: str, mtime: float) -> Optional[int]:
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data"))
        shape = data["observation_space"]["_shape"]
        return int(shape[0]) if len(shape) == 1 else None
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, TypeError, IndexError):
        return None


def read_obs_dim(path: str) -> Optional[int]:
    try:
        return _read_obs_dim(os.path.abspath(path), os.path.getmtime(path))
    except OSError:
        return None


def _load_manifest(folder: Path) -> dict:
    try:
        return json.loads((folder / MANIFEST).read_text()).get("files", {})
    except (OSError, ValueError, AttributeError):
        return {}


def _display_path(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def discover(root: Optional[Path] = None, include_snapshots: bool = False,
             compatible_only: bool = True) -> List[Checkpoint]:
    """All checkpoints under ``root``, best-rated first.

    Skips ``snapshots/`` folders (self-play keeps dozens) unless asked, and folders starting with
    ``_`` or named ``legacy`` (archived, incompatible by construction).
    """
    root = Path(root) if root is not None else models_root()
    if not root.is_dir():
        return []
    found: List[Checkpoint] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith("_") and d != "legacy"
                             and (include_snapshots or d != "snapshots"))
        folder = Path(dirpath)
        manifest = _load_manifest(folder)
        for fname in sorted(filenames):
            if not fname.endswith(".zip"):
                continue
            path = folder / fname
            meta = manifest.get(fname, {})
            rel = folder.relative_to(root)
            default_name = f"{rel}/{fname[:-4]}" if str(rel) != "." else fname[:-4]
            ckpt = Checkpoint(path=_display_path(path), name=meta.get("name", default_name),
                              obs_dim=read_obs_dim(str(path)), elo=meta.get("elo"),
                              description=meta.get("description", ""))
            if compatible_only and not ckpt.compatible:
                continue
            found.append(ckpt)
    # Rated first (highest Elo first), then unrated alphabetically
    found.sort(key=lambda c: (c.elo is None, -(c.elo or 0.0), c.name))
    return found
