"""Training helpers: env/pool construction and callbacks for self-play and win-rate evaluation."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .config import GameConfig, get_scenario
from .matches import play_matches, play_matches_mirrored
from .policies import NetworkPolicy, make_scripted


def build_game_configs(mix: Dict[str, int], **overrides) -> List[GameConfig]:
    """{"duel": 8, "team2": 8} -> list of configs, one per parallel game."""
    configs = []
    for name, count in mix.items():
        configs.extend(get_scenario(name, **overrides) for _ in range(count))
    return configs


class SelfPlayCallback(BaseCallback):
    """Adds frozen learner snapshots to the pool and logs pool / training outcome stats."""

    def __init__(self, snapshot_every: int, snapshot_dir: Optional[str] = None,
                 log_every: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.snapshot_every = snapshot_every
        self.snapshot_dir = snapshot_dir
        self.log_every = log_every
        self._last_snap = 0
        self._last_log = 0
        self._last_outcomes = None

    def _venv(self):
        """Unwrap VecMonitor etc. down to the Team(Parallel)VecEnv."""
        venv = self.training_env
        while not hasattr(venv, "add_snapshot") and hasattr(venv, "venv"):
            venv = venv.venv
        return venv

    def _on_training_start(self):
        self._venv().set_learner(self.model.policy)
        if self.snapshot_dir:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def _on_rollout_start(self):
        self._venv().sync_learner(self.model.policy)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        venv = self._venv()
        if self.snapshot_every and t - self._last_snap >= self.snapshot_every:
            self._last_snap = t
            name = f"snap_{t // 1000}k"
            venv.add_snapshot(name, self.model.policy)
            if self.snapshot_dir:
                self.model.save(os.path.join(self.snapshot_dir, name))
        if t - self._last_log >= self.log_every:
            self._last_log = t
            out = venv.get_outcomes()
            if self._last_outcomes is not None:
                d = {k: out[k] - self._last_outcomes[k] for k in out}
                n = max(1, sum(d.values()))
                self.logger.record("selfplay/train_win_rate", d["win"] / n)
                self.logger.record("selfplay/train_draw_rate", d["draw"] / n)
            self._last_outcomes = dict(out)
            stats = venv.pool_stats()
            snaps = [s for k, s in stats.items() if k.startswith("snap_")]
            self.logger.record("selfplay/num_snapshots", len(snaps))
            if snaps:
                self.logger.record("selfplay/mean_score_vs_snapshots", float(np.mean([s[0] for s in snaps])))
            for k, (score, games) in stats.items():
                if not k.startswith("snap_"):
                    self.logger.record(f"pool/{k}", score)
        return True


class WinRateEvalCallback(BaseCallback):
    """Periodically plays the (deterministic) learner vs. fixed opponents; saves best by mean win rate."""

    def __init__(self, suite: Sequence[Tuple[str, str]], eval_every: int, n_games: int = 100,
                 save_dir: Optional[str] = None, seed: int = 12345, verbose: int = 1):
        """suite: [(scenario_name, opponent_name), ...]"""
        super().__init__(verbose)
        self.suite = list(suite)
        self.eval_every = eval_every
        self.n_games = n_games
        self.save_dir = save_dir
        self.seed = seed
        self.best = -np.inf
        self._last = 0
        self.history: List[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.eval_every:
            return True
        self._last = self.num_timesteps
        self.evaluate()
        return True

    def _on_training_end(self):
        self.evaluate()

    def evaluate(self):
        learner = NetworkPolicy(self.model.policy, name="learner", deterministic=True)
        rates, row = [], {"timesteps": self.num_timesteps}
        for scenario, opp in self.suite:
            cfg = get_scenario(scenario)
            runner = play_matches_mirrored if cfg.n_red == cfg.m_blue else play_matches
            stats = runner(cfg, learner, make_scripted(opp, seed=self.seed), self.n_games,
                           n_parallel=min(32, self.n_games), seed=self.seed)
            s = stats.summary()
            key = f"{scenario}_vs_{opp}"
            self.logger.record(f"eval/{key}/win_rate", s["blue_win_rate"])
            self.logger.record(f"eval/{key}/teamkills", s["blue_teamkills_per_game"])
            self.logger.record(f"eval/{key}/env_deaths", s["blue_env_deaths_per_game"])
            rates.append(s["blue_win_rate"])
            row[key] = s["blue_win_rate"]
        mean = float(np.mean(rates))
        row["mean"] = mean
        self.history.append(row)
        self.logger.record("eval/mean_win_rate", mean)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] mean win rate {mean:.3f} | "
                  + ", ".join(f"{k}={v:.2f}" for k, v in row.items() if k not in ("timesteps", "mean")))
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "eval_history.json"), "w") as f:
                json.dump(self.history, f, indent=1)
            if mean > self.best:
                self.best = mean
                self.model.save(os.path.join(self.save_dir, "best_model"))
