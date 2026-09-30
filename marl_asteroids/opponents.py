"""Opponent pool for self-play / league training.

Entries are scripted bots, frozen snapshots of the learner, and the live learner itself ("self").
Sampling uses prioritized fictitious self-play (PFSP): opponents the learner struggles against are
sampled more often. Win rates are tracked with an exponential moving average per entry.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from .policies import NetworkPolicy, Policy, make_scripted


@dataclass
class PoolEntry:
    id: str
    kind: str  # "scripted" | "snapshot" | "self"
    policy: Policy
    score_ema: float = 0.5  # learner's score vs this entry (win=1, draw=0.5, loss=0)
    games: int = 0


class OpponentPool:
    def __init__(self, scripted: List[str], p_self: float = 0.0, p_scripted: float = 1.0,
                 pfsp_power: float = 2.0, ema: float = 0.03, max_snapshots: int = 30,
                 policy_factory: Optional[Callable[[], "th.nn.Module"]] = None, seed: Optional[int] = None):
        """
        Args:
            scripted: scripted opponent names (see policies.make_scripted).
            p_self: probability to play the current learner (mirror self-play).
            p_scripted: probability to play a scripted bot. The remainder goes to snapshots
                (falls back to scripted until a snapshot exists).
            pfsp_power: weight = (1 - score)^power; higher focuses harder opponents.
            policy_factory: builds an empty SB3 policy to load snapshot state_dicts into
                (needed only when snapshots are added as state_dicts).
        """
        assert p_self + p_scripted <= 1.0 + 1e-9
        self.rng = np.random.default_rng(seed)
        self.p_self, self.p_scripted = p_self, p_scripted
        self.pfsp_power, self.ema = pfsp_power, ema
        self.max_snapshots = max_snapshots
        self.policy_factory = policy_factory
        self.entries: Dict[str, PoolEntry] = {}
        for k, name in enumerate(scripted):
            self.entries[name] = PoolEntry(name, "scripted", make_scripted(name, seed=None if seed is None else seed + k))
        self._self_entry: Optional[PoolEntry] = None

    # -------------------------------------------------------------- membership

    def set_learner(self, sb3_policy):
        """Register the live learner policy as the 'self' opponent (stochastic)."""
        self._self_entry = PoolEntry("self", "self", NetworkPolicy(sb3_policy, name="self", deterministic=False))
        self.entries["self"] = self._self_entry

    def update_learner_weights(self, state_dict):
        """For worker processes that hold a copy of the learner rather than a live reference."""
        if self._self_entry is None:
            self.set_learner(self._build(state_dict))
        else:
            self._self_entry.policy.policy.load_state_dict(state_dict)

    def add_snapshot(self, name: str, policy_or_state_dict):
        if isinstance(policy_or_state_dict, dict):
            pol = self._build(policy_or_state_dict)
        else:
            pol = copy.deepcopy(policy_or_state_dict)
            pol.optimizer = None
            pol.set_training_mode(False)
        # New snapshots start at the score of the most recent one (they're about as strong)
        prev = [e for e in self.entries.values() if e.kind == "snapshot"]
        init = prev[-1].score_ema if prev else 0.5
        self.entries[name] = PoolEntry(name, "snapshot", NetworkPolicy(pol, name=name, deterministic=False),
                                       score_ema=init)
        snaps = [e for e in self.entries.values() if e.kind == "snapshot"]
        while len(snaps) > self.max_snapshots:
            # Drop the snapshot the learner beats most easily (keeps the pool challenging and diverse)
            easiest = max(snaps[:-1], key=lambda e: e.score_ema)
            del self.entries[easiest.id]
            snaps.remove(easiest)

    def _build(self, state_dict):
        assert self.policy_factory is not None, "policy_factory required to load state_dicts"
        pol = self.policy_factory()
        pol.load_state_dict(state_dict)
        pol.set_training_mode(False)
        return pol

    # ---------------------------------------------------------------- sampling

    def _pfsp(self, entries: List[PoolEntry]) -> PoolEntry:
        w = np.array([(1.0 - e.score_ema) ** self.pfsp_power + 0.02 for e in entries])
        return entries[self.rng.choice(len(entries), p=w / w.sum())]

    def sample(self) -> PoolEntry:
        r = self.rng.random()
        scripted = [e for e in self.entries.values() if e.kind == "scripted"]
        snaps = [e for e in self.entries.values() if e.kind == "snapshot"]
        if r < self.p_self and self._self_entry is not None:
            return self._self_entry
        if r < self.p_self + self.p_scripted or not snaps:
            return self._pfsp(scripted) if scripted else (self._self_entry or self._pfsp(snaps))
        return self._pfsp(snaps)

    def report(self, entry_id: str, score: float):
        e = self.entries.get(entry_id)
        if e is None:  # evicted while a game was running
            return
        e.games += 1
        e.score_ema += self.ema * (score - e.score_ema)

    def stats(self) -> Dict[str, Tuple[float, int]]:
        return {e.id: (e.score_ema, e.games) for e in self.entries.values()}
