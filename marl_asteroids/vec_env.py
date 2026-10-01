"""SB3 VecEnv that runs many team games in-process with parameter sharing across teammates.

Each learner-controlled ship is one VecEnv slot. Games of different sizes (1v1, 2v2, 3v3, ...)
can be mixed. The opposing team is driven by an opponent sampled from an OpponentPool at every
game reset, with batched inference per opponent policy.

Dead-agent handling ("death masking"): a slot whose ship died keeps returning an all-zero
observation (alive flag = 0) and keeps receiving its share of team rewards (team spirit and the
final win/loss), so agents learn that the team outcome matters even after they die. All slots of a
game are marked done together when the game ends.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .config import GameConfig
from .env import BLUE, RED, TeamAsteroidsEnv
from .opponents import OpponentPool, PoolEntry


class TeamVecEnv(VecEnv):
    def __init__(self, configs: Sequence[GameConfig], pool: OpponentPool, learner_team: int = BLUE,
                 seed: Optional[int] = None):
        self.games = [TeamAsteroidsEnv(cfg) for cfg in configs]
        self.pool = pool
        self.learner_team = learner_team
        self.opp_team = RED if learner_team == BLUE else BLUE
        self.slots = [(g, a) for g, env in enumerate(self.games) for a in env.team_agents[learner_team]]
        assert self.slots, "no learner-controlled ships"
        self.game_slots: List[List[int]] = [[] for _ in self.games]
        for k, (g, _) in enumerate(self.slots):
            self.game_slots[g].append(k)
        self.opponents: List[Optional[PoolEntry]] = [None] * len(self.games)
        self._seed = seed
        self._resets = 0
        self._actions = None
        # Rolling training outcome counters (read by callbacks)
        self.outcomes = {"win": 0, "loss": 0, "draw": 0}

        env0 = self.games[0]
        self.render_mode = None
        super().__init__(len(self.slots), env0.observation_space(env0.possible_agents[0]),
                         env0.action_space(env0.possible_agents[0]))

    # ------------------------------------------------------------------ core

    def _reset_game(self, g: int):
        seed = None
        if self._seed is not None:
            seed = self._seed + self._resets
        self._resets += 1
        self.games[g].reset(seed=seed)
        self.opponents[g] = self.pool.sample()

    def _slot_obs(self) -> np.ndarray:
        return np.stack([self.games[g].get_obs(a) for g, a in self.slots])

    def reset(self):
        for g in range(len(self.games)):
            self._reset_game(g)
        return self._slot_obs()

    def step_async(self, actions: np.ndarray):
        self._actions = actions

    def step_wait(self):
        actions = np.asarray(self._actions).reshape(self.num_envs, -1)
        per_game: List[Dict[str, np.ndarray]] = [{} for _ in self.games]
        for k, (g, a) in enumerate(self.slots):
            if self.games[g].ships[a].active:
                per_game[g][a] = actions[k]

        # Opponent actions, batched by policy
        groups: Dict[int, tuple] = {}
        for g, env in enumerate(self.games):
            entry = self.opponents[g]
            for a in env.team_agents[self.opp_team]:
                if env.ships[a].active:
                    groups.setdefault(id(entry.policy), (entry.policy, []))[1].append((g, a))
        for policy, members in groups.values():
            obs = np.stack([self.games[g].get_obs(a) for g, a in members])
            acts = policy.act(obs, [(self.games[g], a) for g, a in members])
            for (g, a), act in zip(members, acts):
                per_game[g][a] = act

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        obs = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.float32)

        for g, env in enumerate(self.games):
            env.step(per_game[g])
            for k in self.game_slots[g]:
                a = self.slots[k][1]
                rewards[k] = env.rewards_all[a]
                obs[k] = env.get_obs(a)
            if env.game_over:
                if env.winner == self.learner_team:
                    score, key = 1.0, "win"
                elif env.winner == self.opp_team:
                    score, key = 0.0, "loss"
                else:
                    score, key = 0.5, "draw"
                self.outcomes[key] += 1
                self.pool.report(self.opponents[g].id, score)
                truncated = env.winner is not None and env._out_of_time()
                for k in self.game_slots[g]:
                    dones[k] = True
                    infos[k]["terminal_observation"] = obs[k].copy()
                    infos[k]["TimeLimit.truncated"] = truncated
                    infos[k]["outcome"] = score
                    infos[k]["opponent"] = self.opponents[g].id
                self._reset_game(g)
                for k in self.game_slots[g]:
                    obs[k] = env.get_obs(self.slots[k][1])
        return obs, rewards, dones, infos

    # ----------------------------------------------------- self-play control
    # Same interface as ParallelTeamVecEnv so callbacks don't care which one they get.

    def set_learner(self, sb3_policy):
        self.pool.set_learner(sb3_policy)  # live reference, always current

    def sync_learner(self, sb3_policy):
        pass

    def add_snapshot(self, name: str, sb3_policy):
        self.pool.add_snapshot(name, sb3_policy)

    def pool_stats(self):
        return self.pool.stats()

    def get_outcomes(self):
        return dict(self.outcomes)

    # ------------------------------------------------------- VecEnv plumbing

    def close(self):
        for env in self.games:
            env.close()

    def seed(self, seed: Optional[int] = None):
        self._seed = seed
        self._resets = 0
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs)] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def get_images(self):
        return [None] * self.num_envs


# ---------------------------------------------------------------------------
# Multiprocess version
# ---------------------------------------------------------------------------

def _state_dict_cpu(sb3_policy):
    return {k: v.detach().cpu().clone() for k, v in sb3_policy.state_dict().items()}


def _worker(remote, parent_remote, configs, pool_kwargs, seed):
    import copy

    import torch as th

    th.set_num_threads(1)
    parent_remote.close()
    pool = OpponentPool(**pool_kwargs)
    venv = TeamVecEnv(configs, pool, seed=seed)
    template = None  # CPU copy of the learner used to build snapshots
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                venv.step_async(data)
                remote.send(venv.step_wait())
            elif cmd == "reset":
                remote.send(venv.reset())
            elif cmd == "set_learner":
                template = data
                template.set_training_mode(False)
                pool.set_learner(copy.deepcopy(template))
                remote.send(None)
            elif cmd == "sync_learner":
                pool.update_learner_weights(data)
                remote.send(None)
            elif cmd == "add_snapshot":
                name, state = data
                pol = copy.deepcopy(template)
                pol.load_state_dict(state)
                pool.add_snapshot(name, pol)
                remote.send(None)
            elif cmd == "stats":
                remote.send((pool.stats(), dict(venv.outcomes)))
            elif cmd == "seed":
                remote.send(venv.seed(data))
            elif cmd == "close":
                venv.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class ParallelTeamVecEnv(VecEnv):
    """Splits games across worker processes, each with its own TeamVecEnv and OpponentPool."""

    def __init__(self, configs: Sequence[GameConfig], pool_kwargs: dict, n_workers: int = 4,
                 seed: Optional[int] = None, start_method: str = "forkserver"):
        import multiprocessing as mp

        n_workers = max(1, min(n_workers, len(configs)))
        chunks = [list(configs[i::n_workers]) for i in range(n_workers)]
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_workers)])
        self.processes = []
        self.slot_counts = []
        for w, (wr, r, chunk) in enumerate(zip(self.work_remotes, self.remotes, chunks)):
            kw = dict(pool_kwargs)
            if kw.get("seed") is not None:
                kw["seed"] = kw["seed"] + 1000 * w
            wseed = None if seed is None else seed + 1_000_000 * w
            p = ctx.Process(target=_worker, args=(wr, r, chunk, kw, wseed), daemon=True)
            p.start()
            self.processes.append(p)
            wr.close()
            self.slot_counts.append(sum(cfg.m_blue for cfg in chunk))  # learner = blue
        probe = TeamAsteroidsEnv(configs[0])
        agent = probe.possible_agents[0]
        self.render_mode = None
        super().__init__(sum(self.slot_counts), probe.observation_space(agent), probe.action_space(agent))
        self._splits = np.cumsum(self.slot_counts)[:-1]
        self.closed = False

    def _call_all(self, cmd, data=None):
        for r in self.remotes:
            r.send((cmd, data))
        return [r.recv() for r in self.remotes]

    def reset(self):
        return np.concatenate(self._call_all("reset"))

    def step_async(self, actions):
        for r, a in zip(self.remotes, np.split(np.asarray(actions), self._splits)):
            r.send(("step", a))

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return (np.concatenate(obs), np.concatenate(rews), np.concatenate(dones),
                [i for part in infos for i in part])

    # self-play control
    def set_learner(self, sb3_policy):
        import copy
        cpu_copy = copy.deepcopy(sb3_policy).to("cpu")
        cpu_copy.optimizer = None
        self._call_all("set_learner", cpu_copy)

    def sync_learner(self, sb3_policy):
        self._call_all("sync_learner", _state_dict_cpu(sb3_policy))

    def add_snapshot(self, name, sb3_policy):
        self._call_all("add_snapshot", (name, _state_dict_cpu(sb3_policy)))

    def pool_stats(self):
        """Mean score_ema and summed games per entry across workers."""
        merged: Dict[str, list] = {}
        for stats, _ in self._call_all("stats"):
            for k, (score, games) in stats.items():
                merged.setdefault(k, []).append((score, games))
        return {k: (float(np.mean([s for s, _ in v])), int(sum(g for _, g in v))) for k, v in merged.items()}

    def get_outcomes(self):
        total = {"win": 0, "loss": 0, "draw": 0}
        for _, out in self._call_all("stats"):
            for k in total:
                total[k] += out[k]
        return total

    # plumbing
    def close(self):
        if self.closed:
            return
        for r in self.remotes:
            r.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)
        self.closed = True

    def seed(self, seed=None):
        for w, r in enumerate(self.remotes):
            r.send(("seed", None if seed is None else seed + 1_000_000 * w))
        return [x for r in self.remotes for x in r.recv()]

    def get_attr(self, attr_name, indices=None):
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def get_images(self):
        return [None] * self.num_envs
