import numpy as np
import pytest

from marl_asteroids.config import get_scenario
from marl_asteroids.matches import play_matches
from marl_asteroids.opponents import OpponentPool
from marl_asteroids.policies import make_scripted
from marl_asteroids.training import build_game_configs
from marl_asteroids.vec_env import TeamVecEnv


def make_venv(mix=None, **pool_kw):
    configs = build_game_configs(mix or {"duel": 2, "team2": 2, "team3": 1})
    pool = OpponentPool(pool_kw.pop("scripted", ["heuristic_1.0"]), seed=0, **pool_kw)
    return TeamVecEnv(configs, pool, seed=0)


def test_slot_layout():
    venv = make_venv()
    assert venv.num_envs == 2 * 1 + 2 * 2 + 3
    obs = venv.reset()
    assert obs.shape == (venv.num_envs,) + venv.observation_space.shape


def test_game_slots_done_together_and_dead_slots_zero():
    venv = make_venv({"team3": 2})
    venv.reset()
    rng = np.random.default_rng(0)
    saw_dead = False
    for _ in range(2000):
        acts = rng.integers([2, 3, 2], size=(venv.num_envs, 3))
        before = [venv.games[g].ships[a].active for g, a in venv.slots]
        obs, rew, dones, infos = venv.step(acts)
        for g, slots in enumerate(venv.game_slots):
            d = dones[slots]
            assert d.all() or not d.any()
            if not d.any():
                for k in slots:
                    if not venv.games[g].ships[venv.slots[k][1]].active:
                        assert np.all(obs[k] == 0)
                        saw_dead = True
            else:
                for k in slots:
                    assert "terminal_observation" in infos[k] and "outcome" in infos[k]
    assert saw_dead


def test_outcomes_reported_to_pool():
    venv = make_venv({"duel": 4}, scripted=["dummy"])
    venv.reset()
    for _ in range(3000):
        venv.step(np.tile([0, 0, 1], (venv.num_envs, 1)))
    s = venv.pool.stats()["dummy"]
    assert s[1] > 0


def test_pool_pfsp_prefers_hard_opponents():
    pool = OpponentPool(["dummy", "heuristic_1.0"], seed=0, pfsp_power=2.0)
    pool.entries["dummy"].score_ema = 0.99
    pool.entries["heuristic_1.0"].score_ema = 0.2
    counts = {"dummy": 0, "heuristic_1.0": 0}
    for _ in range(2000):
        counts[pool.sample().id] += 1
    assert counts["heuristic_1.0"] > 10 * counts["dummy"]


def test_play_matches_counts():
    stats = play_matches(get_scenario("duel"), make_scripted("heuristic_1.0", 0), make_scripted("dummy"), 20,
                         n_parallel=8, seed=0)
    assert stats.games == 20
    assert stats.blue_win_rate > 0.9


def test_play_matches_seeded():
    a = play_matches(get_scenario("team2"), make_scripted("heuristic_1.0", 0), make_scripted("heuristic_1.0", 1), 12, seed=5)
    b = play_matches(get_scenario("team2"), make_scripted("heuristic_1.0", 0), make_scripted("heuristic_1.0", 1), 12, seed=5)
    assert a.lengths == b.lengths
