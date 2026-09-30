"""Tests for map-size independence, mirrored evaluation, static symmetry, asteroids, handicaps."""
import math

import numpy as np
import pytest

from marl_asteroids.config import (EntityState, GameConfig, SCENARIOS, get_scenario,
                                   scale_map, set_team_fire_rate, static_battle)
from marl_asteroids.env import BLUE, RED, TeamAsteroidsEnv
from marl_asteroids.matches import play_matches, play_matches_mirrored
from marl_asteroids.policies import DelayedPolicy, Policy, make_scripted


def _setup(env, placements):
    env.reset(seed=0)
    for agent, (x, y, angle) in placements.items():
        s = env.ships[agent]
        s.pos[:] = (x, y)
        s.angle = angle
        s.vel[:] = 0.0
    env._compute_obs()


# --------------------------------------------------------- map-size independence


def test_obs_is_independent_of_map_size():
    """Identical local geometry must give an identical observation on any map size."""
    obs = {}
    for w in (800.0, 1600.0, 4000.0):
        env = TeamAsteroidsEnv(get_scenario("duel", width=w, height=w))
        _setup(env, {"blue_0": (300, 300, 0.3), "red_0": (420, 380, 1.0)})
        env.ships["red_0"].vel[:] = (1.0, -2.0)
        env._compute_obs()
        obs[w] = env.get_obs("blue_0")
    ref = obs[800.0]
    for w, o in obs.items():
        np.testing.assert_allclose(o, ref, atol=1e-7, err_msg=f"map {w} differs")


def test_sensor_range_sets_the_distance_scale():
    env = TeamAsteroidsEnv(get_scenario("duel", sensor_range=400.0))
    _setup(env, {"blue_0": (100, 100, 0.0), "red_0": (300, 100, 0.0)})
    o = env.get_obs("blue_0")
    from marl_asteroids.env import EGO_DIM
    bearing_x, rng = o[EGO_DIM + 1], o[EGO_DIM + 3]
    assert bearing_x == pytest.approx(1.0)          # unit bearing, straight ahead
    assert rng == pytest.approx(200 / (200 + 400))  # saturating range, 0.5 at sensor_range


# ------------------------------------------------------------- mirrored evaluation


def test_mirrored_eval_is_unbiased():
    """A policy against itself must score exactly 0.5; plain evaluation need not."""
    p = lambda: make_scripted("heuristic_1.0", seed=11)
    stats = play_matches_mirrored(get_scenario("duel"), p(), p(), 120, seed=4)
    assert stats.blue_score == pytest.approx(0.5)
    assert stats.games == 120


def test_mirrored_eval_swaps_per_team_counters():
    stats = play_matches_mirrored(get_scenario("team2"), make_scripted("heuristic_1.0", 0),
                                 make_scripted("dummy"), 40, seed=1)
    assert stats.games == 40
    assert stats.blue_win_rate > 0.7           # heuristic should dominate dummies either color
    assert stats.kills["blue"] > stats.kills["red"]


def test_mirrored_eval_requires_symmetric_teams():
    with pytest.raises(ValueError):
        play_matches_mirrored(get_scenario("team2_vs_1"), make_scripted("dummy"),
                             make_scripted("dummy"), 2)


# ------------------------------------------------------------- static symmetry


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n.startswith("demo_")])
def test_static_scenarios_are_exactly_symmetric(name):
    """180-degree rotation maps red onto blue, so identical deterministic policies must draw."""
    cfg = get_scenario(name)
    stats = play_matches(cfg, make_scripted("heuristic_1.0"), make_scripted("heuristic_1.0"),
                         3, n_parallel=3, seed=0)
    assert stats.blue_wins == 0 and stats.red_wins == 0, "one side has an advantage"


def test_static_spawns_are_exact_and_repeatable():
    cfg = get_scenario("demo_team2")
    a, b = TeamAsteroidsEnv(cfg), TeamAsteroidsEnv(cfg)
    a.reset(seed=1)
    b.reset(seed=999)  # different seed must not move static spawns
    for agent in cfg_agents(cfg):
        np.testing.assert_array_equal(a.ships[agent].pos, b.ships[agent].pos)
        assert a.ships[agent].angle == b.ships[agent].angle
    # red mirrors blue under a 180-degree rotation about the map center
    w, h = cfg.width, cfg.height
    reds = sorted(tuple(np.round(a.ships[f"red_{i}"].pos, 6)) for i in range(cfg.n_red))
    blues = sorted(tuple(np.round(((w - a.ships[f"blue_{i}"].pos[0]) % w,
                                  (h - a.ships[f"blue_{i}"].pos[1]) % h), 6))
                   for i in range(cfg.m_blue))
    assert reds == blues


def cfg_agents(cfg):
    return [f"red_{i}" for i in range(cfg.n_red)] + [f"blue_{i}" for i in range(cfg.m_blue)]


def test_static_battle_rejects_antipodal_inset():
    """inset=0.25 places teams exactly half a map apart, where shortest-path direction ties."""
    cfg = static_battle(1, 1200, 1200, inset=0.25)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    dx, _, dist = env.wrapped_delta(env.ships["red_0"].pos, env.ships["blue_0"].pos)
    assert dist == pytest.approx(cfg.width / 2)  # documents the degenerate case we avoid
    cfg_ok = static_battle(1, 1200, 1200)
    env_ok = TeamAsteroidsEnv(cfg_ok)
    env_ok.reset(seed=0)
    _, _, d_ok = env_ok.wrapped_delta(env_ok.ships["red_0"].pos, env_ok.ships["blue_0"].pos)
    assert d_ok < cfg_ok.width / 2


# ------------------------------------------------------------------- asteroids


def _shoot_rock(size, split):
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=1, asteroids_split=split,
                     asteroid_spawns=[EntityState(x=500.0, y=400.0, size=size)])
    env = TeamAsteroidsEnv(cfg)
    _setup(env, {"red_0": (300, 400, 0.0), "blue_0": (100, 100, 0.0)})
    env.step({"red_0": [0, 0, 1], "blue_0": [0, 0, 0]})
    for _ in range(40):
        env.step({a: [0, 0, 0] for a in env.agents})
    return env


def test_large_asteroid_splits_into_two_smaller_faster():
    env = _shoot_rock(3, split=True)
    rocks = [r for r in env.asteroids if r.active]
    assert len(rocks) == 2
    assert all(r.size == 2 for r in rocks)
    assert all(r.radius == env.config.asteroid_medium_radius for r in rocks)
    # Fragments must actually separate, even from a stationary parent
    speeds = [np.hypot(*r.vel) for r in rocks]
    assert all(s >= env.config.asteroid_split_min_speed - 1e-9 for s in speeds)
    assert env._wrapped_dist(rocks[0].pos, rocks[1].pos) > rocks[0].radius


def test_smallest_asteroid_is_destroyed_not_split():
    env = _shoot_rock(1, split=True)
    assert [r for r in env.asteroids if r.active] == []


def test_non_splitting_asteroids_are_indestructible_cover():
    env = _shoot_rock(3, split=False)
    rocks = [r for r in env.asteroids if r.active]
    assert len(rocks) == 1 and rocks[0].size == 3
    assert not any(b.active for b in env.bullets)  # the rock still absorbs the shot


def test_split_fragments_appear_in_observations():
    env = _shoot_rock(3, split=True)
    from marl_asteroids.env import ASTEROID_DIM, BASE_DIM, EGO_DIM, ENEMY_DIM, ALLY_DIM
    start = EGO_DIM + env.config.obs_enemies * ENEMY_DIM + env.config.obs_allies * ALLY_DIM
    slots = env.get_obs("red_0")[start:start + 2 * ASTEROID_DIM]
    assert slots[0] == 1.0 and slots[ASTEROID_DIM] == 1.0  # both fragments present


# ------------------------------------------------------------------- handicaps


def test_set_team_fire_rate_only_affects_that_team():
    cfg = set_team_fire_rate(get_scenario("demo_team2"), "red", 110)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    assert [env.ships[f"red_{i}"].fire_rate for i in range(2)] == [110, 110]
    assert [env.ships[f"blue_{i}"].fire_rate for i in range(2)] == [cfg.fire_rate] * 2


def test_set_team_fire_rate_preserves_static_spawns():
    base, handicapped = get_scenario("demo_team2"), None
    handicapped = set_team_fire_rate(base, "red", 90)
    a, b = TeamAsteroidsEnv(base), TeamAsteroidsEnv(handicapped)
    a.reset(seed=0)
    b.reset(seed=0)
    for agent in cfg_agents(base):
        np.testing.assert_array_equal(a.ships[agent].pos, b.ships[agent].pos)


def test_set_team_fire_rate_works_on_randomized_scenarios():
    cfg = set_team_fire_rate(get_scenario("team3"), "blue", 77)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    assert all(env.ships[f"blue_{i}"].fire_rate == 77 for i in range(3))
    assert all(env.ships[f"red_{i}"].fire_rate == cfg.fire_rate for i in range(3))


class _Recorder(Policy):
    name = "recorder"

    def __init__(self):
        self.seen = []

    def act(self, obs, contexts):
        self.seen.append(np.asarray(obs).copy())
        return np.zeros((len(obs), 3), dtype=np.int64)


def test_delayed_policy_lags_by_requested_steps_and_resets():
    env = TeamAsteroidsEnv(get_scenario("demo_duel"))
    rec = _Recorder()
    delayed = DelayedPolicy(rec, 3)
    env.reset(seed=0)
    history = []
    for _ in range(8):
        obs = np.stack([env.get_obs(a) for a in env.agents])
        history.append(obs.copy())
        delayed.act(obs, [(env, a) for a in env.agents])
        env.step({a: [1, 1, 0] for a in env.agents})
    for i, seen in enumerate(rec.seen):
        np.testing.assert_allclose(seen, history[max(0, i - 3)])
    env.reset(seed=1)  # a new episode must not leak the previous one's observations
    obs = np.stack([env.get_obs(a) for a in env.agents])
    delayed.act(obs, [(env, a) for a in env.agents])
    np.testing.assert_allclose(rec.seen[-1], obs)


def test_zero_delay_is_passthrough():
    rec = _Recorder()
    d = DelayedPolicy(rec, 0)
    env = TeamAsteroidsEnv(get_scenario("duel"))
    env.reset(seed=0)
    obs = np.stack([env.get_obs(a) for a in env.agents])
    d.act(obs, [(env, a) for a in env.agents])
    np.testing.assert_allclose(rec.seen[0], obs)


def test_thrusting_flag_tracks_commanded_thrust():
    env = TeamAsteroidsEnv(get_scenario("duel"))
    env.reset(seed=0)
    env.step({"red_0": [1, 0, 0], "blue_0": [0, 0, 0]})
    assert env.ships["red_0"].thrusting and not env.ships["blue_0"].thrusting


# ------------------------------------------------------- large maps / scaling


def test_bearing_survives_beyond_sensor_range():
    """Past sensor_range the range feature saturates, but bearing must stay exact and distinct."""
    from marl_asteroids.env import EGO_DIM
    cfg = get_scenario("duel", width=4000.0, height=4000.0)

    def probe(bearing_deg, dist):
        env = TeamAsteroidsEnv(cfg)
        env.reset(seed=0)
        b = math.radians(bearing_deg)
        _setup(env, {"blue_0": (2000, 2000, 0.0),
                     "red_0": (2000 + math.cos(b) * dist, 2000 + math.sin(b) * dist, 0.0)})
        return env.get_obs("blue_0")[EGO_DIM:EGO_DIM + 4]

    for dist in (300, 900, 2000):
        feats = {bd: probe(bd, dist) for bd in (20, 45, 70)}
        for bd, f in feats.items():
            # unit bearing vector, exact at any range
            assert np.hypot(f[1], f[2]) == pytest.approx(1.0, abs=1e-4)
            assert math.degrees(math.atan2(f[2], f[1])) == pytest.approx(bd, abs=0.5)
        bearings = [tuple(np.round(f[1:3], 4)) for f in feats.values()]
        assert len(set(bearings)) == 3, f"bearings collapsed at {dist}px"


def test_range_feature_is_monotone_and_never_saturates():
    from marl_asteroids.env import EGO_DIM
    cfg = get_scenario("duel", width=8000.0, height=8000.0, sensor_range=500.0)
    vals = []
    for dist in (100, 500, 1000, 3000):
        env = TeamAsteroidsEnv(cfg)
        env.reset(seed=0)
        _setup(env, {"blue_0": (4000, 4000, 0.0), "red_0": (4000 + dist, 4000, 0.0)})
        vals.append(env.get_obs("blue_0")[EGO_DIM + 3])
    assert all(a < b for a, b in zip(vals, vals[1:])), vals
    assert vals[1] == pytest.approx(0.5)  # 0.5 exactly at sensor_range
    assert vals[-1] < 1.0


def test_scale_map_scales_geometry_not_ships():
    base = get_scenario("team2")
    big = scale_map(base, 3.0)
    assert (big.width, big.height) == (base.width * 3, base.height * 3)
    assert big.max_frames == int(base.max_frames * 3)
    assert big.min_spawn_dist == base.min_spawn_dist * 3
    assert big.num_asteroids == round(base.num_asteroids * 3)
    # ships, weapons and sensing are untouched: a genuinely bigger world, not a zoom
    for field in ("ship_radius", "max_speed", "bullet_speed", "fire_rate", "sensor_range"):
        assert getattr(big, field) == getattr(base, field)


def test_scale_map_keeps_static_layouts_symmetric():
    big = scale_map(get_scenario("demo_team2"), 2.0, asteroid_density=0.0)
    env = TeamAsteroidsEnv(big)
    env.reset(seed=0)
    w, h = big.width, big.height
    assert big.num_asteroids == len(big.asteroid_spawns) == 4  # still fully deterministic
    reds = sorted(tuple(np.round(env.ships[f"red_{i}"].pos, 6)) for i in range(big.n_red))
    blues = sorted(tuple(np.round(((w - env.ships[f"blue_{i}"].pos[0]) % w,
                                  (h - env.ships[f"blue_{i}"].pos[1]) % h), 6))
                   for i in range(big.m_blue))
    assert reds == blues
    # spawns really did move out with the map rather than clustering near the origin
    assert all(env.ships[a].pos.max() > big.width * 0.25 for a in cfg_agents(big))


@pytest.mark.parametrize("name", ["duel_large", "team2_large", "team3_large", "team3_huge",
                                  "demo_team2_large"])
def test_large_scenarios_are_playable(name):
    cfg = get_scenario(name)
    assert cfg.width > 1200
    stats = play_matches(cfg, make_scripted("heuristic_1.0", 0), make_scripted("heuristic_1.0", 1),
                         4, n_parallel=4, seed=0)
    assert stats.games == 4
    assert sum(stats.kills.values()) > 0, "bots cannot find each other on this map"


def test_static_large_scenario_is_still_symmetric():
    stats = play_matches(get_scenario("demo_team2_large"), make_scripted("heuristic_1.0"),
                         make_scripted("heuristic_1.0"), 2, n_parallel=2, seed=0)
    assert stats.blue_wins == 0 and stats.red_wins == 0
