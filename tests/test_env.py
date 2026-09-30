import math

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from marl_asteroids.config import GameConfig, EntityState, get_scenario, SCENARIOS
from marl_asteroids.env import BLUE, RED, TeamAsteroidsEnv, obs_dim


def run_random(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    traj = []
    env.reset(seed=seed)
    for _ in range(steps):
        if not env.agents:
            env.reset()
        acts = {a: rng.integers([2, 3, 2]) for a in env.agents}
        o, r, *_ = env.step(acts)
        traj.append((sorted(o), {k: v.copy() for k, v in o.items()}, dict(r)))
    return traj


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_parallel_api(name):
    parallel_api_test(TeamAsteroidsEnv(get_scenario(name)), num_cycles=200)


def test_obs_shape_and_bounds():
    for name in SCENARIOS:
        env = TeamAsteroidsEnv(get_scenario(name))
        for _, obs, _ in run_random(env, 300):
            for o in obs.values():
                assert o.shape == (obs_dim(env.config),)
                assert o.dtype == np.float32
                assert np.all(o >= -1) and np.all(o <= 1)


def test_seeding_is_deterministic():
    a = run_random(TeamAsteroidsEnv(get_scenario("team3")), 200, seed=7)
    b = run_random(TeamAsteroidsEnv(get_scenario("team3")), 200, seed=7)
    for (ka, oa, ra), (kb, ob, rb) in zip(a, b):
        assert ka == kb and ra == rb
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])


def _place(env, agent, x, y, angle=0.0, vel=(0.0, 0.0)):
    s = env.ships[agent]
    s.pos[:] = (x, y)
    s.angle = angle
    s.vel[:] = vel


def _no_shaping(cfg):
    cfg.rewards.shaping_scale = 0.0
    return cfg


def test_speed_is_capped():
    env = TeamAsteroidsEnv(GameConfig(n_red=1, m_blue=0, num_asteroids=0))
    env.reset(seed=0)
    for _ in range(300):
        env.step({"red_0": [1, 0, 0]})
    assert np.hypot(*env.ships["red_0"].vel) <= env.config.max_speed + 1e-6


def test_ally_slots_exclude_self():
    cfg = GameConfig(n_red=1, m_blue=2, num_asteroids=0)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "blue_0", 100, 100)
    _place(env, "blue_1", 300, 100)
    _place(env, "red_0", 600, 600)
    env._compute_obs()
    from marl_asteroids.env import EGO_DIM, ENEMY_DIM, ALLY_DIM
    start = EGO_DIM + cfg.obs_enemies * ENEMY_DIM
    o = env.get_obs("blue_0")
    ally0, ally1 = o[start:start + ALLY_DIM], o[start + ALLY_DIM:start + 2 * ALLY_DIM]
    # Present, and at the real 200px separation rather than self at distance 0
    assert ally0[0] == 1.0
    assert ally0[3] == pytest.approx(200 / (200 + cfg.sensor_range), abs=1e-4)
    assert np.hypot(ally0[1], ally0[2]) == pytest.approx(1.0, abs=1e-4)  # unit bearing
    assert np.all(ally1 == 0.0)  # only one real ally -> second slot padded


def test_teamkill_is_penalized_not_rewarded():
    cfg = _no_shaping(GameConfig(n_red=1, m_blue=2, num_asteroids=0, friendly_fire=True))
    cfg.rewards.team_spirit = 0.0
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "blue_0", 100, 400, angle=0.0)
    _place(env, "blue_1", 160, 400)
    _place(env, "red_0", 600, 100)
    total = {a: 0.0 for a in env.possible_agents}
    acts = {"blue_0": [0, 0, 1], "blue_1": [0, 0, 0], "red_0": [0, 0, 0]}
    for _ in range(20):
        env.step(acts)
        for a, r in env.rewards_all.items():
            total[a] += r
        acts = {a: [0, 0, 0] for a in env.agents}
        if not env.ships["blue_1"].active:
            break
    assert ("blue_0", "blue_1") in env.kill_log
    assert not env.game_over
    assert total["blue_0"] == pytest.approx(cfg.rewards.teamkill)
    assert total["blue_1"] == pytest.approx(cfg.rewards.death)


def test_kill_reward_win_and_dead_teammate_gets_outcome():
    cfg = _no_shaping(GameConfig(n_red=1, m_blue=2, num_asteroids=0))
    cfg.rewards.team_spirit = 0.5
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "blue_0", 100, 400, angle=0.0)
    _place(env, "red_0", 160, 400)
    _place(env, "blue_1", 600, 100)
    env.ships["blue_1"].active = False  # already dead teammate
    env.agents = ["blue_0", "red_0"]
    env.step({"blue_0": [0, 0, 1], "red_0": [0, 0, 0]})
    for _ in range(20):
        if env.game_over:
            break
        env.step({a: [0, 0, 0] for a in env.agents})
    assert env.game_over and env.winner == BLUE
    rc = cfg.rewards
    # shooter: (1-ts)*kill + ts*mean(kill, 0) + win
    assert env.rewards_all["blue_0"] == pytest.approx((1 - 0.5) * rc.kill + 0.5 * rc.kill / 2 + rc.win)
    # dead teammate still receives team-spirit share + win
    assert env.rewards_all["blue_1"] == pytest.approx(0.5 * rc.kill / 2 + rc.win)
    assert env.rewards_all["red_0"] == pytest.approx(rc.death + rc.loss)


def test_entities_use_instance_config():
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=1, ship_radius=17.0, asteroid_large_radius=44.0,
                     bullet_radius=3.5, bullet_lifespan=11)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    assert env.ships["red_0"].radius == 17.0
    assert env.asteroids[0].radius == 44.0
    env.step({"red_0": [0, 0, 1], "blue_0": [0, 0, 0]})
    assert env.bullets[0].radius == 3.5 and env.bullets[0].lifespan <= 11


def test_truncation_and_max_frames():
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=0, max_frames=40, action_repeat=2)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "red_0", 100, 100)
    _place(env, "blue_0", 500, 500)
    steps = 0
    while env.agents:
        _, _, term, trunc, _ = env.step({a: [0, 0, 0] for a in env.agents})
        steps += 1
    assert steps == 20 and env.frame == 40
    assert all(trunc.values()) and not any(term.values()) and env.winner == -1


def test_potential_shaping_telescopes():
    """With gamma=1 the summed shaping over an episode equals -phi(s0) (terminal phi = 0)."""
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=2, max_frames=200)
    rc = cfg.rewards
    rc.shaping_gamma, rc.team_spirit = 1.0, 0.0
    rc.kill = rc.death = rc.win = rc.loss = rc.draw = rc.teamkill = 0.0
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=3)
    phi0 = env._phi.copy()
    total = {a: 0.0 for a in env.possible_agents}
    rng = np.random.default_rng(0)
    while env.agents:
        env.step({a: rng.integers([2, 3, 2]) for a in env.agents})
        for a, r in env.rewards_all.items():
            total[a] += r
    for i, a in enumerate(env.possible_agents):
        assert total[a] == pytest.approx(-rc.shaping_scale * phi0[i], abs=1e-9)


def test_lead_aim_points_at_intercept():
    """Target crossing perpendicular: the aim feature should lead ahead of the target."""
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=0)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "blue_0", 400, 400, angle=0.0)
    _place(env, "red_0", 600, 400, vel=(0.0, 3.0))  # moving +y (down in screen coords)
    env._compute_obs()
    from marl_asteroids.env import EGO_DIM, BASE_DIM
    o = env.get_obs("blue_0")
    aim_cos, aim_sin = o[EGO_DIM + BASE_DIM], o[EGO_DIM + BASE_DIM + 1]
    expected = math.asin(3.0 / cfg.bullet_speed)  # classic intercept angle for a crossing target
    assert math.atan2(aim_sin, aim_cos) == pytest.approx(expected, abs=0.02)
