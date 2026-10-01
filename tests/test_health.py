"""Health, regenerating shield, collision bounce, damage rewards, and the HP timeout tiebreak."""
import numpy as np
import pytest

from marl_asteroids.config import EntityState, GameConfig, get_scenario
from marl_asteroids.env import BLUE, RED, TeamAsteroidsEnv


def _duel(**kw):
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=0, **kw)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    return env


def _place(env, agent, x, y, angle=0.0, vel=(0.0, 0.0)):
    s = env.ships[agent]
    s.pos[:] = (x, y)
    s.angle = angle
    s.vel[:] = vel


# ------------------------------------------------------------------ durability


def test_shield_absorbs_before_hull():
    env = _duel()
    s = env.ships["red_0"]
    assert (s.hull, s.shield, s.hp, s.max_hp) == (3, 1, 4, 4)
    assert s.apply_damage(1) == 1
    assert (s.hull, s.shield) == (3, 0)
    assert s.apply_damage(2) == 2
    assert (s.hull, s.shield) == (1, 0) and s.active


def test_damage_dealt_is_clamped_to_remaining_hp():
    env = _duel()
    s = env.ships["red_0"]
    assert s.apply_damage(99) == 4  # never reports more than the ship had
    assert s.hull == 0 and not s.active


def test_damage_on_a_dead_ship_is_ignored():
    env = _duel()
    s = env.ships["red_0"]
    s.apply_damage(99)
    assert s.apply_damage(5) == 0


def test_shield_regenerates_only_after_the_delay():
    env = _duel(shield_regen_delay=100, shield_regen_frames=20)
    s = env.ships["red_0"]
    s.apply_damage(1)
    assert s.shield == 0
    for _ in range(99):
        s.tick_shield(100, 20)
    assert s.shield == 0, "regen started before the delay elapsed"
    for _ in range(20):
        s.tick_shield(100, 20)
    assert s.shield == 1


def test_shield_regen_resets_when_hit_again():
    env = _duel(shield_regen_delay=50, shield_regen_frames=10)
    s = env.ships["red_0"]
    s.apply_damage(1)
    for _ in range(55):
        s.tick_shield(50, 10)
    s.apply_damage(1)  # takes hull damage and restarts the clock
    assert s.shield == 0 and s.frames_since_damage == 0
    for _ in range(49):
        s.tick_shield(50, 10)
    assert s.shield == 0


def test_shield_never_exceeds_max():
    env = _duel(shield_regen_delay=1, shield_regen_frames=1)
    s = env.ships["red_0"]
    for _ in range(500):
        s.tick_shield(1, 1)
    assert s.shield == s.max_shield == 1


def test_bullets_take_several_hits_to_kill():
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=0, friendly_fire=False,
                     shield_regen_delay=10 ** 6, max_frames=4000)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "red_0", 300, 400, angle=0.0)
    _place(env, "blue_0", 420, 400, angle=0.0)
    hits = 0
    for _ in range(600):
        if not env.ships["blue_0"].active:
            break
        before = env.ships["blue_0"].hp
        env.step({"red_0": [0, 0, 1], "blue_0": [0, 0, 0]})
        _place(env, "red_0", 300, 400, angle=0.0)  # hold station so every shot lands
        _place(env, "blue_0", 420, 400, angle=0.0)
        if env.ships["blue_0"].hp < before:
            hits += 1
    assert not env.ships["blue_0"].active
    assert hits == cfg.max_hull + cfg.max_shield == 4


# ------------------------------------------------------------- collision bounce


def test_ram_damages_both_once_and_separates_them():
    env = _duel(max_frames=600)
    _place(env, "red_0", 390, 400, vel=(2.0, 0.0))
    _place(env, "blue_0", 415, 400, vel=(-2.0, 0.0))
    env.step({a: [0, 0, 0] for a in env.agents})
    hp = [env.ships[a].hp for a in ("red_0", "blue_0")]
    assert hp == [4 - env.config.ram_damage] * 2
    d0 = env._wrapped_dist(env.ships["red_0"].pos, env.ships["blue_0"].pos)
    for _ in range(4):
        env.step({a: [0, 0, 0] for a in env.agents})
    # No repeat damage while overlapping, and they are moving apart
    assert [env.ships[a].hp for a in ("red_0", "blue_0")] == hp
    assert env._wrapped_dist(env.ships["red_0"].pos, env.ships["blue_0"].pos) > d0


def test_asteroid_contact_damages_once_and_bounces():
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=1, max_frames=600,
                     asteroid_spawns=[EntityState(x=500.0, y=400.0)])
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "red_0", 455, 400, vel=(3.0, 0.0))
    _place(env, "blue_0", 100, 100)
    env.step({a: [0, 0, 0] for a in env.agents})
    hp = env.ships["red_0"].hp
    assert hp == 4 - cfg.asteroid_damage
    assert env.ships["red_0"].vel[0] < 0, "did not bounce back off the rock"
    for _ in range(5):
        env.step({a: [0, 0, 0] for a in env.agents})
    assert env.ships["red_0"].hp == hp  # cooldown prevents per-frame re-damage


def test_collision_cooldown_expires():
    env = _duel(collision_damage_cooldown=10, max_frames=600)
    s = env.ships["red_0"]
    s.collision_cooldown = 10
    for _ in range(6):  # 6 decisions x action_repeat 2 = 12 frames
        env.step({a: [0, 0, 0] for a in env.agents})
    assert s.collision_cooldown == 0


# ------------------------------------------------------------- damage rewards


def test_damage_without_a_kill_still_pays():
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=0, friendly_fire=False, max_frames=4000)
    cfg.rewards.shaping_scale = 0.0
    cfg.rewards.team_spirit = 0.0
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "red_0", 300, 400, angle=0.0)
    _place(env, "blue_0", 420, 400, angle=0.0)
    rc = cfg.rewards
    for _ in range(80):
        env.step({"red_0": [0, 0, 1], "blue_0": [0, 0, 0]})
        _place(env, "red_0", 300, 400, angle=0.0)
        _place(env, "blue_0", 420, 400, angle=0.0)
        if env.damage_log:
            break
    assert env.damage_log, "no hit landed"
    assert env.rewards_all["red_0"] == pytest.approx(rc.damage_dealt)
    assert env.rewards_all["blue_0"] == pytest.approx(rc.damage_taken)
    assert not env.game_over  # a single hit is no longer a kill


def test_kill_pays_more_than_chip_damage():
    """Finishing a target must beat spreading the same damage around."""
    rc = GameConfig().rewards
    full_kill = rc.damage_dealt * 4 + rc.kill
    spread = rc.damage_dealt * 4
    assert full_kill > spread * 1.5


# ---------------------------------------------------------------- HP tiebreak


def _timeout_with_hp(red_hp, blue_hp):
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=0, max_frames=4,
                     shield_regen_delay=10 ** 6)
    cfg.rewards.shaping_scale = 0.0
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "red_0", 100, 100)
    _place(env, "blue_0", 700, 700)
    env.ships["red_0"].hull = red_hp
    env.ships["red_0"].shield = 0
    env.ships["blue_0"].hull = blue_hp
    env.ships["blue_0"].shield = 0
    while env.agents:
        env.step({a: [0, 0, 0] for a in env.agents})
    return env


def test_timeout_is_won_by_the_healthier_team():
    env = _timeout_with_hp(red_hp=3, blue_hp=1)
    assert env.winner == RED
    rc = env.config.rewards
    assert env.rewards_all["red_0"] == pytest.approx(rc.win)
    assert env.rewards_all["blue_0"] == pytest.approx(rc.loss)

    env = _timeout_with_hp(red_hp=1, blue_hp=3)
    assert env.winner == BLUE


def test_timeout_with_equal_hp_is_a_draw():
    env = _timeout_with_hp(red_hp=2, blue_hp=2)
    assert env.winner == -1
    assert env.rewards_all["red_0"] == pytest.approx(env.config.rewards.draw)


def test_team_hp_sums_living_ships_only():
    cfg = GameConfig(n_red=2, m_blue=1, num_asteroids=0)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    assert env.team_hp(RED) == 8
    env.ships["red_0"].apply_damage(99)
    assert env.team_hp(RED) == 4


# ------------------------------------------------------------- no time limit


def test_no_timeout_scenarios_never_truncate():
    cfg = GameConfig(n_red=1, m_blue=1, num_asteroids=0, max_frames=None)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "red_0", 100, 100)
    _place(env, "blue_0", 700, 700)
    for _ in range(400):
        _, _, term, trunc, _ = env.step({a: [0, 0, 0] for a in env.agents})
        assert not any(trunc.values()) and not any(term.values())
    assert env.agents and not env.game_over
    assert env.get_obs("red_0")[7] == pytest.approx(1.0)  # time_left pinned with no limit


@pytest.mark.parametrize("name", ["demo_duel", "demo_team2", "demo_team3", "demo_open"])
def test_demo_scenarios_have_no_time_limit(name):
    assert get_scenario(name).max_frames is None


def test_scale_map_preserves_no_time_limit():
    from marl_asteroids.config import scale_map
    assert scale_map(get_scenario("demo_team2"), 2.0).max_frames is None
    assert scale_map(get_scenario("team2"), 2.0).max_frames == 2400


# ------------------------------------------------------------- observations


def test_health_appears_in_ego_and_ship_slots():
    from marl_asteroids.env import ALLY_DIM, EGO_DIM, ENEMY_DIM
    cfg = GameConfig(n_red=1, m_blue=2, num_asteroids=0, shield_regen_delay=10 ** 6)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    _place(env, "blue_0", 300, 300)
    _place(env, "blue_1", 400, 300)
    _place(env, "red_0", 600, 300)
    env.ships["blue_0"].apply_damage(2)   # hull 2/3, shield 0
    env.ships["red_0"].apply_damage(1)    # hull 3/3, shield 0
    env._compute_obs()
    o = env.get_obs("blue_0")
    assert o[8] == pytest.approx(2 / 3) and o[9] == pytest.approx(0.0)      # own hull, shield
    enemy = o[EGO_DIM:EGO_DIM + ENEMY_DIM]
    assert enemy[ENEMY_DIM - 2] == pytest.approx(1.0)                        # enemy hull full
    assert enemy[ENEMY_DIM - 1] == pytest.approx(0.0)                        # enemy shield down
    ally = o[EGO_DIM + cfg.obs_enemies * ENEMY_DIM:][:ALLY_DIM]
    assert ally[ALLY_DIM - 2] == pytest.approx(1.0) and ally[ALLY_DIM - 1] == pytest.approx(1.0)


def test_focus_fire_metric_counts_hits_on_wounded_targets():
    from marl_asteroids.matches import MatchStats, record_game
    cfg = GameConfig(n_red=1, m_blue=2, num_asteroids=0, max_frames=10)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    env.winner = BLUE
    env.damage_log = [("blue_0", "red_0", 1, False),   # first hit: not focus
                      ("blue_1", "red_0", 1, False),   # target already wounded: focus
                      ("blue_0", "red_0", 2, True)]    # still focus
    stats = MatchStats()
    record_game(env, stats)
    assert stats.damage["blue"] == 4
    assert stats.focused_damage["blue"] == 3
    assert stats.summary()["blue_focus_fire_rate"] == pytest.approx(0.75)
