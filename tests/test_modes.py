"""Game modes: arena builder, station siege, survival, and the objective-aware bots."""
import numpy as np
import pytest

from marl_asteroids.config import (ARENA_ASTEROIDS, ARENA_SIZES, MODE_TIME_LIMIT, GameConfig,
                                   make_game)
from marl_asteroids.entities import Bullet, Station
from marl_asteroids.env import BLUE, RED, TeamAsteroidsEnv, obs_dim
from marl_asteroids.policies import make_scripted


def env_for(mode, arena="medium", m_blue=2, n_red=2, seed=0, **overrides):
    cfg = make_game(mode, arena, m_blue, n_red, num_asteroids=0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=seed)
    return env


def idle(env, steps=1):
    for _ in range(steps):
        env.step({a: [0, 0, 0] for a in env.agents})


def park(env, agent, x, y):
    s = env.ships[agent]
    s.pos[:] = (x, y)
    s.vel[:] = 0.0


def play(env, blue="heuristic_1.0", red="heuristic_1.0", max_steps=20000):
    b, r = make_scripted(blue, 1), make_scripted(red, 2)
    for _ in range(max_steps):
        if not env.agents:
            break
        acts = {}
        for pol, team in ((b, "blue"), (r, "red")):
            ags = [a for a in env.agents if a.startswith(team)]
            if ags:
                acts.update(zip(ags, pol.act(None, [(env, a) for a in ags])))
        env.step(acts)
    return env


# ------------------------------------------------------------------------- arena builder


def test_make_game_sizes_and_defaults():
    for arena, size in ARENA_SIZES.items():
        for mode in ("deathmatch", "siege", "survival"):
            cfg = make_game(mode, arena, 2, 3)
            assert (cfg.width, cfg.height) == (size, size)
            assert cfg.num_asteroids == ARENA_ASTEROIDS[arena]
            limit = MODE_TIME_LIMIT[mode]
            assert cfg.max_frames == (None if limit is None else int(limit * 60))
            assert (cfg.m_blue, cfg.n_red, cfg.mode) == (2, 3, mode)


def test_make_game_overrides_and_validation():
    cfg = make_game("siege", "small", 1, 1, num_asteroids=0, asteroids_split=False,
                    time_limit=None, use_mode_time_limit=False)
    assert cfg.num_asteroids == 0 and not cfg.asteroids_split and cfg.max_frames is None
    for bad in (dict(mode="tag"), dict(arena="galactic"), dict(m_blue=0)):
        with pytest.raises(ValueError):
            make_game(**{"mode": "deathmatch", "arena": "small", **bad})


def test_fixed_deathmatch_spawns_face_off_and_random_spawns_vary():
    a = env_for("deathmatch", seed=1)
    b = env_for("deathmatch", seed=2)
    for agent in a.possible_agents:
        np.testing.assert_array_equal(a.ships[agent].pos, b.ships[agent].pos)
    assert all(a.ships[f"red_{i}"].pos[0] < a.ships[f"blue_{i}"].pos[0] for i in range(2))
    cfg = make_game("deathmatch", "medium", 2, 2, fixed_spawns=False)
    r1, r2 = TeamAsteroidsEnv(cfg), TeamAsteroidsEnv(cfg)
    r1.reset(seed=1)
    r2.reset(seed=2)
    assert not np.allclose(r1.ships["red_0"].pos, r2.ships["red_0"].pos)


@pytest.mark.parametrize("mode", ["deathmatch", "siege", "survival"])
def test_observation_size_is_unchanged_so_trained_agents_load(mode):
    env = env_for(mode)
    assert obs_dim(env.config) == 221
    assert env.get_obs("blue_0").shape == (221,)


# ------------------------------------------------------------------------------ station


def test_station_shield_absorbs_first_and_regenerates_continuously():
    st = Station(0, 0, RED, GameConfig(station_shield=4, station_hull=5, station_shield_regen=6.0))
    assert st.apply_damage(3) == 3 and (st.shield, st.hull) == (1, 5)
    assert st.apply_damage(2) == 2 and (st.shield, st.hull) == (0, 4)
    for _ in range(10):  # 6 points/s = one point every 10 frames, with no quiet period needed
        st.tick()
    assert st.shield == 1
    st.apply_damage(99)
    assert not st.active and st.hp == 0
    st.tick()
    assert st.shield == 0  # a destroyed station stays destroyed


def test_full_station_does_not_bank_regen():
    st = Station(0, 0, RED, GameConfig(station_shield=4, station_shield_regen=60.0))
    for _ in range(100):
        st.tick()
    st.apply_damage(4)
    st.tick()
    assert st.shield == 1  # one frame's worth, not 100 frames of saved-up regen


def frames_to_destroy(n_ships, cfg=None, hit_rate=1.0, limit=60 * 600):
    """Uncontested attack: each ship lands ``hit_rate`` of its shots at the max fire rate."""
    cfg = cfg or GameConfig()
    st = Station(0, 0, BLUE, cfg)
    shot_every = cfg.fire_rate / hit_rate
    acc = 0.0
    for frame in range(1, limit):
        acc += n_ships / shot_every
        while acc >= 1:
            acc -= 1
            st.apply_damage(cfg.bullet_damage)
        st.tick()
        if not st.active:
            return frame
    return None


def test_station_needs_more_than_one_attacker():
    """The siege tuning: a lone ship is slow, two or three ships take a station quickly."""
    solo = frames_to_destroy(1)
    pair = frames_to_destroy(2)
    trio = frames_to_destroy(3)
    assert solo > 60 * 60          # over a minute even if every shot lands
    assert pair < 30 * 60          # well under half a minute
    assert trio < 15 * 60
    assert frames_to_destroy(1, hit_rate=0.75) is None  # a realistic lone attacker never gets through


def test_stations_sit_half_a_map_apart():
    env = env_for("siege")
    red, blue = env.stations[RED], env.stations[BLUE]
    _, _, d = env.wrapped_delta(red.pos, blue.pos)
    assert d == pytest.approx(env.width / 2)  # same distance both ways round the torus


def test_ships_start_fanned_out_in_front_of_their_station_and_protected():
    env = env_for("siege", m_blue=3, n_red=3)
    for agent, ship in env.ships.items():
        st = env.stations[ship.team_id]
        d = env._wrapped_dist(ship.pos, st.pos)
        assert st.radius < d < st.radius + 80
        assert ship.protection > 0


# -------------------------------------------------------------------------- siege rules


def test_enemy_fire_damages_a_station_and_own_fire_passes_through():
    env = env_for("siege")
    blue_st = env.stations[BLUE]
    for owner, team in (("red_0", RED), ("blue_0", BLUE)):
        b = Bullet(blue_st.pos[0] - blue_st.radius - 1, blue_st.pos[1], 3.0, 0.0, owner, team, env.config)
        env.bullets = [b]
        before = blue_st.hp
        env._resolve_collisions()
        if team == RED:
            assert blue_st.hp == before - env.config.bullet_damage and not b.active
            assert env._station_events and env._station_events[0][:2] == ("red_0", BLUE)
        else:
            assert blue_st.hp == before and b.active


def test_station_damage_rewards_attacker_and_costs_defenders():
    env = env_for("siege")
    rc = env.config.rewards
    rc.shaping_scale = 0.0
    rc.team_spirit = 0.0
    st = env.stations[BLUE]
    env.bullets = [Bullet(st.pos[0] - st.radius - 1, st.pos[1], 3.0, 0.0, "red_0", RED, env.config)]
    idle(env)
    assert env.rewards_all["red_0"] == pytest.approx(rc.station_damage)
    assert env.rewards_all["blue_0"] == pytest.approx(-rc.station_damage / 2)
    assert env.station_damage[RED] == 1


def test_destroyed_ship_respawns_at_its_station_while_it_stands():
    env = env_for("siege")
    env.ships["blue_0"].protection = 0
    env.ships["blue_0"].apply_damage(99)
    idle(env)
    ship = env.ships["blue_0"]
    assert not ship.active and ship.respawn_timer is not None
    assert "blue_0" in env.agents                      # still in the game while waiting
    assert np.all(env.get_obs("blue_0") == 0)          # dead ships see nothing
    idle(env, env.config.respawn_delay // env.config.action_repeat + 1)
    assert ship.active and ship.hp == ship.max_hp and ship.protection > 0
    assert env._wrapped_dist(ship.pos, env.stations[BLUE].pos) < env.stations[BLUE].radius + 80


def test_kills_are_still_logged_when_ships_respawn():
    """Results and stats read the kill log, so a respawning ship's deaths must still count."""
    env = env_for("siege")
    ship = env.ships["red_0"]
    ship.protection = 0
    ship.apply_damage(3)  # one hull point left
    park(env, "red_0", 50, 50)
    env.bullets = [Bullet(48, 50, 1.0, 0.0, "blue_0", BLUE, env.config)]
    idle(env)
    assert ("blue_0", "red_0") in env.kill_log and ship.respawn_timer is not None


def test_spawn_protection_blocks_all_damage():
    env = env_for("siege")
    ship = env.ships["red_0"]
    assert ship.protection > 0
    assert ship.apply_damage(5) == 0 and ship.hp == ship.max_hp


def test_losing_the_station_ends_respawns_and_then_the_game():
    env = env_for("siege")
    for a in env.team_agents[BLUE]:
        env.ships[a].protection = 0
        env.ships[a].apply_damage(99)
    idle(env)
    assert all(env.ships[a].respawn_timer is not None for a in env.team_agents[BLUE])
    st = env.stations[BLUE]
    st.apply_damage(st.hp - 1)
    env.bullets = [Bullet(st.pos[0] - st.radius - 1, st.pos[1], 3.0, 0.0, "red_0", RED, env.config)]
    idle(env)
    assert not st.active
    assert env.game_over and env.winner == RED
    assert all(env.ships[a].respawn_timer is None for a in env.team_agents[BLUE])


def test_station_alone_keeps_a_team_in_the_game():
    env = env_for("siege")
    for a in env.team_agents[RED]:
        env.ships[a].protection = 0
        env.ships[a].apply_damage(99)
    idle(env)
    assert not env.game_over  # red will respawn


def test_siege_timeout_goes_to_the_healthier_station():
    env = env_for("siege", max_frames=6)
    env.stations[RED].apply_damage(5)
    idle(env, 3)
    assert env.game_over and env.winner == BLUE
    assert env.summary()["stations"][RED]["hp"] < env.summary()["stations"][BLUE]["hp"]


def test_ramming_an_enemy_station_hurts_the_ship_not_the_station():
    env = env_for("siege")
    st = env.stations[BLUE]
    ship = env.ships["red_0"]
    ship.protection = 0
    park(env, "red_0", st.pos[0] - st.radius - ship.radius + 3, st.pos[1])
    ship.vel[:] = (2.0, 0.0)
    before = st.hp
    env._resolve_collisions()
    assert st.hp == before
    assert ship.hp == ship.max_hp - env.config.ram_damage
    assert ship.vel[0] < 0  # bounced off


def test_own_station_is_solid_but_harmless():
    env = env_for("siege")
    st = env.stations[RED]
    ship = env.ships["red_0"]
    ship.protection = 0
    park(env, "red_0", st.pos[0] - st.radius - ship.radius + 3, st.pos[1])
    ship.vel[:] = (2.0, 0.0)
    env._resolve_collisions()
    assert ship.hp == ship.max_hp and ship.vel[0] < 0


def test_asteroids_bounce_off_stations():
    cfg = make_game("siege", "medium", 1, 1, num_asteroids=1)
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    st = env.stations[RED]
    rock = env.asteroids[0]
    rock.pos[:] = (st.pos[0] - st.radius - rock.radius + 4, st.pos[1])
    rock.vel[:] = (1.5, 0.0)
    env._resolve_collisions()
    assert rock.vel[0] < 0 and env._wrapped_dist(rock.pos, st.pos) >= st.radius + rock.radius


def test_asteroids_never_spawn_on_a_station():
    cfg = make_game("siege", "small", 2, 2, num_asteroids=20)
    for seed in range(5):
        env = TeamAsteroidsEnv(cfg)
        env.reset(seed=seed)
        for rock in env.asteroids:
            for st in env.stations.values():
                assert env._wrapped_dist(rock.pos, st.pos) > st.radius + rock.radius


# ------------------------------------------------------------------------------ survival


def test_survival_starts_with_a_warm_up_wave_and_reserves():
    env = env_for("survival", m_blue=2, n_red=4)
    active = [a for a in env.team_agents[RED] if env.ships[a].active]
    reserve = [a for a in env.team_agents[RED] if env.ships[a].reserve]
    assert len(active) == env.wave_size(1) == 1 and len(reserve) == 3
    assert [env.wave_size(k) for k in range(1, 7)] == [1, 2, 3, 4, 4, 4]  # capped by n_red
    for a in active:  # enemies arrive far from blue
        d = min(env._wrapped_dist(env.ships[a].pos, env.ships[b].pos) for b in env.team_agents[BLUE])
        assert d >= env.config.wave_spawn_distance * env.width - 1e-6


def test_clearing_a_wave_revives_and_repairs_blue_then_sends_the_next():
    env = env_for("survival", m_blue=2, n_red=4)
    env.ships["blue_1"].apply_damage(99)
    env.ships["blue_0"].apply_damage(3)  # shield + 2 hull
    for a in env.team_agents[RED]:
        env.ships[a].protection = 0
        env.ships[a].apply_damage(99)
    idle(env)
    assert env.waves_cleared == 1 and env.wave_cooldown is not None
    assert env.ships["blue_1"].active
    assert env.ships["blue_0"].hull == env.ships["blue_0"].max_hull
    idle(env, env.config.wave_delay // env.config.action_repeat + 1)
    assert env.wave == 2 and env.wave_cooldown is None
    assert sum(env.ships[a].active for a in env.team_agents[RED]) == env.wave_size(2)
    assert not env.game_over


def test_survival_ends_when_blue_is_wiped_out():
    env = env_for("survival")
    for a in env.team_agents[BLUE]:
        env.ships[a].protection = 0
        env.ships[a].apply_damage(99)
    idle(env)
    assert env.game_over and env.winner == RED
    assert env.summary() == {"mode": "survival", "waves_cleared": 0, "wave": 1}


def test_survival_timeout_is_a_blue_win():
    env = env_for("survival", max_frames=4)
    idle(env, 2)
    assert env.game_over and env.winner == BLUE


@pytest.mark.parametrize("mode", ["siege", "survival"])
def test_respawning_modes_keep_every_agent_until_the_end(mode):
    env = env_for(mode, max_frames=40)
    everyone = list(env.agents)
    rng = np.random.default_rng(0)
    while env.agents:
        assert env.agents == everyone
        _, _, term, trunc, _ = env.step({a: rng.integers([2, 3, 2]) for a in env.agents})
        if env.agents:
            assert not any(term.values()) and not any(trunc.values())
    assert all(term.values()) or all(trunc.values())


# ---------------------------------------------------------------------------------- bots


def test_bots_take_an_undefended_station():
    env = env_for("siege", arena="small", max_frames=60 * 120)
    play(env, blue="heuristic_1.0", red="dummy")
    assert env.game_over and env.winner == BLUE
    assert not env.stations[RED].active


def test_bots_defend_and_attack_in_siege():
    env = env_for("siege", arena="small", max_frames=60 * 40)
    play(env)
    assert env.station_damage[RED] > 0 and env.station_damage[BLUE] > 0


def test_bots_play_survival():
    env = env_for("survival", arena="small", m_blue=2, n_red=3, max_frames=60 * 60)
    play(env)
    assert env.game_over and env.frame > 60  # a real fight, not an instant wipe


def test_deathmatch_bot_behaviour_is_unchanged_without_stations():
    """The heuristic only adds station logic when stations exist; deathmatch is bit-identical."""
    from marl_asteroids.config import get_scenario
    from marl_asteroids.matches import play_matches
    s = play_matches(get_scenario("demo_duel", max_frames=1500), make_scripted("heuristic_1.0"),
                     make_scripted("heuristic_1.0"), 2, n_parallel=2, seed=0)
    assert s.blue_wins == 0 and s.red_wins == 0  # symmetric arena still draws exactly


# -------------------------------------------------------------------------------- render


@pytest.mark.parametrize("mode", ["siege", "survival"])
@pytest.mark.parametrize("follow", [None, "blue_0"])
def test_modes_render(mode, follow):
    from marl_asteroids.render import Renderer
    env = env_for(mode, arena="small")
    r = Renderer(env, window=300, follow=follow)
    rng = np.random.default_rng(0)
    for _ in range(60):
        if not env.agents:
            env.reset(seed=1)
        env.step({a: rng.integers([2, 3, 2]) for a in env.agents})
        assert r.render("surface").get_size()[0] == 300
    env.ships["blue_0"].apply_damage(99)  # followed ship down: HUD shows the respawn message
    r.render("surface")
