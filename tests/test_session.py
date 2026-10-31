"""MatchSpec -> Match: overrides, controllers, handicaps, camera, serialization."""
import json

import pytest

from marl_asteroids.config import get_scenario
from marl_asteroids.policies import DelayedPolicy, HumanPolicy
from marl_asteroids.session import (DIFFICULTY, HUMAN, KEEP, PHYSICS_FPS, MatchSpec, agent_names,
                                    build_config, build_match)


def test_defaults_match_the_scenario():
    cfg = build_config(MatchSpec(scenario="team3"))
    base = get_scenario("team3")
    for field in ("n_red", "m_blue", "num_asteroids", "asteroids_split", "max_frames", "width"):
        assert getattr(cfg, field) == getattr(base, field)


def test_overrides_apply():
    cfg = build_config(MatchSpec(scenario="team2", n_red=3, m_blue=1, num_asteroids=7,
                                 asteroids_split=False, time_limit=90.0))
    assert (cfg.n_red, cfg.m_blue, cfg.num_asteroids, cfg.asteroids_split) == (3, 1, 7, False)
    assert cfg.max_frames == 90 * PHYSICS_FPS


def test_time_limit_none_removes_the_clock_and_keep_preserves_it():
    assert build_config(MatchSpec(scenario="team2", time_limit=None)).max_frames is None
    assert build_config(MatchSpec(scenario="team2", time_limit=KEEP)).max_frames == 1200
    assert build_config(MatchSpec(scenario="demo_team2")).max_frames is None


def test_scale_then_asteroid_count():
    cfg = build_config(MatchSpec(scenario="team2", scale=2.0, num_asteroids=3))
    assert cfg.width == 1600 and cfg.num_asteroids == 3


def test_shrinking_a_static_scenario_trims_spawns():
    cfg = build_config(MatchSpec(scenario="demo_team3", n_red=1, m_blue=2, num_asteroids=1))
    assert len(cfg.red_spawns) == 1 and len(cfg.blue_spawns) == 2
    assert len(cfg.asteroid_spawns) == 1 and cfg.num_asteroids == 1


def test_growing_a_team_adds_random_spawns():
    from marl_asteroids.env import TeamAsteroidsEnv
    cfg = build_config(MatchSpec(scenario="demo_duel", m_blue=3))
    env = TeamAsteroidsEnv(cfg)
    env.reset(seed=0)
    assert len([a for a in env.possible_agents if a.startswith("blue")]) == 3


def test_empty_team_is_rejected():
    with pytest.raises(ValueError):
        build_config(MatchSpec(scenario="duel", n_red=0))


def test_controllers_default_and_override():
    m = build_match(MatchSpec(scenario="team2", default_controller="dummy",
                              controllers={"blue_0": "heuristic_1.0"}))
    assert m.assignment == {"blue_0": "heuristic_1.0", "blue_1": "dummy",
                            "red_0": "dummy", "red_1": "dummy"}
    assert m.policies["blue_1"] is m.policies["red_0"]  # one instance per controller


def test_unknown_ship_and_two_humans_are_rejected():
    with pytest.raises(ValueError, match="not in this match"):
        build_match(MatchSpec(scenario="duel", controllers={"blue_5": "dummy"}))
    with pytest.raises(ValueError, match="only one"):
        build_match(MatchSpec(scenario="duel", controllers={"blue_0": HUMAN, "red_0": HUMAN}))


def test_handicaps_apply_only_when_a_human_plays():
    ai = build_match(MatchSpec(scenario="team2", difficulty="easy"))
    assert ai.delay == 0 and ai.fire_rate is None
    assert not any(isinstance(p, DelayedPolicy) for p in ai.policies.values())

    m = build_match(MatchSpec(scenario="team2", difficulty="easy",
                              controllers={"blue_0": HUMAN}))
    delay, fire_rate = DIFFICULTY["easy"]
    assert (m.delay, m.fire_rate) == (delay, fire_rate)
    assert isinstance(m.policies["blue_0"], HumanPolicy)
    assert isinstance(m.policies["blue_1"], DelayedPolicy)  # wingman: delayed too
    assert isinstance(m.policies["red_0"], DelayedPolicy)
    # Reload handicap only on the team without the human
    assert all(bp.fire_rate == fire_rate for bp in m.config.red_spawns)
    assert all(bp.fire_rate is None for bp in m.config.blue_spawns)


def test_difficulty_defaults():
    assert build_match(MatchSpec(scenario="duel", controllers={"blue_0": HUMAN})).difficulty == "normal"
    assert build_match(MatchSpec(scenario="duel")).difficulty == "brutal"
    with pytest.raises(ValueError):
        build_match(MatchSpec(scenario="duel", difficulty="impossible"))


def test_camera_follow_rules():
    assert build_match(MatchSpec(scenario="duel", controllers={"red_0": HUMAN})).follow == "red_0"
    assert build_match(MatchSpec(scenario="duel")).follow is None
    assert build_match(MatchSpec(scenario="duel", follow="red_0")).follow == "red_0"
    assert build_match(MatchSpec(scenario="duel", follow="none",
                                 controllers={"blue_0": HUMAN})).follow is None
    with pytest.raises(ValueError):
        build_match(MatchSpec(scenario="duel", follow="blue_3"))


def test_full_game_runs_and_reports_stats():
    m = build_match(MatchSpec(scenario="duel", time_limit=20.0, seed=1,
                              controllers={"blue_0": "heuristic_1.0", "red_0": "dummy"}))
    m.make_env()
    m.reset()
    while not m.step():
        pass
    stats = m.stats()
    assert stats.games == 1 and stats.blue_wins == 1
    m.close()


def test_seeded_matches_repeat():
    lengths = []
    for _ in range(2):
        m = build_match(MatchSpec(scenario="team2", seed=7, time_limit=15.0))
        m.make_env()
        m.reset()
        while not m.step():
            pass
        lengths.append(m.env.frame)
        m.close()
    assert lengths[0] == lengths[1]


def test_networks_are_loaded_once_and_shared():
    path = "models/phaseA/ladder_1M.zip"
    a = build_match(MatchSpec(scenario="duel", default_controller=path))
    b = build_match(MatchSpec(scenario="team2", default_controller=path))
    assert a.policies["blue_0"] is b.policies["red_1"]


def test_spec_round_trips_through_json_and_tolerates_old_keys():
    spec = MatchSpec(scenario="team3", controllers={"blue_0": HUMAN}, time_limit=None, scale=2.0)
    again = MatchSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec
    old = MatchSpec.from_dict({"scenario": "duel", "removed_option": 3})
    assert old.scenario == "duel"


def test_agent_names_order_blue_first():
    assert agent_names(get_scenario("team2")) == ["blue_0", "blue_1", "red_0", "red_1"]
