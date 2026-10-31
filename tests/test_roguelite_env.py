"""Roguelite survival in the env: per-ship stats, wave setup, run end, and a headless run."""
import dataclasses
import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from marl_asteroids.app.screens import MODE_OPTIONS
from marl_asteroids.config import MODES, GameConfig, make_game
from marl_asteroids.entities import Loadout, Ship, ShipStats
from marl_asteroids.env import BLUE, PLAYER, RED, TeamAsteroidsEnv, obs_dim
from marl_asteroids.policies import NOOP, Policy, ReactionDelayPolicy, make_scripted
from marl_asteroids.render import SHAPES, TEAM_COLORS, ship_look
from marl_asteroids.roguelite import (DEFAULT_ROGUELITE, RogueliteRun, RunState, effective_stats,
                                      wave_size, wingman_tier)

CFG = DEFAULT_ROGUELITE
WING_ITEMS = ["wing.tier", "wing.hull", "wing.shield", "wing.fire_rate", "wing.engine"]
PLAYER_ITEMS = ["player.hull", "player.shield", "player.fire_rate", "player.engine"]


def bot_run(cfg=CFG, state=None, seed=0):
    return RogueliteRun(cfg, state=state, player_policy=make_scripted("heuristic_1.0", seed=1),
                        seed=seed)


# --------------------------------------------------------------------- per-ship stats


@settings(max_examples=100)
@given(st.integers(1, 9), st.integers(0, 5), st.integers(5, 120), st.floats(0.01, 1.0),
       st.floats(0.5, 10.0), st.integers(10, 300), st.one_of(st.none(), st.integers(5, 120)))
def test_ship_stats_fall_back_to_config_and_round_trip(hull, shield, fire, thrust, speed, life, bp):
    """Feature: survival-roguelite, Property 13: Per-ship stats fall back to the config and round-trip"""
    cfg = GameConfig(max_hull=hull, max_shield=shield, fire_rate=fire, thrust_power=thrust,
                     max_speed=speed, bullet_lifespan=life)
    ship = Ship(0, 0, BLUE, "blue_0", cfg, fire_rate=bp)
    assert ship.stats == ShipStats.from_config(cfg, bp)
    assert (ship.hull, ship.shield) == (hull, shield)
    custom = ShipStats(hull + 1, shield, fire + 1, thrust * 2, speed + 1, life + 3)
    assert Ship(0, 0, BLUE, "blue_0", cfg, stats=custom).stats == custom


def test_ship_stats_read_access_is_plain_numbers():
    env = TeamAsteroidsEnv(make_game("survival", "small"))
    env.reset(seed=0)
    stats = env.ship_stats("blue_0")
    assert all(type(v) in (int, float) for v in stats.values())
    assert stats["bullet_range"] == env.config.bullet_speed * env.config.bullet_lifespan


@pytest.mark.parametrize("mode", list(MODES))
def test_existing_modes_keep_config_stats_and_221_obs(mode):
    env = TeamAsteroidsEnv(make_game(mode, "small"))
    obs, _ = env.reset(seed=1)
    cfg = env.config
    assert obs_dim(cfg) == 221 and all(o.shape == (221,) for o in obs.values())
    for ship in env.ships.values():
        assert ship.thrust_power == cfg.thrust_power and ship.max_speed == cfg.max_speed
        assert ship.bullet_lifespan == cfg.bullet_lifespan and ship.variant is None


def test_roguelite_is_not_a_setup_mode():
    assert "roguelite" not in [m for _, m in MODE_OPTIONS] and "roguelite" not in MODES
    with pytest.raises(ValueError):
        make_game("roguelite", "medium")


def test_start_wave_only_in_roguelite():
    env = TeamAsteroidsEnv(make_game("survival", "small"))
    env.reset(seed=0)
    lo = Loadout(ShipStats.from_config(env.config))
    with pytest.raises(ValueError):
        env.start_wave(1, [lo], {"blue_0": lo})


def test_start_wave_rejects_too_many_enemies():
    run = bot_run()
    run.start()
    lo = Loadout(ShipStats.from_config(run.game_config), "standard", "basic")
    with pytest.raises(ValueError):
        run.env.start_wave(2, [lo] * 21, {PLAYER: lo})


def test_reaction_delay_policy_replays_old_actions():
    class Counter(Policy):
        name = "counter"
        def __init__(self):
            self.n = 0
        def act(self, obs, contexts):
            self.n += 1
            return np.array([[1, self.n % 3, 0]] * len(contexts))

    class FakeEnv:
        decision_step = 0

    env = FakeEnv()
    pol = ReactionDelayPolicy(Counter(), delay=2)
    out = []
    for step in range(5):
        env.decision_step = step
        out.append(pol.act(np.zeros((1, 3)), [(env, "a")])[0].tolist())
    assert out[0] == NOOP.tolist() and out[1] == NOOP.tolist()
    assert out[2] == [1, 1, 0] and out[3] == [1, 2, 0]
    env.decision_step = 9  # a gap (the ship was down): start over, no stale actions
    assert pol.act(np.zeros((1, 3)), [(env, "a")])[0].tolist() == NOOP.tolist()


# ------------------------------------------------------------------------ wave setup


@st.composite
def fleet_states(draw):
    state = RunState.new(CFG, seed=draw(st.integers(0, 10_000)))
    state.wave = draw(st.integers(1, 30))
    state.wingmen = draw(st.integers(0, 3))
    for item in WING_ITEMS + PLAYER_ITEMS:
        state.upgrades[item] = draw(st.integers(0, CFG.item(item).max_level))
    return state


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(fleet_states(), st.data())
def test_wave_start_places_and_restores_the_fleet(state, data):
    """Feature: survival-roguelite, Property 4: Wave start places and restores the fleet"""
    run = bot_run(state=state)
    run.start()
    env = run.env
    for _ in range(data.draw(st.integers(0, 20))):
        run.step()
        if run.phase != "wave":
            return
    player = env.ships[PLAYER]
    if player.hp > 1:
        player.protection = 0
        player.apply_damage(1)  # some prior damage to repair
    for i in range(1, state.wingmen + 1):
        if data.draw(st.booleans()):
            env.ships[f"blue_{i}"].protection = 0
            env.ships[f"blue_{i}"].apply_damage(100)
    run._start_wave()
    blues = [a for a in env.team_agents[BLUE] if env.ships[a].active]
    assert blues == [f"blue_{i}" for i in range(state.wingmen + 1)]
    for agent in blues:
        ship = env.ships[agent]
        role = "player" if agent == PLAYER else "wingmen"
        assert ship.stats == effective_stats(CFG, run.state, role)
        assert (ship.hull, ship.shield) == (ship.max_hull, ship.max_shield)
        if agent != PLAYER:
            assert ship.tier == wingman_tier(CFG, run.state).name
            pol = run.policies[agent]
            assert pol.inner.kwargs["p"] == wingman_tier(CFG, run.state).p
    for agent in env.team_agents[BLUE][state.wingmen + 1:]:
        assert env.ships[agent].reserve and not env.ships[agent].active
    reds = [a for a in env.team_agents[RED] if env.ships[a].active]
    assert len(reds) == wave_size(CFG, run.state.wave)
    assert all(env.ships[a].reserve for a in env.team_agents[RED] if a not in reds)
    assert player.active


def test_respawned_wingmen_form_up_on_the_player():
    state = RunState.new(CFG, seed=2)
    state.wingmen = 3
    run = bot_run(state=state)
    run.start()
    env = run.env
    for i in (1, 2, 3):
        ship = env.ships[f"blue_{i}"]
        ship.protection = 0
        ship.pos[:] = (10.0, 10.0)  # far from the player, so placement is what's being tested
        ship.apply_damage(100)
        assert not ship.active
    run._start_wave()
    d = 3 * env.config.ship_radius + 20
    for i in (1, 2, 3):
        assert env.ships[f"blue_{i}"].active
        assert env._wrapped_dist(env.ships[f"blue_{i}"].pos, env.ships[PLAYER].pos) <= d + 1e-6


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 10_000), st.integers(0, 3), st.integers(1, 40))
def test_enemies_spawn_far_from_blue(seed, wingmen, wave):
    """Feature: survival-roguelite, Property 5: Enemies spawn far from blue"""
    state = RunState.new(CFG, seed=seed)
    state.wingmen, state.wave = wingmen, wave
    run = bot_run(state=state)
    run.start()
    env = run.env
    min_d = env.config.wave_spawn_distance * min(env.width, env.height)
    blues = [s for a, s in env.ships.items() if a.startswith("blue") and s.active]
    for a in env.team_agents[RED]:
        ship = env.ships[a]
        if ship.active:
            assert all(env._wrapped_dist(ship.pos, b.pos) >= min_d - 1e-6 for b in blues)


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 3), st.integers(0, 10_000), st.data())
def test_run_ends_exactly_when_the_player_dies(wingmen, seed, data):
    """Feature: survival-roguelite, Property 8: The run ends exactly when the player dies"""
    state = RunState.new(CFG, seed=seed)
    state.wingmen, state.wave = wingmen, 8
    run = bot_run(state=state)
    run.start()
    env = run.env
    for i in range(1, wingmen + 1):  # losing wingmen never ends the run
        if data.draw(st.booleans()):
            env.ships[f"blue_{i}"].protection = 0
            env.ships[f"blue_{i}"].apply_damage(100)
    env.ships[PLAYER].protection = 0
    assert run.step() == "wave"
    env.ships[PLAYER].protection = 0
    env.ships[PLAYER].apply_damage(100)
    assert run.step() == "over" and env.game_over and env.winner == RED


# ------------------------------------------------------------------------ headless run


def test_headless_run_through_the_shop():
    tough = dataclasses.replace(CFG, player_base=CFG.player_base.replace(max_hull=80))
    run = bot_run(tough, seed=3)
    run.start()
    env = run.env
    assert run.phase == "wave" and env.wave == 1
    for _ in range(20_000):
        if run.step() != "wave":
            break
    assert run.phase == "shop" and run.state.waves_cleared == 1
    assert run.state.credits > tough.start_credits  # kills and the clear bonus paid out
    with pytest.raises(RuntimeError):
        run.step()  # nothing moves while shopping
    run.state = dataclasses.replace(run.state, credits=1000)
    run.buy("wing.buy")
    run.buy("player.hull")
    run.next_wave()
    assert run.phase == "wave" and env.wave == 2 and run.env is env
    assert env.ships["blue_1"].active and env.ships["blue_1"].variant == "wingman"
    assert env.ships[PLAYER].max_hull == 81
    for _ in range(30):
        run.step()
    assert run.summary()["wave"] == 2
    run.close()


def test_kills_pay_credits_in_play():
    run = bot_run(seed=1)
    run.start()
    env = run.env
    red = env.ships["red_0"]
    red.protection = 0
    before = run.state.credits
    env.kill_log.append(("blue_0", "red_0"))  # as if shot down this step
    red.apply_damage(100)
    run.step()
    variant, tier = run.looks["red_0"]
    assert run.state.credits >= before + CFG.payouts[variant][tier]
    assert run.state.player_kills == 1


# ---------------------------------------------------------------------------- looks


class FakeShip:
    def __init__(self, team, variant, tier):
        self.team_id, self.variant, self.tier = team, variant, tier


def test_every_look_is_distinct_and_team_colored():
    """Feature: survival-roguelite, Property 17: Every look is distinct and team-colored"""
    variants = ["standard", "scout", "heavy", "sniper"]
    tiers = ["basic", "veteran", "elite"]
    shapes = {v: ship_look(FakeShip(RED, v, "basic")).shape for v in variants}
    assert len(set(shapes.values())) == len(variants)
    for v in variants:
        looks = [ship_look(FakeShip(RED, v, t)) for t in tiers]
        for a, b in itertools.combinations(looks, 2):
            assert (a.body, a.pips, a.ring) != (b.body, b.pips, b.ring)
        for look in looks:
            r, g, b = look.body
            assert r > g and r > b
    wing = ship_look(FakeShip(BLUE, "wingman", "basic"))
    player = ship_look(FakeShip(BLUE, None, None))
    assert wing.shape != player.shape and wing.body != player.body
    for look in (wing, player):
        r, g, b = look.body
        assert b > r and b >= g
    assert player.shape == SHAPES["standard"] and player.body == TEAM_COLORS[BLUE]["hull"]


def test_roguelite_renders_every_variant_tier_and_wingman():
    pytest.importorskip("pygame")
    state = RunState.new(CFG, seed=3)
    state.wave, state.wingmen = 14, 3
    run = bot_run(state=state)
    run.make_env(render_mode="rgb_array", follow=PLAYER, view_size=1000.0,
                 hud_text=lambda: run.hud_text())
    run.start()
    env = run.env
    # Force one of everything into view
    combos = list(itertools.product(["standard", "scout", "heavy", "sniper"],
                                    ["basic", "veteran", "elite"]))
    p = env.ships[PLAYER].pos
    for k, (v, t) in enumerate(combos):
        ship = env.ships[f"red_{k}"]
        ship.active, ship.reserve, ship.variant, ship.tier = True, False, v, t
        ship.pos[:] = ((p[0] + 60 * (k % 6) - 150) % env.width, (p[1] + 200 + 60 * (k // 6)) % env.height)
    frame = env.render()
    assert frame.shape[2] == 3 and frame.std() > 0
    for _ in range(3):
        run.step()
        env.render()
    run.close()
