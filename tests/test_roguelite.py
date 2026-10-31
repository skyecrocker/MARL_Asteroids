"""Roguelite survival: pure logic (catalog, shop, economy, wave schedule, run state)."""
import dataclasses
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from marl_asteroids.config import ARENA_SIZES, SURVIVAL_MAX_ENEMIES
from marl_asteroids.roguelite import (DEFAULT_ROGUELITE, PurchaseError, RogueliteConfig, RunState,
                                      UpgradeDef, WaveBand, advance_wave, buy, can_buy,
                                      effective_stats, elite_count, hud_line, kill_credits,
                                      level_of, make_roguelite_config, on_wave_cleared, shop_rows,
                                      upgrade_cost, wave_clear_bonus, wave_composition, wave_size,
                                      wingman_tier)

CFG = DEFAULT_ROGUELITE
ITEMS = [u.id for u in CFG.catalog]
FIELDS = [f.name for f in dataclasses.fields(RunState)]

# ----------------------------------------------------------------------- strategies


@st.composite
def run_states(draw):
    """States reachable by playing: random buys, wave clears and wave advances from a new run."""
    state = RunState.new(CFG, seed=draw(st.integers(0, 2 ** 31 - 1)))
    for op in draw(st.lists(st.sampled_from(["buy", "clear", "next", "rich"]), max_size=40)):
        if op == "buy":
            item = draw(st.sampled_from(ITEMS))
            if can_buy(state, item, CFG)[0]:
                state = buy(state, item, CFG)
        elif op == "clear":
            state = on_wave_cleared(state, CFG)
        elif op == "next":
            state = advance_wave(state)
        else:
            state = dataclasses.replace(state, credits=state.credits + 300)
    return state


# --------------------------------------------------------------------------- examples


def test_new_run_starts_clean():
    s = RunState.new(CFG, seed=4)
    assert (s.wave, s.waves_cleared, s.credits, s.wingmen) == (1, 0, CFG.start_credits, 0)
    assert all(level_of(s, i) == 0 for i in ITEMS)
    assert s.seed == 4
    assert isinstance(RunState.new(CFG).seed, int)  # no seed given: a random one


def test_default_config_is_valid_and_uses_the_medium_arena():
    CFG.validate()
    assert CFG.arena == "medium"
    game = make_roguelite_config(CFG)
    assert game.width == game.height == ARENA_SIZES["medium"]
    assert game.m_blue == 4 and game.n_red == SURVIVAL_MAX_ENEMIES
    assert game.max_frames is None and not game.friendly_fire and game.mode == "roguelite"


@pytest.mark.parametrize("change", [
    dict(tiers=tuple(reversed(DEFAULT_ROGUELITE.tiers))),               # strongest first
    dict(payouts={**DEFAULT_ROGUELITE.payouts, "heavy": {"basic": 40, "veteran": 27, "elite": 18}}),
    dict(payouts={k: v for k, v in DEFAULT_ROGUELITE.payouts.items() if k != "sniper"}),
    dict(schedule=DEFAULT_ROGUELITE.schedule[1:]),                     # doesn't start at wave 1
    dict(schedule=(WaveBand(1, (("standard", 1), ("scout", 1), ("heavy", 1)), 0, 0),)),  # 3 > 2 ships
    dict(elite_majority_from=9),                                       # band at 9 is only 25% elite
    dict(arena="enormous"),
    dict(catalog=DEFAULT_ROGUELITE.catalog + (UpgradeDef("player.hull", "x", "player", (), 1, 1),)),
])
def test_validate_rejects_broken_tables(change):
    with pytest.raises(ValueError):
        dataclasses.replace(DEFAULT_ROGUELITE, **change).validate()


def test_variant_stat_relations():
    v = CFG.variants
    std = v["standard"].stats
    assert v["scout"].stats.max_speed > std.max_speed and v["scout"].stats.max_hull < std.max_hull
    heavy = v["heavy"].stats
    assert heavy.max_speed < std.max_speed
    assert heavy.max_hull + heavy.max_shield > std.max_hull + std.max_shield
    assert v["sniper"].stats.bullet_lifespan > std.bullet_lifespan
    assert v["sniper"].standoff > v["standard"].standoff


def test_tiers_and_payouts_are_monotone():
    tiers = CFG.tiers
    assert [t.name for t in tiers] == ["basic", "veteran", "elite"]
    for weak, strong in zip(tiers, tiers[1:]):
        assert strong.reaction_frames <= weak.reaction_frames and strong.p >= weak.p
    for variant, pays in CFG.payouts.items():
        assert pays["basic"] <= pays["veteran"] <= pays["elite"], variant


def test_wingman_upgrades_need_a_wingman():
    s = dataclasses.replace(RunState.new(CFG, seed=0), credits=10_000)
    assert can_buy(s, "wing.hull", CFG) == (False, "no wingmen")
    s = buy(s, "wing.buy", CFG)
    assert can_buy(s, "wing.hull", CFG) == (True, "")


def test_fleet_caps_at_three_wingmen_and_intelligence_at_elite():
    s = dataclasses.replace(RunState.new(CFG, seed=0), credits=100_000)
    for _ in range(3):
        s = buy(s, "wing.buy", CFG)
    assert s.wingmen == 3 and can_buy(s, "wing.buy", CFG) == (False, "maxed")
    assert wingman_tier(CFG, s).name == "basic"
    s = buy(buy(s, "wing.tier", CFG), "wing.tier", CFG)
    assert wingman_tier(CFG, s).name == "elite" and can_buy(s, "wing.tier", CFG)[1] == "maxed"
    rows = {r.id: r for r in shop_rows(s, CFG)}
    assert rows["wing.buy"].maxed and rows["wing.buy"].cost is None
    assert rows["wing.buy"].level_text == "3/3 owned" and rows["wing.tier"].level_text == "Elite"


def test_fire_rate_floor():
    s = dataclasses.replace(RunState.new(CFG, seed=0), credits=100_000)
    for _ in range(5):
        s = buy(s, "player.fire_rate", CFG)
    assert effective_stats(CFG, s, "player").fire_rate == 12  # 30 - 5 * 4 = 10, floored at 12


def test_kill_credits_examples():
    looks = {"red_0": ("heavy", "elite"), "red_1": ("scout", "basic")}
    events = [("blue_0", "red_0"), ("blue_2", "red_1"), ("ram", "red_1"), ("asteroid", "red_0"),
              ("red_0", "blue_1")]
    assert kill_credits(events, looks, CFG) == (40 + 12, 1, 2)


def test_early_waves_are_gentle():
    for k in (1, 2):
        assert all((e.variant, e.tier) == ("standard", "basic") for e in wave_composition(CFG, k, 0))
    assert wave_size(CFG, 1) == CFG.base_wave_size


# --------------------------------------------------------------------------- properties


@settings(max_examples=100)
@given(st.integers(1, 10_000), st.integers(0, 30))
def test_upgrade_cost_formula(base, level):
    """Feature: survival-roguelite, Property 1: Upgrade cost formula"""
    assert upgrade_cost(base, level) == round(base * 1.5 ** level)
    assert upgrade_cost(base, level + 1) >= upgrade_cost(base, level)


@settings(max_examples=100, deadline=None)
@given(run_states(), st.sampled_from(ITEMS))
def test_purchase_accounting(state, item):
    """Feature: survival-roguelite, Property 2: Purchase accounting and shop state"""
    u = CFG.item(item)
    row = {r.id: r for r in shop_rows(state, CFG)}[item]
    ok, _ = can_buy(state, item, CFG)
    lvl = level_of(state, item)
    cost = upgrade_cost(u.base_cost, lvl)
    expected = (lvl < u.max_level and state.credits >= cost
                and (u.target == "player" or item == "wing.buy" or state.wingmen > 0))
    assert ok == expected == row.enabled
    assert row.maxed == (lvl >= u.max_level)
    before = state.to_dict()
    if ok:
        after = buy(state, item, CFG)
        assert after.credits == state.credits - cost == state.credits - row.cost
        assert level_of(after, item) == lvl + 1
        a = after.to_dict()
        for name in FIELDS:
            if name in ("credits", "upgrades", "wingmen"):
                continue
            assert a[name] == before[name]
        for other in ITEMS:
            if other != item:
                assert level_of(after, other) == level_of(state, other)
    else:
        with pytest.raises(PurchaseError):
            buy(state, item, CFG)
        assert state.to_dict() == before
    assert state.credits >= 0 and 0 <= state.wingmen <= CFG.max_fleet - 1


@settings(max_examples=100, deadline=None)
@given(run_states(), st.sampled_from(ITEMS))
def test_effective_stats_follow_the_catalog_and_stay_isolated(state, item):
    """Feature: survival-roguelite, Property 3: Effective stats follow the catalog and stay isolated"""
    for target, base in (("player", CFG.player_base), ("wingmen", CFG.wingman_base)):
        expected = dataclasses.asdict(base)
        floors = {}
        for u in CFG.catalog:
            if u.target == target:
                for name, delta in u.effects:
                    expected[name] += delta * level_of(state, u.id)
                floors.update(dict(u.floors))
        for name, low in floors.items():
            expected[name] = max(expected[name], low)
        got = dataclasses.asdict(effective_stats(CFG, state, target))
        for name, value in expected.items():
            assert got[name] == pytest.approx(value)
    bumped = dataclasses.replace(state, upgrades={**state.upgrades,
                                                  item: state.upgrades.get(item, 0) + 1})
    untouched = "wingmen" if CFG.item(item).target == "player" else "player"
    assert effective_stats(CFG, bumped, untouched) == effective_stats(CFG, state, untouched)


@settings(max_examples=100, deadline=None)
@given(run_states())
def test_wave_clear_economy(state):
    """Feature: survival-roguelite, Property 6: Wave clear economy"""
    cleared = on_wave_cleared(state, CFG)
    bonus = wave_clear_bonus(CFG, state.wave)
    assert cleared.waves_cleared == state.waves_cleared + 1
    assert cleared.credits == state.credits + bonus
    assert cleared.credits_earned == state.credits_earned + bonus
    assert dataclasses.replace(cleared, waves_cleared=state.waves_cleared, credits=state.credits,
                               credits_earned=state.credits_earned) == state
    nxt = advance_wave(state)
    assert nxt.wave == state.wave + 1 and nxt.credits == state.credits


agents = st.sampled_from([f"blue_{i}" for i in range(4)] + [f"red_{i}" for i in range(6)]
                         + ["ram", "asteroid"])


@settings(max_examples=100)
@given(st.lists(st.tuples(agents, agents), max_size=30),
       st.dictionaries(st.sampled_from([f"red_{i}" for i in range(6)]),
                       st.tuples(st.sampled_from(list(CFG.variants)),
                                 st.sampled_from([t.name for t in CFG.tiers]))))
def test_kill_credit_attribution(events, looks):
    """Feature: survival-roguelite, Property 7: Kill credit attribution"""
    credits, mine, fleet = kill_credits(events, looks, CFG)
    paid = [(s, v) for s, v in events if s.startswith("blue_") and v.startswith("red_")]
    assert credits == sum(CFG.payouts[looks.get(v, ("standard", "basic"))[0]]
                          [looks.get(v, ("standard", "basic"))[1]] for _, v in paid)
    assert mine == sum(1 for s, _ in paid if s == "blue_0")
    assert fleet == len(paid)


@settings(max_examples=100)
@given(st.integers(1, 200))
def test_wave_size_formula(k):
    """Feature: survival-roguelite, Property 9: Wave size formula"""
    assert wave_size(CFG, k) == min(20, CFG.base_wave_size + (k - 1) // 2)
    assert wave_size(CFG, k + 1) >= wave_size(CFG, k)


@settings(max_examples=100)
@given(st.integers(1, 60), st.integers(0, 2 ** 31 - 1))
def test_composition_follows_the_bands(k, seed):
    """Feature: survival-roguelite, Property 10: Composition follows the bands"""
    comp = wave_composition(CFG, k, seed)
    assert len(comp) == wave_size(CFG, k)
    variants = [e.variant for e in comp]
    tiers = [e.tier for e in comp]
    if k < 3:
        assert set(variants) == {"standard"} and set(tiers) == {"basic"}
    if k < 6:
        assert "sniper" not in variants and set(tiers) == {"basic"}
    if k >= 3:
        assert "scout" in variants and "heavy" in variants
    if k >= 6:
        assert "sniper" in variants and "veteran" in tiers
    if k >= CFG.elite_majority_from:
        assert tiers.count("elite") * 2 > len(tiers)


@settings(max_examples=100)
@given(st.integers(1, 60), st.integers(0, 2 ** 31 - 1))
def test_elite_share_never_decreases(k, seed):
    """Feature: survival-roguelite, Property 11: Elite share never decreases"""
    def share(w):
        return sum(e.tier == "elite" for e in wave_composition(CFG, w, seed)) / wave_size(CFG, w)
    assert share(k + 1) >= share(k)
    assert elite_count(CFG, k) == round(share(k) * wave_size(CFG, k))


@settings(max_examples=100)
@given(st.integers(1, 60), st.integers(0, 2 ** 31 - 1))
def test_composition_is_deterministic(k, seed):
    """Feature: survival-roguelite, Property 12: Composition is deterministic"""
    assert wave_composition(CFG, k, seed) == wave_composition(RogueliteConfig(), k, seed)


@settings(max_examples=100, deadline=None)
@given(run_states())
def test_run_state_is_plain_data(state):
    """Feature: survival-roguelite, Property 14: RunState is plain data"""
    assert RunState.from_dict(json.loads(json.dumps(state.to_dict()))) == state
    assert RunState.from_dict({**state.to_dict(), "from_the_future": 1}) == state


@settings(max_examples=100, deadline=None)
@given(run_states(), st.integers(0, 20))
def test_hud_line_shows_wave_enemies_and_credits(state, left):
    """Feature: survival-roguelite, Property 15: HUD line shows wave, enemies, and credits"""
    line = hud_line(state, left)
    assert str(state.wave) in line and str(left) in line and str(state.credits) in line
