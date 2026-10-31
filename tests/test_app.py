"""Desktop app: settings persistence, setup rules, and a headless run through every screen."""
import json

import pygame
import pytest

from marl_asteroids.app import screens as S
from marl_asteroids.app.main import App
from marl_asteroids.app.settings import WINDOW_SIZES, AppSettings
from marl_asteroids.session import HUMAN, MatchSpec


@pytest.fixture
def app(tmp_path):
    a = App(settings_path=tmp_path / "app.json")
    a.settings.attract_mode = False  # keep tests fast and independent of the background match
    a.push(S.MainMenuScreen(a))
    yield a
    while a.stack:
        a.stack.pop().exit()
    pygame.quit()


def run(app, n):
    for _ in range(n):
        app.step()


def press(app, text):
    for element, callback in app.screen._buttons.items():
        if element.text == text:
            callback()
            return
    raise KeyError(f"no button {text!r} on {type(app.screen).__name__}")


def texts_on(screen):
    """Text of every label/button on a screen."""
    return {el.text for el in screen.root.get_container().elements if hasattr(el, "text")}


def run_until(app, screen_type, limit=20000):
    for _ in range(limit):
        if isinstance(app.screen, screen_type):
            return
        app.step()
    raise AssertionError(f"never reached {screen_type.__name__}; on {type(app.screen).__name__}")


# --------------------------------------------------------------------------- settings


def test_settings_defaults_when_missing_or_corrupt(tmp_path):
    assert AppSettings.load(tmp_path / "nope.json") == AppSettings()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert AppSettings.load(bad) == AppSettings()
    bad.write_text("[1, 2, 3]")
    assert AppSettings.load(bad) == AppSettings()


def test_settings_round_trip_and_tolerance(tmp_path):
    p = tmp_path / "s.json"
    s = AppSettings(window=[1600, 1000], effects=False, last_setup={"mode": "siege"})
    s.save(p)
    assert AppSettings.load(p) == s
    data = json.loads(p.read_text())
    data["window"] = [5, 5]       # nonsense window size is reset
    data["future_option"] = True  # unknown keys are ignored
    p.write_text(json.dumps(data))
    loaded = AppSettings.load(p)
    assert loaded.window == list(WINDOW_SIZES[0]) and loaded.effects is False


def test_settings_save_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "app.json"
    AppSettings().save(p)
    assert p.exists() and not p.with_suffix(".tmp").exists()


def test_config_path_env_override(tmp_path, monkeypatch):
    from marl_asteroids.app.settings import config_path
    monkeypatch.setenv("MARL_ASTEROIDS_CONFIG", str(tmp_path / "x.json"))
    assert config_path() == tmp_path / "x.json"


# ----------------------------------------------------------------------------- setup


def test_sanitize_replaces_missing_models_and_bad_settings():
    spec = MatchSpec(mode="deathmatch", arena="small", m_blue=9, n_red=1,
                     controllers={"blue_0": "models/gone.zip", "red_0": "dummy"},
                     default_controller="models/also_gone.zip", difficulty="nightmare")
    clean = S.sanitize(spec)
    assert S.valid_controller(clean.default_controller)
    assert clean.controllers["blue_0"] == clean.default_controller
    assert clean.controllers["red_0"] == "dummy"
    assert clean.difficulty == "normal"
    assert clean.m_blue == 2 and clean.n_red == 1  # out-of-range team size reset
    # Unknown mode, unknown arena, or an old scenario-only setup: start from the defaults
    for bad in (MatchSpec(mode="capture_the_moon", arena="small"),
                MatchSpec(mode="siege", arena="gigantic"),
                MatchSpec(scenario="demo_team2")):
        assert S.sanitize(bad) == S.default_spec()


def test_defaults_put_you_in_blue_1_in_a_deathmatch(app):
    setup = S.SetupScreen(app)
    assert setup.spec.controllers["blue_0"] == HUMAN
    assert setup.spec.mode == "deathmatch" and setup.spec.arena == "small"


def test_main_menu_has_no_watch_button(app):
    run(app, 1)
    texts = {el.text for el in app.screen._buttons}
    assert texts == {"Play", "Gallery", "Settings", "Quit"}


def test_switching_to_an_objective_mode_swaps_trained_agents_for_bots(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    setup.spec.controllers = {"blue_0": HUMAN, "blue_1": "models/phaseA/selfplay_final.zip",
                              "red_0": "models/phaseA/baseline.zip", "red_1": "dummy"}
    setup._set_mode("siege")
    assert setup.spec.controllers == {"blue_0": HUMAN, "blue_1": S.BOT, "red_0": S.BOT,
                                      "red_1": "dummy"}
    assert "Trained agents" in setup.status
    assert setup.spec.time_limit == S.KEEP  # mode default clock
    # The trained agents can still be picked deliberately
    setup._set_controller("red_0", "models/phaseA/baseline.zip")
    assert setup.spec.controllers["red_0"] == "models/phaseA/baseline.zip"


def test_spawns_option_only_in_combat_modes(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    for mode, shown in (("deathmatch", True), ("sudden_death", True), ("siege", False),
                        ("survival", False)):
        setup._set_mode(mode)
        assert ("Spawns" in texts_on(setup)) is shown, mode
    assert "Max per wave" in texts_on(setup)


def test_trained_agents_stay_in_both_combat_modes(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    agent = "models/phaseA/selfplay_final.zip"
    setup.spec.controllers["red_0"] = agent
    for mode in ("sudden_death", "deathmatch"):
        setup._set_mode(mode)
        assert setup.spec.controllers["red_0"] == agent and not setup.status


def test_deathmatch_always_has_a_clock(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    time_choice = [c for c in setup._choices.values() if c.options[0][1] == S.KEEP][0]
    assert None not in [v for _, v in time_choice.options]  # no "No limit" for deathmatch
    setup._set_mode("sudden_death")
    time_choice = [c for c in setup._choices.values() if c.options[0][1] == S.KEEP][0]
    assert None in [v for _, v in time_choice.options]
    saved = MatchSpec(mode="deathmatch", arena="small", time_limit=None)
    assert S.sanitize(saved).time_limit == S.KEEP


def test_arena_change_resets_asteroid_count(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    setup._set_asteroids(17)
    setup._set_arena("large")
    assert setup.spec.num_asteroids is None and setup._cfg().width == 4800


def test_only_one_human_ship(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    setup._set_controller("red_0", HUMAN)
    humans = [a for a, c in setup.spec.controllers.items() if c == HUMAN]
    assert humans == ["red_0"]


def test_team_size_change_keeps_existing_choices(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    setup.spec.controllers["blue_0"] = "dummy"
    setup._set("m_blue", 4, rebuild=True)
    assert list(a for a in setup.spec.controllers if a.startswith("blue")) == \
        ["blue_0", "blue_1", "blue_2", "blue_3"]
    assert setup.spec.controllers["blue_0"] == "dummy"
    setup._set("m_blue", 1, rebuild=True)
    assert [a for a in setup.spec.controllers if a.startswith("blue")] == ["blue_0"]


def test_mode_change_keeps_team_sizes(app):
    setup = S.SetupScreen(app)
    app.push(setup)
    setup._set("n_red", 3, rebuild=True)
    setup._set_mode("survival")
    assert setup.spec.n_red == 3 and set(setup.spec.controllers) == {
        "blue_0", "blue_1", "red_0", "red_1", "red_2"}


def test_bad_setup_reports_instead_of_crashing(app, monkeypatch):
    setup = S.SetupScreen(app)
    app.push(setup)
    monkeypatch.setattr(S, "build_match", lambda spec: (_ for _ in ()).throw(ValueError("boom")))
    setup.start()
    run(app, 4)
    assert app.screen is setup and setup.status_is_error and "boom" in setup.status


# ------------------------------------------------------------------------- full flow


def quick_watch(app, mode="deathmatch"):
    """Bots only and a short clock, so games end fast and deterministically."""
    setup = S.SetupScreen(app)
    setup.spec = MatchSpec(mode=mode, arena="small", m_blue=1, n_red=1, time_limit=8.0, seed=3,
                           controllers={"blue_0": "heuristic_1.0", "red_0": "dummy"})
    app.push(setup)
    return setup


def test_full_flow_through_every_screen(app):
    run(app, 3)
    press(app, "Settings"); run(app, 2)
    assert isinstance(app.screen, S.SettingsScreen)
    press(app, "Back"); run(app, 2)
    press(app, "Gallery"); run(app, 2)
    assert isinstance(app.screen, S.GalleryScreen)
    press(app, "Back"); run(app, 2)

    setup = quick_watch(app)
    press(app, "Start")
    run_until(app, S.GameScreen, limit=10)
    game = app.screen
    assert game.env is not None and game.env.on_frame is not None
    run_until(app, S.ResultsScreen)
    results = app.screen
    assert results.winner == 1 and results.stats.games == 1  # heuristic beats a dummy
    assert app.settings.last_setup["arena"] == "small"       # setup remembered

    press(app, "Play again")
    assert isinstance(app.screen, S.GameScreen) and app.screen.game_index == 1
    run_until(app, S.ResultsScreen)
    press(app, "Change setup")
    assert app.screen is setup
    press(app, "Back")
    assert isinstance(app.screen, S.MainMenuScreen) and len(app.stack) == 1


def test_game_draws_every_physics_frame(app):
    quick_watch(app)
    press(app, "Start")
    run_until(app, S.GameScreen, limit=10)
    game = app.screen
    before_frames, before_physics = app.frames, game.env.frame
    run(app, 20)  # 20 decisions
    drawn = app.frames - before_frames
    advanced = game.env.frame - before_physics
    assert advanced > 0 and drawn >= advanced  # at least one presented frame per physics frame


def test_pause_freezes_the_game_and_resume_continues(app):
    quick_watch(app)
    press(app, "Start")
    run_until(app, S.GameScreen, limit=10)
    game = app.screen
    run(app, 5)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    run(app, 1)
    assert game.paused
    frozen = game.env.frame
    run(app, 30)
    assert game.env.frame == frozen
    press(app, "Resume")
    run(app, 5)
    assert not game.paused and game.env.frame > frozen


def test_restart_and_main_menu_from_pause(app):
    quick_watch(app)
    press(app, "Start")
    run_until(app, S.GameScreen, limit=10)
    game = app.screen
    run(app, 10)
    game.pause()
    press(app, "Restart")
    assert not game.paused and game.env.frame == 0 and game.game_index == 1
    game.pause()
    press(app, "Main menu")
    assert isinstance(app.screen, S.MainMenuScreen) and len(app.stack) == 1
    assert game.env is None  # match closed on the way out


def test_tab_toggles_sensor_view(app):
    quick_watch(app)
    press(app, "Start")
    run_until(app, S.GameScreen, limit=10)
    game = app.screen
    start = game.renderer.show_obs
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
    run(app, 1)
    assert game.renderer.show_obs is (not start)


def test_human_ship_idles_without_input_and_game_still_ends(app):
    setup = S.SetupScreen(app)
    setup.spec = MatchSpec(mode="deathmatch", arena="small", m_blue=1, n_red=1, time_limit=5.0,
                           seed=1, difficulty="brutal",
                           controllers={"blue_0": HUMAN, "red_0": "dummy"})
    app.push(setup)
    press(app, "Start")
    run_until(app, S.ResultsScreen)
    assert app.screen.human == "blue_0"


def test_window_resize_rebuilds_current_screen(app):
    app.set_window_size((1440, 900))
    assert app.window.get_size() == (1440, 900) and app.settings.window == [1440, 900]
    run(app, 2)
    assert isinstance(app.screen, S.MainMenuScreen) and app.screen.root.rect.size == (1440, 900)


def test_escape_on_setup_goes_back(app):
    quick_watch(app)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    run(app, 1)
    assert isinstance(app.screen, S.MainMenuScreen)


def test_attract_mode_background_runs(tmp_path):
    a = App(settings_path=tmp_path / "app.json")
    a.settings.attract_mode = True
    a.push(S.MainMenuScreen(a))
    for _ in range(40):
        a.step()
    assert a._background is not None and a._background.env.frame > 0
    while a.stack:
        a.stack.pop().exit()
    pygame.quit()


@pytest.mark.parametrize("mode", ["siege", "survival"])
def test_objective_modes_play_through_to_results(app, mode):
    setup = quick_watch(app, mode)
    setup.spec.controllers = {"blue_0": "heuristic_1.0", "red_0": "heuristic_1.0"}
    press(app, "Start")
    run_until(app, S.ResultsScreen)
    results = app.screen
    assert results.summary["mode"] == mode
    texts = texts_on(results)
    if mode == "survival":
        assert any(t.startswith("Survived") for t in texts)
    else:
        assert "Station damage" in texts
