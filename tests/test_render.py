"""Renderer tests: headless rgb_array output, follow camera, toroidal projection."""
import numpy as np
import pytest

from marl_asteroids.config import get_scenario
from marl_asteroids.env import TeamAsteroidsEnv


def _env(scenario="team3_huge", **kw):
    env = TeamAsteroidsEnv(get_scenario(scenario), render_mode="rgb_array", **kw)
    env.reset(seed=0)
    return env


def test_whole_map_render_shape():
    env = _env("team2")
    frame = env.render()
    assert frame.shape == (1000, 1000, 3) and frame.dtype == np.uint8
    env.close()


def test_follow_render_shape_and_window():
    env = _env(follow="blue_0", view_size=1200.0, window=800)
    frame = env.render()
    assert frame.shape == (800, 800, 3)
    env.close()


def test_view_size_clamped_to_map():
    env = _env("duel", follow="blue_0", view_size=99999.0)  # 800x800 map
    env.render()
    assert env.renderer.view_size == max(env.config.width, env.config.height)
    env.close()


def test_followed_ship_projects_to_screen_center():
    env = _env(follow="blue_0", view_size=1000.0, window=600)
    env.render()
    r = env.renderer
    for pos in [(10.0, 10.0), (1600.0, 800.0), (env.width - 1, env.height - 1)]:
        env.ships["blue_0"].pos[:] = pos
        cam = r._camera_center()
        to_px = r._projector(cam, (600, 600), (0.0, 0.0))
        assert to_px(env.ships["blue_0"].pos) == (300, 300)
    env.close()


def test_projection_wraps_across_the_torus_seam():
    """A ship just across the map edge must appear adjacent, not a whole map away."""
    env = _env(follow="blue_0", view_size=1200.0, window=600)
    env.render()
    r = env.renderer
    env.ships["blue_0"].pos[:] = (5.0, 300.0)
    env.ships["red_0"].pos[:] = (env.width - 5.0, 300.0)  # 10 px away across the seam
    cam = r._camera_center()
    to_px = r._projector(cam, (600, 600), (0.0, 0.0))
    px = to_px(env.ships["red_0"].pos)
    assert abs(px[0] - 300) < 20 and px[1] == 300
    assert r._visible(env.ships["red_0"].pos, cam)
    env.close()


def test_visibility_culling_uses_wrapped_distance():
    env = _env(follow="blue_0", view_size=1000.0)
    env.render()
    r = env.renderer
    env.ships["blue_0"].pos[:] = (100.0, 100.0)
    cam = r._camera_center()
    assert r._visible(np.array([400.0, 100.0]), cam)             # inside the viewport
    assert not r._visible(np.array([1500.0, 100.0]), cam)        # well outside
    assert r._visible(np.array([env.width - 50.0, 100.0]), cam)  # just across the seam
    env.close()


def test_whole_map_mode_shows_everything():
    env = _env("team3_huge")
    env.render()
    r = env.renderer
    assert r.follow is None
    cam = r._camera_center()
    assert r._visible(np.array([0.0, 0.0]), cam)
    assert r._visible(np.array([env.width - 1.0, env.height - 1.0]), cam)
    env.close()


def test_camera_holds_position_after_followed_ship_dies():
    env = _env(follow="blue_0", view_size=1000.0)
    env.render()
    r = env.renderer
    env.ships["blue_0"].pos[:] = (700.0, 900.0)
    last = r._camera_center().copy()
    env.ships["blue_0"].active = False
    np.testing.assert_allclose(r._camera_center(), last)
    env.render()  # must not crash with the followed ship gone
    env.close()


def test_render_runs_over_a_full_episode_with_effects():
    """Exercises particles, trails, deaths, splitting rocks and the minimap together."""
    env = TeamAsteroidsEnv(get_scenario("asteroid_storm"), render_mode="rgb_array",
                           follow="blue_0", view_size=900.0, window=400)
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(300):
        if not env.agents:
            env.reset(seed=2)
        env.step({a: rng.integers([2, 3, 2]) for a in env.agents})
        assert env.render().shape == (400, 400, 3)
    env.close()


def test_renderer_is_not_created_without_render_mode():
    env = TeamAsteroidsEnv(get_scenario("duel"))
    env.reset(seed=0)
    assert env.render() is None and env.renderer is None


def test_surface_mode_returns_a_surface_and_never_owns_the_display():
    import pygame

    pygame.display.init()
    window = pygame.display.set_mode((320, 320))  # stands in for an app-owned window
    env = TeamAsteroidsEnv(get_scenario("duel"), render_mode="surface", window=300)
    env.reset(seed=0)
    surf = env.render()
    assert isinstance(surf, pygame.Surface) and surf.get_size() == (300, 300)
    assert env.renderer.screen is None  # did not create its own window
    env.close()
    # The app's window must survive the env closing
    assert pygame.display.get_init() and pygame.display.get_surface() is window
    pygame.display.quit()


def test_on_frame_runs_once_per_physics_frame():
    env = TeamAsteroidsEnv(get_scenario("duel", max_frames=600))
    env.reset(seed=0)
    frames = []
    env.on_frame = lambda: frames.append(env.frame)
    env.step({a: [0, 0, 0] for a in env.agents})
    assert frames == list(range(1, env.config.action_repeat + 1))


def test_on_frame_stops_at_game_end():
    env = TeamAsteroidsEnv(get_scenario("duel", max_frames=3, action_repeat=2))
    env.reset(seed=0)
    calls = []
    env.on_frame = lambda: calls.append(env.frame)
    while env.agents:
        env.step({a: [0, 0, 0] for a in env.agents})
    assert calls == [1, 2, 3]  # last decision stops at the time limit mid-repeat
