# Desktop app design (v1)

A single pygame window for playing and watching, so nothing has to be run from the command line except
training. Scripts (`train.py`, `evaluate.py`, `league.py`, `watch.py`) keep working unchanged.

## Stack

- **pygame-ce + pygame_gui.** pygame_gui provides dropdowns, sliders and text boxes and targets
  pygame-ce, a maintained drop-in fork of pygame. The project switched from `pygame` to `pygame-ce`.
- Theme: `marl_asteroids/app/theme.json`, matching the renderer palette.

## Architecture

```
marl_asteroids/
  session.py      MatchSpec (plain data: scenario, overrides, per-ship controllers, camera...)
                  -> build_match() -> Match (config, env, policies, renderer settings)
                  Shared by the app and scripts/watch.py, so both build games the same way.
  checkpoints.py  Finds models under models/, keeps only those whose observation size matches
                  the current environment (stale checkpoints would crash), and reads display
                  names / Elo from optional manifest.json files.
  app/
    main.py       App: owns the window, the pygame_gui manager, the screen stack and the loop
    screens.py    MainMenu, MatchSetup, Game (+ pause overlay), Results, Settings, Gallery
    settings.py   Persisted preferences and last match setup (~/.config/marl-asteroids/app.json)
```

Two refactors make this possible:

1. **The app owns the window.** The renderer gains a `surface` mode: it draws to an offscreen
   surface and never opens or closes the display. The app blits that surface into the window.
2. **Per-physics-frame hook.** Agents decide every `action_repeat` (2) physics frames. The env
   calls an optional `on_frame` callback after each physics frame, which the game screen uses to
   draw and pace at 60 fps. Without it the game would only update at 30 fps.

## Screens

- **Main menu:** Play, Watch, Gallery, Settings, Quit.
- **Match setup:** scenario, ships per team, map scale, asteroid count, splitting, time limit, a
  controller per ship (you, a trained model, or a scripted bot), difficulty, camera. Play defaults
  to you flying `blue_0`; Watch defaults to AI on both sides. The last setup is remembered.
- **Game:** `Esc` pauses (resume / restart / quit to menu), `Tab` toggles the observation overlay.
- **Results:** winner, duration, kills, damage, team kills. Play again, change setup, or menu.
- **Settings:** window size, visual effects, minimap, observation overlay default, frame rate.
- **Gallery:** placeholder listing planned MARL exhibits.

Menus run a dimmed bot-vs-bot match in the background (scripted bots only, so it never needs
checkpoints); it can be turned off in Settings.

## Rules

- At most one human-controlled ship (one keyboard).
- Difficulty handicaps apply only to AI ships on teams without the human, and only when a human
  is playing (same rules as `watch.py`).
- Training stays in scripts; the app never launches training.
