# Desktop app design

A single pygame window for playing and watching, so nothing has to be run from the command line except
training. Scripts (`train.py`, `evaluate.py`, `league.py`, `watch.py`) keep working unchanged.

## Stack

- **pygame-ce + pygame_gui.** pygame_gui provides dropdowns, sliders and text boxes and targets
  pygame-ce, a maintained drop-in fork of pygame. The project switched from `pygame` to `pygame-ce`.
- Theme: `marl_asteroids/app/theme.json`, matching the renderer palette.

## Architecture

```
marl_asteroids/
  session.py      MatchSpec (plain data: mode + arena size, or a named scenario; per-ship
                  controllers, camera...)
                  -> build_match() -> Match (config, env, policies, renderer settings)
                  Shared by the app and scripts/watch.py, so both build games the same way.
  checkpoints.py  Finds models under models/, keeps only those whose observation size matches
                  the current environment (stale checkpoints would crash), and reads display
                  names / Elo from optional manifest.json files.
  app/
    main.py       App: owns the window, the pygame_gui manager, the screen stack and the loop
    screens.py    MainMenu, ExperimentalMenu, MatchSetup, Game (+ pause overlay), Results,
                  RogueliteGame, Shop, RunOver, Settings, Gallery
  roguelite.py    RogueliteRun: run state, shop, wave schedule; drives the env between waves.
                  Plain data and pure functions, no pygame, so a run plays headless in tests.
    settings.py   Persisted preferences and last match setup (~/.config/marl-asteroids/app.json)
```

Two refactors make this possible:

1. **The app owns the window.** The renderer gains a `surface` mode: it draws to an offscreen
   surface and never opens or closes the display. The app blits that surface into the window.
2. **Per-physics-frame hook.** Agents decide every `action_repeat` (2) physics frames. The env
   calls an optional `on_frame` callback after each physics frame, which the game screen uses to
   draw and pace at 60 fps. Without it the game would only update at 30 fps.

## Screens

- **Main menu:** Roguelite survival, Classic survival, Experimental, Settings, Quit.
  Classic survival opens match setup with survival preselected.
- **Experimental:** Custom match (match setup with every mode) and the Gallery.
- **Roguelite game:** you fly `blue_0` in the medium arena. The side panels show your fleet (with
  health pips and wingman tier) and the wave, enemies left, credits, and the enemy legend (`L`
  toggles it). Clearing a wave shows "WAVE CLEARED" for about a second, then pushes the shop; the
  game screen keeps its env and renderer, so Continue resumes the same run. `Esc` pauses (resume,
  quit run). Your ship's destruction ends the run.
- **Shop:** credits and the wave just cleared on top; "Your Ship" and "Wingmen" columns with level,
  next cost and Buy per item; the legend on the right; Continue (or Enter) starts the next wave.
  Wingman upgrades unlock with the first wingman. `Esc` does nothing here, so the next wave can't
  be skipped by accident.
- **Run over:** wave reached, waves cleared, kills (yours and the fleet's), wingmen, credits
  earned. New run or main menu.
- **Match setup:** mode (deathmatch, sudden death, station siege, survival, capture the flag, escort), arena size
  (mini 600 / small 1200 / medium 2400 / large 4800), ships per team, spawns (combat modes), asteroid count, splitting, time limit, a controller per
  ship (you, a trained model, or a scripted bot), difficulty, camera. Defaults to you flying
  `blue_0` in a 2v2 deathmatch. Switching to an objective mode swaps trained agents for the
  objective-aware bot, since the agents can't see stations or waves. The last setup is remembered.
- **Game:** `Esc` pauses (resume / restart / quit to menu), `Tab` toggles the observation overlay.
- **Results:** winner (or waves survived), how it ended, kills, damage, station damage (siege),
  team kills, crashes. Play again, change setup, or menu.
- **Settings:** window size, visual effects, minimap, observation overlay default, frame rate.
- **Gallery:** placeholder listing planned MARL exhibits (under Experimental).

Menus run a dimmed bot-vs-bot station siege in the background (scripted bots only, so it never
needs checkpoints); it can be turned off in Settings.

## Rules

- At most one human-controlled ship (one keyboard).
- Difficulty handicaps apply only to AI ships on teams without the human, and only when a human
  is playing (same rules as `watch.py`).
- Training stays in scripts; the app never launches training.
