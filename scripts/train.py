"""Train a shared-parameter PPO team policy with self-play.

Examples:
    uv run python scripts/train.py --preset baseline --run baseline_heuristic
    uv run python scripts/train.py --preset selfplay --run sp_v1 --timesteps 20_000_000
    uv run python scripts/train.py --preset selfplay --run sp_v1b --init models/sp_v1/final_model.zip
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from marl_asteroids.opponents import OpponentPool
from marl_asteroids.training import SelfPlayCallback, WinRateEvalCallback, build_game_configs
from marl_asteroids.vec_env import ParallelTeamVecEnv, TeamVecEnv

EVAL_SUITE = [("duel", "heuristic_1.0"), ("duel_rocks", "heuristic_1.0"),
              ("team2", "heuristic_1.0"), ("team3", "heuristic_1.0")]

PRESETS = {
    # PPO vs scripted bots only (the legacy setup, but with the fixed env/obs/reward)
    "baseline": dict(
        mix={"duel": 6, "duel_rocks": 6, "team2": 6, "team3": 4},
        scripted=["heuristic_0.25", "heuristic_0.5", "heuristic_0.75", "heuristic_1.0", "evasion"],
        p_self=0.0, p_scripted=1.0, snapshot_every=0,
    ),
    # League: live self-play + frozen snapshots (PFSP) + scripted bots for robustness
    "selfplay": dict(
        mix={"duel": 6, "duel_rocks": 6, "team2": 6, "team3": 4, "team2_vs_1": 2},
        scripted=["heuristic_0.5", "heuristic_1.0", "evasion"],
        p_self=0.35, p_scripted=0.2, snapshot_every=250_000,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="selfplay")
    ap.add_argument("--run", required=True, help="Run name (models/<run>, runs/tensorboard/<run>)")
    ap.add_argument("--timesteps", type=float, default=10e6)
    ap.add_argument("--init", default=None, help="Warm-start from a checkpoint .zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=float, default=500_000)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--team-spirit", type=float, default=None, help="Override RewardConfig.team_spirit")
    ap.add_argument("--shaping-scale", type=float, default=None, help="Override RewardConfig.shaping_scale")
    ap.add_argument("--threads", type=int, default=4, help="Torch threads for the PPO update")
    ap.add_argument("--workers", type=int, default=11, help="Env worker processes (1 = in-process)")
    ap.add_argument("--games-mult", type=int, default=3, help="Multiply the preset's game counts")
    args = ap.parse_args()

    th.set_num_threads(args.threads)
    preset = PRESETS[args.preset]
    run_dir = os.path.join("models", args.run)
    os.makedirs(run_dir, exist_ok=True)

    configs = build_game_configs({k: v * args.games_mult for k, v in preset["mix"].items()})
    for cfg in configs:
        cfg.rewards.shaping_gamma = args.gamma
        if args.team_spirit is not None:
            cfg.rewards.team_spirit = args.team_spirit
        if args.shaping_scale is not None:
            cfg.rewards.shaping_scale = args.shaping_scale

    pool_kwargs = dict(scripted=preset["scripted"], p_self=preset["p_self"],
                       p_scripted=preset["p_scripted"], seed=args.seed)
    if args.workers > 1:
        base_venv = ParallelTeamVecEnv(configs, pool_kwargs, n_workers=args.workers, seed=args.seed)
    else:
        base_venv = TeamVecEnv(configs, OpponentPool(**pool_kwargs), seed=args.seed)
    venv = VecMonitor(base_venv)
    print(f"{venv.num_envs} learner slots across {len(configs)} games; obs dim {venv.observation_space.shape[0]}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "preset": preset}, f, indent=1, default=str)

    tb = "runs/tensorboard"
    if args.init:
        model = PPO.load(args.init, env=venv, device="cpu", tensorboard_log=tb,
                         custom_objects={"learning_rate": args.lr, "ent_coef": args.ent_coef})
    else:
        model = PPO(
            "MlpPolicy", venv, device="cpu", seed=args.seed, tensorboard_log=tb, verbose=1,
            learning_rate=args.lr, n_steps=args.n_steps, batch_size=4096, n_epochs=5,
            gamma=args.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
            vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=th.nn.Tanh),
        )

    callbacks = [
        SelfPlayCallback(snapshot_every=preset["snapshot_every"],
                         snapshot_dir=os.path.join(run_dir, "snapshots") if preset["snapshot_every"] else None),
        WinRateEvalCallback(EVAL_SUITE, eval_every=int(args.eval_every), n_games=args.eval_games, save_dir=run_dir),
    ]
    t0 = time.time()
    model.learn(total_timesteps=int(args.timesteps), callback=callbacks, tb_log_name=args.run,
                reset_num_timesteps=args.init is None)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Models in {run_dir}/")
    venv.close()


if __name__ == "__main__":
    main()
