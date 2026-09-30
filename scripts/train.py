import os
import random
import math
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from pettingzoo.utils.env import ParallelEnv
from typing import Callable

os.environ["SDL_VIDEODRIVER"] = "dummy"

from marl_asteroids.env import TeamAsteroidsEnv
from marl_asteroids.rewards import DenseRewardWrapper
from marl_asteroids.config import SCENARIO_CURRICULUM_LEVEL1, SCENARIO_CURRICULUM_LEVEL2, SCENARIO_CURRICULUM_LEVEL3, SCENARIO_CURRICULUM_LEVEL4, GameConfig

from marl_asteroids.policies import get_dummy_action, get_heuristic_action, get_evasion_dummy_action


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 and goes down to 0.0
        return progress_remaining * initial_value
    return func

# ==========================================
# 2. THE MIXED AGENT WRAPPER
# ==========================================

class MixedAgentWrapper(ParallelEnv):
    """
    Hides non-learning agents from Stable-Baselines3. 
    It automatically generates bot actions in the background so the NN only sees itself.
    """
    def __init__(self, env, base_env, agent_mapping):
        self.env = env
        self.base_env = base_env # Need direct access to physics for the bots
        self.agent_mapping = agent_mapping
        
        self.metadata = env.metadata
        
        # EXPOSE ONLY PPO AGENTS to the neural network
        self.possible_agents = [a for a in env.possible_agents if agent_mapping.get(a) == "ppo"]
        self.agents = []

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)
    
    @property
    def render_mode(self):
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, value):
        self.env.render_mode = value
        
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = [a for a in self.env.agents if self.agent_mapping.get(a) == "ppo"]
        
        # Filter down to only PPO agents
        ppo_obs = {a: obs[a] for a in self.agents}
        ppo_infos = {a: infos[a] for a in self.agents}
        return ppo_obs, ppo_infos

    def step(self, actions):
        import numpy as np # Needed for zero-padding
        
        # 1. Capture exactly who PyTorch thinks is alive BEFORE the physics step
        agents_before_step = self.agents[:] 

        # 2. Inject bot actions for everyone else
        full_actions = actions.copy()
        for agent in self.env.agents:
            if agent not in actions: 
                b_type = self.agent_mapping.get(agent, "dummy")
                if b_type == "heuristic":
                    full_actions[agent] = get_heuristic_action(self.base_env, agent, p=1.0)
                elif b_type == "dumber_heuristic":
                    full_actions[agent] = get_heuristic_action(self.base_env, agent, p=0.3)
                elif b_type == "evasion_heuristic":
                    full_actions[agent] = get_evasion_dummy_action(self.base_env, agent)
                elif b_type == "evasion_heuristic_easy":
                    actions[agent] = get_evasion_dummy_action(self.base_env, agent, ast_only=True)
                else:
                    full_actions[agent] = get_dummy_action()

        # 3. Step the underlying environment
        obs, rewards, terms, truncs, infos = self.env.step(full_actions)

        # 4. Filter results perfectly for PyTorch
        # Get the shape of a valid observation to create a blank one just in case
        sample_agent = self.possible_agents[0]
        blank_obs = np.zeros(self.env.observation_space(sample_agent).shape, dtype=np.float32)

        ppo_obs = {}
        ppo_rewards = {}
        ppo_terms = {}
        ppo_truncs = {}
        ppo_infos = {}

        # PyTorch MUST receive dicts for everyone who was alive before the step.
        # If the base env dropped them from the dict early, we safely pad them.
        for a in agents_before_step:
            ppo_obs[a] = obs.get(a, blank_obs)
            ppo_rewards[a] = rewards.get(a, 0.0)
            
            # If an agent is missing from the base env's terms, assume it died
            ppo_terms[a] = terms.get(a, True) 
            ppo_truncs[a] = truncs.get(a, False)
            ppo_infos[a] = infos.get(a, {})

        # 5. Update who is ACTUALLY still alive for the next frame
        self.agents = [
            a for a in self.env.agents 
            if self.agent_mapping.get(a) == "ppo" 
            # CRITICAL: Check both terms AND truncs!
            and not (ppo_terms.get(a, False) or ppo_truncs.get(a, False)) 
        ]

        return ppo_obs, ppo_rewards, ppo_terms, ppo_truncs, ppo_infos

# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================

def run_training(scenario: GameConfig, agent_mapping: dict, run_name: str, total_timesteps: int, load_path: str = None):
    print(f"--- Starting Training Run: {run_name} ---")
    
    # ==========================================
    # --- NEW: Create a dedicated folder for this run ---
    # ==========================================
    run_dir = f"models/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Initialize core environments
    base_env = TeamAsteroidsEnv(config=scenario)
    base_env.render_mode = None 
    env = DenseRewardWrapper(base_env)

    # 2. Apply our custom mixed agent filter
    env = MixedAgentWrapper(env, base_env, agent_mapping)

    # 3. SuperSuit Vectorization
    env = ss.black_death_v3(env) 
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    env = VecMonitor(env)

    # 4. Curriculum Loading vs. New Model
    if load_path and os.path.exists(load_path + ".zip"):
        print(f"Curriculum Learning: Loading weights from {load_path}")
        model = PPO.load(load_path, env=env, tensorboard_log="runs/tensorboard/")
    else:
        print("Initializing fresh PPO brain...")
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,           
            batch_size=256,         
            gamma=0.99,             
            ent_coef=0.01,          
            tensorboard_log="runs/tensorboard/"
        )

    eval_callback = EvalCallback(
        env, 
        best_model_save_path=run_dir, # <--- Dumps best_model.zip into the run folder
        log_path=f"runs/evals/{run_name}/", # <--- Isolates eval logs too!
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )

    # 5. Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=eval_callback)

    # 6. Save
    # <--- Dumps final_model.zip into the run folder alongside best_model.zip
    save_target = f"{run_dir}/final_model" 
    model.save(save_target)
    print(f"Training complete. Models safely stored in: {run_dir}/")
    
    env.close()

# ==========================================
# 4. LAUNCH CONFIGURATION
# ==========================================

if __name__ == "__main__":
    
    # --- PHASE 1:  (Target Practice) ---
    # Goal: Teach the agent how to aim and shoot a stationary target.
    
    run_training(
        scenario=SCENARIO_CURRICULUM_LEVEL4,
        agent_mapping={
            "red_0": "heuristic",
            "red_1": "heuristic",
            "blue_0": "ppo",
            "blue_1": "ppo",
        },
        run_name="ppo_level4a_v4aa_8M",
        total_timesteps=8_000_000,
        load_path="models/legacy/ppo_level3a_v4aa_8M/best_model"
    )
    
    # --- PHASE 2: THE JOUST (Curriculum Learning) ---
    # Goal: Take the aiming skills from Phase 1, and teach it to dodge a moving enemy.
    # Uncomment this when Phase 1 finishes!
    
    """
    run_training(
        scenario=SCENARIO_JOUST,
        agent_mapping={
            "red_0": "ppo",        
            "blue_0": "heuristic"  # Now it fights back!
        },
        run_name="ppo_phase2_joust",
        total_timesteps=500_000,
        load_path="models/ppo_phase1_preschool" # <-- Inherit the brain!
    )
    """