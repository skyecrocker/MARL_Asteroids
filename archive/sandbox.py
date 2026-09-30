import pygame
import math
import random
from stable_baselines3 import PPO

# Import your environment and scenarios
from Environment import TeamAsteroidsEnv
from settings import SCENARIO_CURRICULUM_LEVEL1, SCENARIO_CURRICULUM_LEVEL2, SCENARIO_CURRICULUM_LEVEL3, GameConfig
from agents import get_human_action, get_heuristic_action, get_dummy_action, get_skeet_dummy_action
from RewardShaping import DenseRewardWrapper



# ==========================================
# 2. THE SANDBOX RUNNER
# ==========================================

def run_sandbox(scenario: GameConfig, agent_mapping: dict, model_path: str = None):
    # Initialize base environment with rendering
    env = TeamAsteroidsEnv(config=scenario)
    env = DenseRewardWrapper(env)
    env.render_mode = "human"
    
    # Load PPO Model ONLY if a neural network agent is in the mapping
    model = None
    if "ppo" in agent_mapping.values():
        print(f"Loading PPO Model from {model_path}...")
        model = PPO.load(model_path)
    
    env.reset()
    env.render()
    
    frame_count = 0
    running = True
    while running and len(env.agents) > 0:
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Frame: {frame_count}")
        # Handle OS Events (Window closing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = {}
        for agent in env.agents:
            # Fallback to 'dummy' if an agent isn't explicitly mapped
            brain_type = agent_mapping.get(agent, "ppo")
            
            if brain_type == "human":
                actions[agent] = get_human_action()
                
            elif brain_type == "heuristic":
                actions[agent] = get_heuristic_action(env.env, agent, p=1.0)
                
            elif brain_type == "dumber_heuristic":
                actions[agent] = get_heuristic_action(env.env, agent, p=0.6)

            elif brain_type == "skeet_dummy":
                actions[agent] = get_skeet_dummy_action(env.env, agent)
                
            elif brain_type == "ppo" and model is not None:
                # The Neural Network needs its specially formatted 49-dim array
                obs = env._get_obs(agent)
                action, _ = model.predict(obs, deterministic=True)
                actions[agent] = action
                
            else:
                actions[agent] = get_dummy_action()
                
        # Step and render
        _, _, terminations, _, _ = env.step(actions)
        
        for agent, term in terminations.items():
            if term:
                # Fetch the final cumulative score from our tracker
                final_score = env.env._cumulative_rewards.get(agent, 0.0)
                
                # Print a clean, formatted death log to the console
                print(f"💀 {agent.upper()} Terminated | Final Score: {final_score:+.2f}")

        env.render()
        
    env.close()
    print("Sandbox Simulation Complete.")

# ==========================================
# 3. CONFIGURE AND LAUNCH
# ==========================================

if __name__ == "__main__":
    # --- Setup Your Match Here ---
    
    # Pick the scenario from your config
    selected_scenario = SCENARIO_CURRICULUM_LEVEL3
    
    # Map who controls which ship
    my_matchup = {
        "red_0": "heuristic",
        "blue_0": "ppo", 
    }
    
    # Provide the path to your trained model
    #saved_model = "models/ppo_level1_v4_2M.zip"
    saved_model = "models/ppo_level2_v4_2M/best_model.zip"

    print("Launching Sandbox...")
    run_sandbox(scenario=selected_scenario, 
                agent_mapping=my_matchup, 
                model_path=saved_model)