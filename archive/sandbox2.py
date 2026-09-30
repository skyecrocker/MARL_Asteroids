import pygame
import math
import random
import time
from stable_baselines3 import PPO

# Import your environment and scenarios
from Environment import TeamAsteroidsEnv
from settings import *
from agents import get_human_action, get_heuristic_action, get_dummy_action, get_evasion_dummy_action
from RewardShaping import DenseRewardWrapper

# ==========================================
# 2. THE SANDBOX RUNNER (Visual Mode)
# ==========================================

def run_sandbox(scenario: GameConfig, agent_mapping: dict, model_path: str = None, model_alt_path: str = None):
    # Initialize base environment with rendering
    env = TeamAsteroidsEnv(config=scenario)
    env = DenseRewardWrapper(env)
    env.render_mode = "human"
    
    # Load PPO Model ONLY if a neural network agent is in the mapping
    model = None
    if "ppo" in agent_mapping.values():
        print(f"Loading PPO Model from {model_path}...")
        model = PPO.load(model_path)

    model_alt = None
    if "ppo_alt" in agent_mapping.values():
        print(f"Loading PPO Model from {model_alt_path}...")
        model_alt = PPO.load(model_alt_path)
    
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
            brain_type = agent_mapping.get(agent, "ppo")
            
            if brain_type == "human":
                actions[agent] = get_human_action()
            elif brain_type == "heuristic":
                actions[agent] = get_heuristic_action(env.env, agent, p=1.0)
            elif brain_type == "dumber_heuristic":
                actions[agent] = get_heuristic_action(env.env, agent, p=0.6)
            elif brain_type == "evasion_heuristic":
                    actions[agent] = get_evasion_dummy_action(env.env, agent)
            elif brain_type == "evasion_heuristic_easy":
                    actions[agent] = get_evasion_dummy_action(env.env, agent, ast_only=True)
            elif brain_type == "ppo" and model is not None:
                obs = env._get_obs(agent)
                action, _ = model.predict(obs, deterministic=True)
                actions[agent] = action
            elif brain_type == "ppo_alt" and model_alt is not None:
                obs = env._get_obs(agent)
                action, _ = model_alt.predict(obs, deterministic=True)
                actions[agent] = action
            else:
                actions[agent] = get_dummy_action()
                
        # Step and render
        _, _, terminations, _, _ = env.step(actions)
        
        for agent, term in terminations.items():
            if term:
                final_score = env.env._cumulative_rewards.get(agent, 0.0)
                print(f"💀 {agent.upper()} Terminated | Final Score: {final_score:+.2f}")

        env.render()
        
    env.close()
    print("Sandbox Simulation Complete.")


# ==========================================
# 3. THE EVALUATOR (Headless Mode)
# ==========================================

def run_evaluation(scenario: GameConfig, agent_mapping: dict, model_path: str = None, model_alt_path: str = None, num_games: int = 100):
    print(f"\n--- Starting Headless Evaluation: {num_games} Games ---")
    
    # No rendering for maximum speed
    env = TeamAsteroidsEnv(config=scenario)
    env = DenseRewardWrapper(env)
    env.render_mode = None 
    
    model = None
    if "ppo" in agent_mapping.values():
        print(f"Loading PPO Model from {model_path}...")
        model = PPO.load(model_path)

    model_alt = None
    if "ppo_alt" in agent_mapping.values():
        print(f"Loading alt PPO Model from {model_alt_path}...")
        model_alt = PPO.load(model_alt_path)

    stats = {
        "red_wins": 0,
        "blue_wins": 0,
        "draws_timeout": 0,
        "draws_mutual_destruction": 0,
        "game_lengths": []
    }

    start_time = time.time()

    for game in range(num_games):
        env.reset()
        frame_count = 0
        
        # Run until the episode ends (either by termination or truncation)
        while len(env.agents) > 0:
            frame_count += 1
            
            actions = {}
            for agent in env.agents:
                brain_type = agent_mapping.get(agent, "ppo")
                if brain_type == "heuristic":
                    actions[agent] = get_heuristic_action(env.env, agent, p=1.0)
                elif brain_type == "dumber_heuristic":
                    actions[agent] = get_heuristic_action(env.env, agent, p=0.3)
                elif brain_type == "evasion_heuristic":
                    actions[agent] = get_evasion_dummy_action(env.env, agent)
                elif brain_type == "evasion_heuristic_easy":
                    actions[agent] = get_evasion_dummy_action(env.env, agent, ast_only=True, is_shooting=False)
                elif brain_type == "ppo" and model is not None:
                    obs = env._get_obs(agent)
                    action, _ = model.predict(obs, deterministic=True)
                    actions[agent] = action
                elif brain_type == "ppo_alt" and model_alt is not None:
                    obs = env._get_obs(agent)
                    action, _ = model_alt.predict(obs, deterministic=True)
                    actions[agent] = action
                else:
                    actions[agent] = get_dummy_action()
                    
            _, _, terminations, truncations, _ = env.step(actions)
            
            # Check for a time-limit draw
            if any(truncations.values()):
                stats["draws_timeout"] += 1
                break 
                
        else:
            # If we exit the while loop naturally, the game ended via death/termination.
            # We look at the base environment to see whose ship is actually still active.
            red_alive = sum(1 for name, ship in env.env.ships.items() if "red" in name and ship.active)
            blue_alive = sum(1 for name, ship in env.env.ships.items() if "blue" in name and ship.active)
            
            if red_alive > 0 and blue_alive == 0:
                stats["red_wins"] += 1
            elif blue_alive > 0 and red_alive == 0:
                stats["blue_wins"] += 1
            else:
                # If they crashed into each other or died on the exact same frame
                stats["draws_mutual_destruction"] += 1

        # Record length
        stats["game_lengths"].append(frame_count)
        
        # Progress tracker
        if (game + 1) % 20 == 0 or (game + 1) == num_games:
            print(f"Completed {game + 1}/{num_games} matches...")

    # Calculate final statistics
    end_time = time.time()
    avg_len = sum(stats["game_lengths"]) / num_games
    min_len = min(stats["game_lengths"])
    max_len = max(stats["game_lengths"])
    
    print("\n=================================")
    print("      EVALUATION RESULTS         ")
    print("=================================")
    print(f"Matches Played : {num_games}")
    print(f"Time Taken     : {end_time - start_time:.2f} seconds")
    print(f"Blue Wins      : {stats['blue_wins']} ({(stats['blue_wins']/num_games)*100:.1f}%)")
    print(f"Red Wins       : {stats['red_wins']} ({(stats['red_wins']/num_games)*100:.1f}%)")
    print(f"Draw (Timeout) : {stats['draws_timeout']} ({(stats['draws_timeout']/num_games)*100:.1f}%)")
    print(f"Draw (Mutual)  : {stats['draws_mutual_destruction']} ({(stats['draws_mutual_destruction']/num_games)*100:.1f}%)")
    print("---------------------------------")
    print(f"Match Lengths  : Avg {avg_len:.1f} | Min {min_len} | Max {max_len} frames")
    print("=================================\n")


# ==========================================
# 4. CONFIGURE AND LAUNCH
# ==========================================

if __name__ == "__main__":
    # --- SETUP YOUR MATCH ---
    selected_scenario = SCENARIO_EVAL_LEVEL3
    my_matchup = {
        "red_0": "heuristic",
        "red_1": "heuristic",
        "blue_0": "ppo", 
        "blue_1": "ppo"
    }
    saved_model = "models/ppo_level3a_v4aa_8M/best_model.zip"
    alt_saved_model = "models/ppo_level2_v4a_2M/best_model.zip"


    # --- THE TOGGLE SWITCH ---
    EVALUATION_MODE = False  # Set to False to watch a single game visually
    NUM_GAMES_TO_EVAL = 100

    if EVALUATION_MODE:
        run_evaluation(scenario=selected_scenario, 
                       agent_mapping=my_matchup, 
                       model_path=saved_model,
                       model_alt_path=alt_saved_model,
                       num_games=NUM_GAMES_TO_EVAL)
    else:
        print("Launching Visual Sandbox...")
        run_sandbox(scenario=selected_scenario, 
                    agent_mapping=my_matchup, 
                    model_path=saved_model,
                    model_alt_path=alt_saved_model)