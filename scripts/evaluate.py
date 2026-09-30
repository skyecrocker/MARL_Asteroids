import pygame
import math
import random
import time
import csv
from stable_baselines3 import PPO

# Import your environment and scenarios
from marl_asteroids.env import TeamAsteroidsEnv
from marl_asteroids.config import *
from marl_asteroids.policies import get_human_action, get_heuristic_action, get_dummy_action, get_evasion_dummy_action
from marl_asteroids.rewards import DenseRewardWrapper

# ==========================================
# 2. THE SANDBOX RUNNER (Visual Mode)
# ==========================================

def run_sandbox(scenario: GameConfig, agent_mapping: dict, model_path: str = None, model_alt_path: str = None):
    env = TeamAsteroidsEnv(config=scenario)
    env = DenseRewardWrapper(env)
    env.render_mode = "human"
    
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
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = {}
        for agent in env.agents:
            # --- NEW: Team Color Fallback Parser ---
            # Try to get the specific agent (e.g., "red_0"). If not, get the team (e.g., "red").
            brain_type = agent_mapping.get(agent)
            if not brain_type:
                team_color = agent.split("_")[0]
                brain_type = agent_mapping.get(team_color, "ppo") # Default to ppo if totally unmapped
            
            if brain_type == "human":
                actions[agent] = get_human_action()
            elif brain_type.startswith("heuristic_"):
                p_val = float(brain_type.split("_")[1])
                actions[agent] = get_heuristic_action(env.env, agent, p=p_val)
            elif brain_type == "heuristic":
                actions[agent] = get_heuristic_action(env.env, agent, p=1.0)
            elif brain_type == "dumber_heuristic":
                actions[agent] = get_heuristic_action(env.env, agent, p=0.3)
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
    env = TeamAsteroidsEnv(config=scenario)
    env = DenseRewardWrapper(env)
    env.render_mode = None 
    
    model = None
    # We check if 'ppo' is requested either as a specific agent or a team
    if "ppo" in agent_mapping.values():
        model = PPO.load(model_path)

    model_alt = None
    if "ppo_alt" in agent_mapping.values():
        model_alt = PPO.load(model_alt_path)

    stats = {
        "red_wins": 0,
        "blue_wins": 0,
        "draws_timeout": 0,
        "draws_mutual_destruction": 0,
        "game_lengths": []
    }

    for game in range(num_games):
        env.reset()
        frame_count = 0
        
        while len(env.agents) > 0:
            frame_count += 1
            actions = {}
            for agent in env.agents:
                # --- NEW: Team Color Fallback Parser ---
                brain_type = agent_mapping.get(agent)
                if not brain_type:
                    team_color = agent.split("_")[0]
                    brain_type = agent_mapping.get(team_color, "ppo")
                
                if brain_type.startswith("heuristic_"):
                    p_val = float(brain_type.split("_")[1])
                    actions[agent] = get_heuristic_action(env.env, agent, p=p_val)
                elif brain_type == "heuristic":
                    actions[agent] = get_heuristic_action(env.env, agent, p=1.0)
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
                    
            _, _, terminations, truncations, _ = env.step(actions)
            
            if any(truncations.values()):
                stats["draws_timeout"] += 1
                break 
                
        else:
            red_alive = sum(1 for name, ship in env.env.ships.items() if "red" in name and ship.active)
            blue_alive = sum(1 for name, ship in env.env.ships.items() if "blue" in name and ship.active)
            
            if red_alive > 0 and blue_alive == 0:
                stats["red_wins"] += 1
            elif blue_alive > 0 and red_alive == 0:
                stats["blue_wins"] += 1
            else:
                stats["draws_mutual_destruction"] += 1

        stats["game_lengths"].append(frame_count)

    return stats


# ==========================================
# 4. CSV BULK EVALUATOR RUNNER
# ==========================================

def run_bulk_evaluations(scenarios_dict: dict, blue_agents: list, red_agents: list, main_ppo_path: str, alt_ppo_path: str = None, num_games: int = 100, csv_filename = "comprehensive_evaluation_results.csv"):
    print(f"\n--- Starting Bulk Evaluation Matrix ---")
    print(f"Scenarios: {len(scenarios_dict)} | Blue Agents: {len(blue_agents)} | Red Agents: {len(red_agents)} | Games/Match: {num_games}")
    
    results = []
    total_matches = len(scenarios_dict) * len(blue_agents) * len(red_agents)
    current_match = 0
    
    for scenario_name, scenario_config in scenarios_dict.items():
        for blue_brain in blue_agents:
            for red_brain in red_agents:
                current_match += 1
                print(f"[{current_match}/{total_matches}] Scenario: {scenario_name} | Blue: {blue_brain} vs Red: {red_brain}")
                
                # We now map the entire team color to the brain type!
                # If a scenario has 2 blue ships, they both get the blue_brain.
                mapping = {
                    "blue": blue_brain,
                    "red": red_brain
                }
                
                stats = run_evaluation(
                    scenario=scenario_config, 
                    agent_mapping=mapping, 
                    model_path=main_ppo_path, 
                    model_alt_path=alt_ppo_path, 
                    num_games=num_games
                )
                
                avg_len = sum(stats["game_lengths"]) / num_games if num_games > 0 else 0
                win_rate = (stats["blue_wins"] / num_games) * 100
                
                # Updated CSV Headers for clarity
                row = {
                    "Scenario": scenario_name,
                    "Blue Agent": blue_brain,
                    "Red Agent": red_brain,
                    "Blue Wins": stats["blue_wins"],
                    "Red Wins": stats["red_wins"],
                    "Draws (Time)": stats["draws_timeout"],
                    "Draws (Mutual)": stats["draws_mutual_destruction"],
                    "Blue Win Rate (%)": f"{win_rate:.1f}%",
                    "Avg Game Length": f"{avg_len:.1f}"
                }
                results.append(row)

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n✅ Bulk Evaluation Complete! Results saved to '{csv_filename}'.")


# ==========================================
# 5. CONFIGURE AND LAUNCH
# ==========================================

if __name__ == "__main__":
    
    saved_model = "models/legacy/ppo_level3a_v4aa_8M/best_model.zip"
    alt_saved_model = "models/legacy/ppo_level3a_v4aa_8M/best_model.zip"


    # MODES: "VISUAL", "SINGLE_EVAL", "BULK_EVAL"
    MODE = "BULK_EVAL" 
    
    if MODE == "BULK_EVAL":
        # 1. Define your Scenarios
        scenarios_to_test = {
            "Level 1 (1v1 Open Spaace)": SCENARIO_EVAL_LEVEL1,
            "Level 2 (1v1 w/ 2 Asteroids)": SCENARIO_EVAL_LEVEL2,
            "Level 3 (2v2 w/ 2 Asteroids)": SCENARIO_EVAL_LEVEL3,
            "Level 4 (3v3 w/ 4 Asteroids)": SCENARIO_EVAL_LEVEL4
        }
        
        # 2. Build the list of Red Opponents (Baselines/Dummies)
        red_pursuit_pool = [
            "dummy",
            "evasion_heuristic_easy",
            "evasion_heuristic",
        ]
        
        blue_pursuit_pool = [             
            "heuristic_0.25",
            "heuristic_0.5",
            "heuristic_0.75",
            "heuristic_1.0",
            "ppo"
        ]

        red_combat_pool = [
            "heuristic_0.25",
            "heuristic_0.5",
            "heuristic_0.75",
            "heuristic_1.0",
        ]
        
        blue_combat_pool = [             
            "ppo"
        ]

            
        # 4. Fire the cross-product matrix runner
        run_bulk_evaluations(
            scenarios_dict=scenarios_to_test,
            blue_agents=blue_pursuit_pool,
            red_agents=red_pursuit_pool,
            main_ppo_path=saved_model,
            num_games=1000,
            csv_filename="results/pursuit_evaluation_results.csv"
        )

        run_bulk_evaluations(
            scenarios_dict=scenarios_to_test,
            blue_agents=blue_combat_pool,
            red_agents=red_combat_pool,
            main_ppo_path=saved_model,
            num_games=1000,
            csv_filename="results/combat_evaluation_results.csv"
        )

    elif MODE == "SINGLE_EVAL":
        selected_scenario = SCENARIO_CURRICULUM_LEVEL4
        
        # You can still map specific agents, or use the new team strings
        my_matchup = {"red": "evasion_heuristic", "blue": "ppo"}
        
        stats = run_evaluation(selected_scenario, my_matchup, saved_model, num_games=100)
        print(f"Blue Wins: {stats['blue_wins']}")

    elif MODE == "VISUAL":
        selected_scenario = SCENARIO_CURRICULUM_LEVEL4
        
        # Applies "dumber_heuristic" to all red ships, and "ppo" to all blue ships
        my_matchup = {"red": "dumber_heuristic", "blue": "ppo"}
        
        print("Launching Visual Sandbox...")
        run_sandbox(selected_scenario, my_matchup, saved_model)