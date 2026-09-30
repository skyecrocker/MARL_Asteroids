import supersuit as ss
from stable_baselines3 import PPO
from Environment import TeamAsteroidsEnv

def evaluate():
    # 1. Initialize base env WITH rendering
    env = TeamAsteroidsEnv(n_red=4, m_blue=4, num_asteroids=1, friendly_fire=False, asteroids_split=False)
    env.render_mode = "human"
    
    # 2. Load the trained model
    model = PPO.load("models/asteroids_ppo_v1")
    
    # 3. Run the evaluation loop
    env.reset()
    env.render()
    
    running = True
    while running and len(env.agents) > 0:
        actions = {}
        for agent in env.agents:
            # We must use _get_obs because we aren't using the SuperSuit wrapper here,
            # we are directly querying the PettingZoo base environment.
            obs = env._get_obs(agent)
            
            # Predict action using the trained model
            # model.predict returns a tuple (action, state)
            action, _ = model.predict(obs, deterministic=True) 
            actions[agent] = action
            
        # Step the environment
        _, _, terminations, _, _ = env.step(actions)
        
        # Render the frame
        env.render()
        
    env.close()
    print("Simulation Complete.")

if __name__ == "__main__":
    evaluate()