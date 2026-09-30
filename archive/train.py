import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

# Import your environment and wrapper
from Environment import TeamAsteroidsEnv, DenseRewardWrapper

def train():
    # 1. Initialize the core environment
    # Turn off rendering during training for maximum speed
    base_env = TeamAsteroidsEnv(n_red=2, m_blue=2, friendly_fire=False, asteroids_split=True)
    base_env.render_mode = None 
    
    # 2. Add our Custom Reward Shaping
    env = DenseRewardWrapper(base_env)
    env = ss.black_death_v3(env)

    # 3. SuperSuit Magic: Flatten MARL to SARL for Stable-Baselines3
    # This converts the ParallelEnv into a vectorized Gym environment 
    # where all agents share the same observation/action spaces.
    env = ss.pettingzoo_env_to_vec_env_v1(env) 
    
    # Concat environments (we use 1 env instance, but it handles 4 agents)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')


    # 4. Initialize PPO
    print("Initializing PPO global brain...")
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,           # Steps per update
        batch_size=256,         # Batch size
        gamma=0.99,             # Discount factor
        ent_coef=0.01,          # Entropy coefficient (encourages exploration!)
        tensorboard_logs="./tensorboard_logs/"
    )

    # 5. Train! (Let's start with 500,000 timesteps)
    timesteps = 500_000
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    # 6. Save the global policy
    os.makedirs("models", exist_ok=True)
    model.save("models/asteroids_ppo_v1")
    print("Model saved to models/asteroids_ppo_v1.zip")
    
    env.close()

if __name__ == "__main__":
    train()