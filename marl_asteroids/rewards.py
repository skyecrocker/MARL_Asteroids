import math
from pettingzoo import ParallelEnv
from .config import GameConfig

class DenseRewardWrapper(ParallelEnv):
    """
    Wraps the TeamAsteroidsEnv to provide dense micro-rewards for learning,
    without altering the core [-1, 0, 1] game logic.
    """
    def __init__(self, env, max_steps=1000):
        self.env = env
        self.max_steps = max_steps
        self.current_step = 0  # Start the clock
        
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.agents = env.possible_agents[:]

    # FIX: SuperSuit expects these methods to exist
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def render_mode(self):
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, value):
        self.env.render_mode = value

    def reset(self, seed=None, options=None):
        self.current_step = 0  # Reset the clock on a new episode
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]

        # Initialize the cumulative scoreboard at 0
        self.cumulative_rewards = {agent: 0.0 for agent in self.env.possible_agents}

        # Inject it into the base environment so render() can see it
        self.env._cumulative_rewards = self.cumulative_rewards

        return obs, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def step(self, actions):
        self.current_step += 1 # Tick the clock

        # 1. Calculate micro-rewards BEFORE the physics step based on intent
        dense_rewards = {agent: 0.0 for agent in self.agents}
        
        for agent_name, action in actions.items():
            ego_ship = self.env.ships[agent_name]
            if not ego_ship.active:
                continue
                
            thrust, turn, shoot = action
            
            #Find nearest enemy
            closest_dist = float('inf')
            target_pos = None
            
            for other_name, other_ship in self.env.ships.items():
                if other_ship.active and other_ship.team_id != ego_ship.team_id:
                    dx, dy, dist = self.env._get_relative_wrapped_distance(ego_ship.pos, other_ship.pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        target_pos = (ego_ship.pos[0] + dx, ego_ship.pos[1] + dy)
                        
            if target_pos is not None:
                target_angle = math.atan2(target_pos[1] - ego_ship.pos[1], target_pos[0] - ego_ship.pos[0])
                ego_angle = ego_ship.angle % (2 * math.pi)
                target_angle = target_angle % (2 * math.pi)
                
                angle_diff = (target_angle - ego_angle + math.pi) % (2 * math.pi) - math.pi
                
                # ==========================================
                # 1. THE RADAR LOCK (Replaces Turn Rewards)
                # ==========================================
                # alignment_score is 1.0 at dead center, and decays to 0.0 if facing exactly away (pi radians)
                alignment_score = 1.0 - (abs(angle_diff) / math.pi)
                
                # Give a continuous trickle (up to +0.01) just for keeping the crosshairs on the target.
                # The agent will naturally learn to use the turn buttons to maximize this state.
                dense_rewards[agent_name] += 0.02 * alignment_score
                
                # ==========================================
                # 2. THE PROXIMITY GRADIENT (Replaces Binary Sweet Spot)
                # ==========================================
                # Calculate the maximum possible distance in a wrap-around map 
                # (Distance from center to the exact corner)
                max_dist = math.hypot(self.env.width / 2, self.env.height / 2)
                
                # Score is 1.0 if touching the enemy, 0.0 if at the exact opposite side of the map
                proximity_score = 1.0 - (min(closest_dist, max_dist) / max_dist)
                
                # Shift and scale the reward. 
                # If they are far away, they bleed (-0.01). 
                # As they get closer, it smoothly rises up to a positive reward (+0.01).
                # The gradient ALWAYS points toward the enemy!
                distance_reward = (proximity_score * 0.02) - 0.01
                dense_rewards[agent_name] += distance_reward

                # ==========================================
                # 3. THE TRIGGER (The continuous shoot reward we added!)
                # ==========================================
                if shoot == 1 and ego_ship.cooldown <= 0:
                    accuracy_score = 1.0 - (abs(angle_diff) / 0.5)
                    shoot_reward = 0.05 * accuracy_score
                    dense_rewards[agent_name] += max(shoot_reward, -0.01)

                # ==========================================
                # 4. THE REPULSOR FIELD (Asteroid Avoidance)
                # ==========================================
                WARNING_RADIUS = 150.0  # Adjust based on your ship's turn speed
                hazard_penalty = 0.0
                
                # Loop through the base environment's asteroids
                # (Adjust 'self.env.asteroids' to however your base env stores them)
                for rock in self.env.asteroids:
                    if rock.active:
                        _, _, rock_dist = self.env._get_relative_wrapped_distance(ego_ship.pos, rock.pos)
                        
                        if rock_dist < WARNING_RADIUS:
                            # The closer it gets, the worse it hurts. 
                            # At exactly WARNING_RADIUS, penalty is 0.
                            # At point-blank, penalty approaches -0.05 per frame.
                            pain_multiplier = 1.0 - (rock_dist / WARNING_RADIUS)
                            hazard_penalty -= 0.50 * pain_multiplier
                
                # Apply the hazard penalty to the dense rewards
                dense_rewards[agent_name] += hazard_penalty

                # ==========================================
                # 4. THE REPULSOR FIELD (Bullet Avoidance)
                # ==========================================
                WARNING_RADIUS = 100.0  # Adjust based on your ship's turn speed
                hazard_penalty = 0.0
                
                # Loop through the base environment's asteroids
                # (Adjust 'self.env.asteroids' to however your base env stores them)
                for rock in self.env.asteroids:
                    if rock.active:
                        _, _, rock_dist = self.env._get_relative_wrapped_distance(ego_ship.pos, rock.pos)
                        
                        if rock_dist < WARNING_RADIUS:
                            # The closer it gets, the worse it hurts. 
                            # At exactly WARNING_RADIUS, penalty is 0.
                            # At point-blank, penalty approaches -0.05 per frame.
                            pain_multiplier = 1.0 - (rock_dist / WARNING_RADIUS)
                            hazard_penalty -= 0.50 * pain_multiplier
                
                # Apply the hazard penalty to the dense rewards
                dense_rewards[agent_name] += hazard_penalty

        # Step the base environment
        observations, base_rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents[:]
        
        # Combine rewards
        combined_rewards = {}
        for agent in base_rewards.keys():
            # 1. Balanced Existential Dread: Bleed just enough to discourage stalling
            time_penalty = -0.05 
            
            # 2. The Jackpot
            amplified_kill_reward = base_rewards[agent] * 100.0
            
            # 3. Micro-rewards (Now they actually matter against the -0.05 penalty!)
            dense = dense_rewards.get(agent, 0.0) 
            
            combined_rewards[agent] = amplified_kill_reward + dense + time_penalty

        # --- THE WIN CONDITION (1v1 Sniper School) ---
        red_alive = any("red" in a for a in self.agents)
        blue_alive = any("blue" in a for a in self.agents)
        
        # If the dummy dies (or blue dies), end the game immediately
        if not red_alive or not blue_alive:
            for agent in self.agents:
                terminations[agent] = True  # Natural end to the game
            self.agents = []
            self.env.agents = []
            
        # --- ENFORCE THE TIME LIMIT ---
        elif self.current_step >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True  # Forced end due to time
            self.agents = []
            self.env.agents = []

        # <--- NEW: Add the frame's reward to the running total --->
        for agent, rew in combined_rewards.items():
            if agent in self.cumulative_rewards:
                self.cumulative_rewards[agent] += rew
                
        # Inject the updated totals down into the base environment
        self.env._cumulative_rewards = self.cumulative_rewards
                
        return observations, combined_rewards, terminations, truncations, infos
    

    def _get_obs(self, agent_name):
        return self.env._get_obs(agent_name)