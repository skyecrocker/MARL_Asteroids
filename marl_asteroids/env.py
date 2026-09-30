import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv
import math
import numpy as np
import pygame
import random
from .entities import Ship, Asteroid, Bullet
from .config import GameConfig


class TeamAsteroidsEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "team_asteroids_v0"}

    def __init__(self, config: GameConfig):
        self.config = config
        self.n_red = config.n_red
        self.m_blue = config.m_blue
        self.num_asteroids = config.num_asteroids

        self.friendly_fire = config.friendly_fire
        self.asteroids_split = config.asteroids_split

        self.width = config.width
        self.height = config.height
        
        # Define agents
        self.possible_agents = [f"red_{i}" for i in range(self.n_red)] + \
                               [f"blue_{i}" for i in range(self.m_blue)]
        
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}

        # We will define the spaces in a helper method
        self.observation_spaces = {agent: self._get_obs_space() for agent in self.possible_agents}
        self.action_spaces = {agent: self._get_action_space() for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        
        self.agents = self.possible_agents[:]
        self.ships = {}
        
        # --- NEW: Minimum Safe Distance ---
        MIN_SPAWN_DIST = 150.0  # Minimum pixels between ships. Adjust this!
        MAX_ATTEMPTS = 50       # Don't get stuck in an infinite loop
        
        # 1. Spawn Ships
        for agent in self.agents:
            team_id = 0 if "red" in agent else 1
            spawns = self.config.red_spawns if team_id == 0 else self.config.blue_spawns
            
            # SAFEGUARD: Treat None as an empty list
            if spawns is None:
                spawns = []
            
            # Extract the agent index
            agent_idx = int(agent.split("_")[1])
            
            valid_spawn = False
            attempts = 0
            
            # Keep rolling the dice until we find a safe spot (or run out of attempts)
            while not valid_spawn and attempts < MAX_ATTEMPTS:
                
                # Propose coordinates
                if agent_idx < len(spawns):
                    bp = spawns[agent_idx]
                    if bp.random_spawn:
                        x = np.random.uniform(0, self.width)
                        y = np.random.uniform(0, self.height)
                        angle = np.random.uniform(0, 2 * math.pi)
                    else:
                        x = np.random.normal(bp.x, bp.x_noise) % self.width
                        y = np.random.normal(bp.y, bp.y_noise) % self.height
                        angle = np.random.normal(bp.angle, bp.angle_noise)
                else:
                    x = np.random.uniform(0, self.width)
                    y = np.random.uniform(0, self.height)
                    angle = np.random.uniform(0, 2 * math.pi)

                # Check proposed (x, y) against all ALREADY placed ships
                too_close = False
                for placed_ship in self.ships.values():
                    # Wrapped distance math (vital for Asteroids-style maps)
                    dx = min(abs(x - placed_ship.pos[0]), self.width - abs(x - placed_ship.pos[0]))
                    dy = min(abs(y - placed_ship.pos[1]), self.height - abs(y - placed_ship.pos[1]))
                    dist = math.hypot(dx, dy)
                    
                    if dist < MIN_SPAWN_DIST:
                        too_close = True
                        break # Break the inner loop, this spot is dead, try again
                
                if not too_close:
                    valid_spawn = True # We found a safe spot!
                    
                attempts += 1

            # --- Now instantiate the ship at the valid coordinates ---
            if agent_idx < len(spawns):
                bp = spawns[agent_idx]
                ship = Ship(x, y, team_id, agent, fire_rate=bp.fire_rate)
                ship.vel[0] = np.random.normal(bp.vx, bp.vx_noise)
                ship.vel[1] = np.random.normal(bp.vy, bp.vy_noise)
                ship.angle = angle
            else:
                # Fallback TRUE random ship
                ship = Ship(x, y, team_id, agent)
                ship.angle = angle
                
            self.ships[agent] = ship

        # 2. Spawn Asteroids
        self.asteroids = []
        
        # --- SAFEGUARD: Treat None as an empty list ---
        ast_spawns = self.config.asteroid_spawns if self.config.asteroid_spawns is not None else []
        
        # First, spawn explicitly defined asteroids
        for bp in ast_spawns:
            x = np.random.normal(bp.x, bp.x_noise) % self.width
            y = np.random.normal(bp.y, bp.y_noise) % self.height
            vx = np.random.normal(bp.vx, bp.vx_noise)
            vy = np.random.normal(bp.vy, bp.vy_noise)
            
            self.asteroids.append(Asteroid(x, y, vx, vy, size=bp.size))
            
        # Second, fill the remaining required num_asteroids with random spawns
        asteroids_needed = self.config.num_asteroids - len(ast_spawns)
        for _ in range(max(0, asteroids_needed)):
            safe = False
            while not safe:
                ax = np.random.uniform(0, self.width)
                ay = np.random.uniform(0, self.height)
                safe = True
                for ship in self.ships.values():
                    # Standard distance check for safe spawning
                    dist = math.hypot(ax - ship.pos[0], ay - ship.pos[1])
                    if dist < 100.0:  
                        safe = False
                        break
            
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(0.5, 1.5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.asteroids.append(Asteroid(ax, ay, vx, vy, size=3))

        self.bullets = []
        
        # Return initial observations
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # 1. Apply Actions (Thrust, Turn, Shoot)
        for agent, action in actions.items():
            self._apply_action(agent, action)
            
        # 2. Physics Step
        self._step_physics()
        
        # 3. Collision Detection 
        # (Modify this function to return who killed who, e.g., a list of tuples: [("blue_0", "red_0")])
        kill_events = self._resolve_collisions() 
        
        # 4. Calculate Rewards & Dones
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # --- THE MISSING SYNCHRONIZATION LOGIC ---
        
        # A. Process the kills from the collision step
        if kill_events:
            for shooter, victim in kill_events:
                if shooter in rewards:
                    rewards[shooter] += 1.0  # Base kill reward! (Wrapper multiplies this by 1000)
                if victim in rewards:
                    rewards[victim] -= 1.0   # Base death penalty
        
        # B. Sync the terminations with the physical ship state
        for agent in self.agents:
            # Assuming you store ship objects in a self.ships dictionary
            ship = self.ships[agent] 
            
            # If the collision logic marked the ship as dead
            if not ship.active:  
                terminations[agent] = True
        
        # -----------------------------------------
        
        # Generate the observation array for everyone currently alive (The Eyes!)
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        self._latest_rewards = rewards.copy()

        # Remove dead agents from self.agents (No more zombies!)
        self.agents = [agent for agent in self.agents if not terminations[agent]]
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return

        # 1. Initialize PyGame lazily
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((int(self.width), int(self.height)))
            self.clock = pygame.time.Clock()

        # 2. Clear Screen (Black space)
        self.screen.fill((0, 0, 0))

        # 3. Draw Asteroids (Grey circles)
        for ast in self.asteroids:
            if ast.active:
                pygame.draw.circle(self.screen, (150, 150, 150), (int(ast.pos[0]), int(ast.pos[1])), int(ast.radius))
                # Draw a slight outline for style
                pygame.draw.circle(self.screen, (200, 200, 200), (int(ast.pos[0]), int(ast.pos[1])), int(ast.radius), 1)

        # 4. Draw Bullets (Yellow circles)
        for bullet in self.bullets:
            if bullet.active:
                pygame.draw.circle(self.screen, (255, 255, 0), (int(bullet.pos[0]), int(bullet.pos[1])), int(bullet.radius))

        # 5. Draw Ships (Triangles indicating heading)
        if not hasattr(self, 'debug_font'):
            pygame.font.init()
            # You can change the size (18) to fit your screen better
            self.debug_font = pygame.font.SysFont('Arial', 18)


        for agent_name, ship in self.ships.items():
            if not ship.active:
                continue
            
            # Color based on team
            color = (255, 50, 50) if ship.team_id == 0 else (50, 150, 255)
            
            # Calculate triangle points based on heading angle (cast to int for PyGame)
            tip = (int(ship.pos[0] + math.cos(ship.angle) * ship.radius),
                   int(ship.pos[1] + math.sin(ship.angle) * ship.radius))
            
            left = (int(ship.pos[0] + math.cos(ship.angle + 2.5) * ship.radius),
                    int(ship.pos[1] + math.sin(ship.angle + 2.5) * ship.radius))
            
            right = (int(ship.pos[0] + math.cos(ship.angle - 2.5) * ship.radius),
                     int(ship.pos[1] + math.sin(ship.angle - 2.5) * ship.radius))
            
            pygame.draw.polygon(self.screen, color, [tip, left, right])

            # <--- NEW: Draw the Cumulative Wrapper Rewards --->
            if hasattr(self, '_cumulative_rewards') and agent_name in self._cumulative_rewards:
                cum_rew = self._cumulative_rewards[agent_name]
                
                # Format to 1 decimal place (e.g., "+100.5", "-45.0")
                color = (0, 255, 0) if cum_rew >= 0 else (255, 100, 100)
                text_surface = self.debug_font.render(f"Total: {cum_rew:+.1f}", True, color)
                
                # Offset by 20 pixels and force standard integers
                text_x = int(ship.pos[0]) + 20
                text_y = int(ship.pos[1]) - 20
                
                #self.screen.blit(text_surface, (text_x, text_y))


        # 6. Update display and tick clock (60 FPS)
        if self.render_mode == "human":
            pygame.event.pump() # Keep OS from thinking window is frozen
            pygame.display.flip()
            self.clock.tick(60)
            
    def close(self):
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
            self.screen = None

    # --- Helper Methods ---
    
    def _get_obs_space(self):
        # Ego(5) + 5 Ast(40) + 3 Enemies(24) + 2 Allys(16) + 3 Bullets(24) = 109 dimensions
        # We use a Box space bounds from -1 to 1 (normalized)
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(109,), dtype=np.float32)

    def _get_obs(self, agent_name):
        ego = self.ships[agent_name]
        
        # If the ship is dead, return an array of zeros to maintain shape
        if not ego.active:
            return np.zeros(109, dtype=np.float32)

        # 1. Ego State (Normalized)
        max_speed = self.config.max_speed
        obs = [
            ego.vel[0] / max_speed,
            ego.vel[1] / max_speed,
            math.cos(ego.angle),
            math.sin(ego.angle),
            ego.cooldown / ego.fire_rate, # Max cooldown
        ]

        # --- Helper for padding/sorting K-nearest ---
        def get_k_nearest(entities, k):
            entity_data = []
            for ent in entities:
                is_enemy = 1.0 if isinstance(ent, Ship) and ent.team_id != ego.team_id else 0.0
                is_ally = 1.0 if isinstance(ent, Ship) and ent.team_id == ego.team_id else 0.0
                is_asteroid = 1.0 if isinstance(ent, Asteroid) else 0.0
                is_bullet = 1.0 if isinstance(ent, Bullet) else 0.0

                dx, dy, dist = self._get_relative_wrapped_distance(ego.pos, ent.pos)
                entity_data.append({
                    'dist': dist,
                    'data': [
                        dx / (self.width / 2),  # Normalize to roughly [-1, 1]
                        dy / (self.height / 2),
                        ent.vel[0] / max_speed,
                        ent.vel[1] / max_speed,
                        is_enemy,
                        is_ally,
                        is_asteroid,
                        is_bullet
                    ]
                })
            
            # Sort by distance
            entity_data.sort(key=lambda x: x['dist'])
            
            # Extract the top K data arrays
            flat_data = []
            for i in range(k):
                if i < len(entity_data):
                    flat_data.extend(entity_data[i]['data'])
                else:
                    # Pad with zeros if fewer than K objects exist
                    flat_data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return flat_data

        # 2. Extract active entities
        active_asteroids = [ast for ast in self.asteroids if ast.active]
        active_enemies = [s for name, s in self.ships.items() if s.active and s.team_id != ego.team_id]
        active_allies = [s for name, s in self.ships.items() if s.active and s.team_id == ego.team_id]
        
        # We only care about bullets that don't belong to our team (if friendly fire is off)
        # or just all active bullets if we want to dodge everything.
        active_bullets = [b for b in self.bullets if b.active and (self.friendly_fire or b.team_id != ego.team_id)]

        # 3. Append to observation
        obs.extend(get_k_nearest(active_asteroids, k=5))
        obs.extend(get_k_nearest(active_enemies, k=3))
        obs.extend(get_k_nearest(active_bullets, k=3))
        obs.extend(get_k_nearest(active_allies, k=2))

        # Ensure the final output is a float32 numpy array
        return np.array(obs, dtype=np.float32)

    def _get_action_space(self):
        # [Thrust (2), Turn (3), Shoot (2)]
        return gym.spaces.MultiDiscrete([2, 3, 2])

    def _apply_action(self, agent_name, action):
        ship = self.ships[agent_name]
        if not ship.active:
            return

        thrust, turn, shoot = action
        
        # 1. Turn Logic (Radians)
        turn_speed = self.config.turn_speed
        if turn == 1:
            ship.angle -= turn_speed # Left
        elif turn == 2:
            ship.angle += turn_speed # Right
            
        # 2. Thrust Logic
        thrust_power = self.config.thrust_power
        if thrust == 1:
            # Standard 2D vector resolution using trigonometry
            ship.vel[0] += math.cos(ship.angle) * thrust_power
            ship.vel[1] += math.sin(ship.angle) * thrust_power
            
        # 3. Shoot Logic
        if shoot == 1 and ship.cooldown <= 0:
            # Spawn bullet slightly ahead of the ship so it doesn't shoot itself
            bullet_x = ship.pos[0] + math.cos(ship.angle) * (ship.radius + 5)
            bullet_y = ship.pos[1] + math.sin(ship.angle) * (ship.radius + 5)
            
            # Bullet velocity is ship velocity + bullet speed
            bullet_speed = self.config.bullet_speed
            b_vx = ship.vel[0] + math.cos(ship.angle) * bullet_speed
            b_vy = ship.vel[1] + math.sin(ship.angle) * bullet_speed
            
            self.bullets.append(Bullet(bullet_x, bullet_y, b_vx, b_vy, owner_name=agent_name, team_id=ship.team_id))
            
            # Prevent rapid-fire (e.g., 15 frames between shots)
            ship.cooldown = ship.fire_rate


    def _step_physics(self):
        # 1. Update Ships
        for ship in self.ships.values():
            if not ship.active: continue
            
            ship.pos += ship.vel
            # Apply space drag (99% velocity retention per frame)
            ship.vel *= self.config.drag_coefficient 
            # Wrap-around
            ship.pos[0] %= self.width
            ship.pos[1] %= self.height
            # Reduce shoot cooldown
            if ship.cooldown > 0:
                ship.cooldown -= 1

        # 2. Update Asteroids
        for ast in self.asteroids:
            if not ast.active: continue
            ast.pos += ast.vel
            ast.pos[0] %= self.width
            ast.pos[1] %= self.height

        # 3. Update Bullets
        active_bullets = []
        for bullet in self.bullets:
            bullet.pos += bullet.vel
            bullet.pos[0] %= self.width
            bullet.pos[1] %= self.height
            bullet.lifespan -= 1
            
            if bullet.lifespan > 0 and bullet.active:
                active_bullets.append(bullet)
        
        self.bullets = active_bullets # Remove dead bullets

    def _get_relative_wrapped_distance(self, ego_pos, target_pos):
        """
        Calculates the shortest dx, dy across a toroidal map.
        """
        dx = (target_pos[0] - ego_pos[0] + self.width / 2) % self.width - self.width / 2
        dy = (target_pos[1] - ego_pos[1] + self.height / 2) % self.height - self.height / 2
        
        distance = math.sqrt(dx**2 + dy**2)
        return dx, dy, distance
    
    def _resolve_collisions(self):
        new_asteroids = []
        kill_events = []  # <--- NEW: Create the receipt book
        
        # 1. Bullets vs. Ships
        for bullet in self.bullets:
            if not bullet.active: continue
            
            for agent_name, ship in self.ships.items():
                if not ship.active: continue
                
                # Prevent a ship from shooting itself on frame 1
                if bullet.owner_name == agent_name: 
                    continue
                    
                # Check Friendly Fire config
                if not self.friendly_fire and bullet.team_id == ship.team_id:
                    continue
                    
                # Distance check using toroidal math
                _, _, dist = self._get_relative_wrapped_distance(bullet.pos, ship.pos)
                
                if dist < (bullet.radius + ship.radius):
                    bullet.active = False
                    ship.active = False
                    
                    # <--- THE MAGIC LINK: Record who killed who! --->
                    kill_events.append((bullet.owner_name, agent_name)) 
                    break # Bullet is destroyed, move to next bullet

        # 2. Bullets vs. Asteroids
        for bullet in self.bullets:
            if not bullet.active: continue
            
            for ast in self.asteroids:
                if not ast.active: continue
                
                _, _, dist = self._get_relative_wrapped_distance(bullet.pos, ast.pos)
                
                if dist < (bullet.radius + ast.radius):
                    bullet.active = False # Bullet always dies on impact
                    
                    if self.asteroids_split:
                        ast.active = False
                        if ast.size > 1:
                            # Split into 2 smaller asteroids
                            new_size = ast.size - 1
                            
                            # Deflect velocities by +45 and -45 degrees and speed them up
                            angle1 = math.atan2(ast.vel[1], ast.vel[0]) + (math.pi / 4)
                            angle2 = math.atan2(ast.vel[1], ast.vel[0]) - (math.pi / 4)
                            speed = math.hypot(ast.vel[0], ast.vel[1]) * 1.5 
                            
                            ast1 = Asteroid(ast.pos[0], ast.pos[1], 
                                            math.cos(angle1) * speed, math.sin(angle1) * speed, 
                                            size=new_size)
                            ast2 = Asteroid(ast.pos[0], ast.pos[1], 
                                            math.cos(angle2) * speed, math.sin(angle2) * speed, 
                                            size=new_size)
                            new_asteroids.extend([ast1, ast2])
                    break

        # 3. Ships vs. Asteroids (Crashing)
        for agent_name, ship in self.ships.items():
            if not ship.active: continue
            
            for ast in self.asteroids:
                if not ast.active: continue
                
                _, _, dist = self._get_relative_wrapped_distance(ship.pos, ast.pos)
                
                if dist < (ship.radius + ast.radius):
                    ship.active = False
                    # <--- NEW: The asteroid claims a victim --->
                    kill_events.append(("asteroid", agent_name)) 
                    break
                    
        # 4. Ships vs. Ships (Ramming)
        # <--- CHANGED: Use items() so we can extract the names --->
        ships_list = list(self.ships.items()) 
        
        for i in range(len(ships_list)):
            name_a, ship_a = ships_list[i]
            if not ship_a.active: continue
            
            for j in range(i + 1, len(ships_list)):
                name_b, ship_b = ships_list[j]
                if not ship_b.active: continue
                
                _, _, dist = self._get_relative_wrapped_distance(ship_a.pos, ship_b.pos)
                
                if dist < (ship_a.radius + ship_b.radius):
                    ship_a.active = False
                    ship_b.active = False
                    
                    # <--- NEW: Both ships suffer a death penalty, neither gets a kill --->
                    kill_events.append(("ram", name_a))
                    kill_events.append(("ram", name_b))

        # 5. Cleanup
        if self.asteroids_split:
            # Remove destroyed asteroids and append new ones
            self.asteroids = [ast for ast in self.asteroids if ast.active]
            self.asteroids.extend(new_asteroids)
            
        # <--- NEW: Hand the receipts back to the step() function --->
        return kill_events

