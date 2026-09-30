import numpy as np
import math
from .config import GameConfig

class Entity:
    def __init__(self, x, y, vx, vy, radius):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([vx, vy], dtype=np.float32)
        self.radius = radius
        self.active = True # False when destroyed

class Ship(Entity):
    def __init__(self, x, y, team_id, agent_name, fire_rate=GameConfig.bullet_cooldown):
        super().__init__(x, y, vx=0.0, vy=0.0, radius=GameConfig.ship_radius)
        self.team_id = team_id      # 0 for Red, 1 for Blue
        self.agent_name = agent_name
        self.angle = 0.0            # Radians

        self.fire_rate = fire_rate
        self.cooldown = 0
        
class Asteroid(Entity):
    def __init__(self, x, y, vx, vy, size=3):
        # Size 3 = Large, 2 = Medium, 1 = Small
        radius_map = {3: GameConfig.asteroid_large_radius, 
                      2: GameConfig.asteroid_medium_radius, 
                      1: GameConfig.asteroid_small_radius}
        super().__init__(x, y, vx, vy, radius=radius_map[size])
        self.size = size

class Bullet(Entity):
    def __init__(self, x, y, vx, vy, owner_name, team_id):
        super().__init__(x, y, vx, vy, radius=GameConfig.bullet_radius)
        self.owner_name = owner_name
        self.team_id = team_id
        self.lifespan = GameConfig.bullet_lifespan