import random
import math
import pygame


def get_human_action():
    """Maps keyboard state to [Thrust, Turn, Shoot]"""
    keys = pygame.key.get_pressed()
    
    # Thrust: W or Up Arrow
    thrust = 1 if keys[pygame.K_UP] or keys[pygame.K_w] else 0
    
    # Turn: A/Left Arrow (1) or D/Right Arrow (2)
    turn = 0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        turn = 1
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        turn = 2
        
    # Shoot: Spacebar
    shoot = 1 if keys[pygame.K_SPACE] else 0
    
    return [thrust, turn, shoot]

def get_dummy_action():
    return [0, 0, 0]

def get_heuristic_action(base_env, agent_name, p=1.0):
    """The smart bot. We pass the base_env so it can read exact physics coordinates."""
    if random.random() > p:
        return [0, 0, 0]
        
    ego_ship = base_env.ships[agent_name]
    if not ego_ship.active: return [0, 0, 0]

    # Evasion Logic
    danger_zone = 100.0
    closest_threat_dist = float('inf')
    threat_dx, threat_dy = 0, 0

    for ast in base_env.asteroids:
        if ast.active:
            dx, dy, dist = base_env._get_relative_wrapped_distance(ego_ship.pos, ast.pos)
            if dist < closest_threat_dist:
                closest_threat_dist = dist
                threat_dx, threat_dy = dx, dy

    for bullet in base_env.bullets:
        if bullet.active and bullet.team_id != ego_ship.team_id:
            dx, dy, dist = base_env._get_relative_wrapped_distance(ego_ship.pos, bullet.pos)
            if dist < closest_threat_dist:
                closest_threat_dist = dist
                threat_dx, threat_dy = dx, dy

    if closest_threat_dist < danger_zone:
        escape_angle = math.atan2(-threat_dy, -threat_dx)
        angle_diff = (escape_angle - (ego_ship.angle % (2 * math.pi)) + math.pi) % (2 * math.pi) - math.pi
        turn = 2 if angle_diff > 0.1 else (1 if angle_diff < -0.1 else 0)
        thrust = 1 if abs(angle_diff) < 1.5 else 0 
        return [thrust, turn, 0]

    # Attack Logic
    closest_dist = float('inf')
    target_pos = None

    for other_name, other_ship in base_env.ships.items():
        if other_name == agent_name: continue
        if other_ship.active and other_ship.team_id != ego_ship.team_id:
            dx, dy, dist = base_env._get_relative_wrapped_distance(ego_ship.pos, other_ship.pos)
            if dist < closest_dist:
                closest_dist = dist
                target_pos = (ego_ship.pos[0] + dx, ego_ship.pos[1] + dy)

    if target_pos is None: return [0, 0, 0]

    target_angle = math.atan2(target_pos[1] - ego_ship.pos[1], target_pos[0] - ego_ship.pos[0])
    angle_diff = (target_angle - (ego_ship.angle % (2 * math.pi)) + math.pi) % (2 * math.pi) - math.pi

    turn = 2 if angle_diff > 0.1 else (1 if angle_diff < -0.1 else 0)
    shoot = 1 if abs(angle_diff) < 0.2 else 0
    thrust = 1 if abs(angle_diff) < 0.5 and closest_dist > 150 else 0

    return [thrust, turn, shoot]


def get_evasion_dummy_action(base_env, agent_name, ast_only=False, is_shooting=False):

    ego_ship = base_env.ships[agent_name]
    if not ego_ship.active: return [0, 0, 0]

    # Evasion Logic
    danger_zone = 150.0
    closest_threat_dist = float('inf')
    threat_dx, threat_dy = 0, 0

    for ast in base_env.asteroids:
        if ast.active:
            dx, dy, dist = base_env._get_relative_wrapped_distance(ego_ship.pos, ast.pos)
            if dist < closest_threat_dist:
                closest_threat_dist = dist
                threat_dx, threat_dy = dx, dy

    if not ast_only:
        for bullet in base_env.bullets:
            if bullet.active and bullet.team_id != ego_ship.team_id:
                dx, dy, dist = base_env._get_relative_wrapped_distance(ego_ship.pos, bullet.pos)
                if dist < closest_threat_dist:
                    closest_threat_dist = dist
                    threat_dx, threat_dy = dx, dy

    for name, ship in base_env.ships.items():
        if name == agent_name: continue
        if ship.active:
            dx, dy, dist = base_env._get_relative_wrapped_distance(ego_ship.pos, ship.pos)
            if dist < closest_threat_dist:
                closest_threat_dist = dist
                threat_dx, threat_dy = dx, dy

    if closest_threat_dist < danger_zone:
        escape_angle = math.atan2(-threat_dy, -threat_dx)
        angle_diff = (escape_angle - (ego_ship.angle % (2 * math.pi)) + math.pi) % (2 * math.pi) - math.pi
        turn = 2 if angle_diff > 0.1 else (1 if angle_diff < -0.1 else 0)
        thrust = 1 if abs(angle_diff) < 1.5 else 0 
        return [thrust, turn, 0]
    
    # Default logic
    thrust = 0
    turn = 0
    shoot = 0

    if random.random() < .1:
        thrust = 1
    if random.random() < .1:
        turn = 1
    if random.random() > .9:
        turn = 2
    if is_shooting:
        if random.random() < .1:
            shoot = 1
    

    return [thrust, turn, shoot]