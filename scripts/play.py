"""Play as red_0 (WASD/arrows + space) against heuristic bots.

Moved from the old ``Environment.py`` ``__main__`` block.
"""
import pygame

from marl_asteroids.config import GameConfig
from marl_asteroids.env import TeamAsteroidsEnv
from marl_asteroids.policies import get_heuristic_action, get_human_action
from marl_asteroids.rewards import DenseRewardWrapper


def main():
    config = GameConfig(asteroids_split=True, num_asteroids=5, n_red=6, m_blue=6)
    env = TeamAsteroidsEnv(config=config)
    env = DenseRewardWrapper(env)
    env.render_mode = "human"

    env.reset()
    env.render()  # Initialize PyGame before the loop starts

    human_agent = "red_0"

    running = True
    while running and len(env.agents) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = {}
        for agent in env.agents:
            if agent == human_agent:
                actions[agent] = get_human_action()
            else:
                actions[agent] = get_heuristic_action(env.env, agent, p=1.0)

        _, _, terminations, _, _ = env.step(actions)
        env.render()

        for agent, term in terminations.items():
            if term:
                print(f"{agent} was destroyed!")

    env.close()
    print("Game Over!")


if __name__ == "__main__":
    main()
