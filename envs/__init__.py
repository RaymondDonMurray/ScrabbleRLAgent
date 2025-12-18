"""
Scrabble RL Environment Package

This package contains the complete mini-Scrabble reinforcement learning environment
compatible with the Gymnasium API.

Main Components:
    - MiniScrabbleEnv: The main Gymnasium environment for training RL agents
    - Board, TileBag, Dictionary: Core game components
    - Player, Scorer: Game mechanics

Usage:
    from envs import MiniScrabbleEnv

    env = MiniScrabbleEnv(dictionary_path='path/to/words.txt', board_size=5, rack_size=5)
    observation, info = env.reset()

    # Training loop
    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
"""

from .game_mechanics import MiniScrabbleEnv, Player, Scorer
from .game_components import Board, TileBag, Dictionary

__all__ = [
    # Main environment
    'MiniScrabbleEnv',

    # Game components
    'Board',
    'TileBag',
    'Dictionary',

    # Game mechanics
    'Player',
    'Scorer',
]

__version__ = '0.1.0'
