"""
Game Mechanics Package

This package contains the game mechanics and environment implementation:
- Player: Player state management (rack, score, tile operations)
- Scorer: Word scoring logic including premiums and bonuses
- MiniScrabbleEnv: Gymnasium-compatible environment for mini-Scrabble
"""

from .player import Player
from .scorer import Scorer
from .miniscrabbleenv import MiniScrabbleEnv

__all__ = [
    'Player',
    'Scorer',
    'MiniScrabbleEnv',
]
