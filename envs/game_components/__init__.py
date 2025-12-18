"""
Game Components Package

This package contains the core game components for the Scrabble RL environment:
- Board: The game board with premium squares and word placement logic
- TileBag: Manages the available tiles and drawing mechanism
- Dictionary: Word validation and dictionary management
"""

from .board import Board
from .tile_bag import TileBag
from .dictionary import Dictionary

__all__ = [
    'Board',
    'TileBag',
    'Dictionary',
]
