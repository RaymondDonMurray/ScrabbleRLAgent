import numpy as np
from typing import Union

class Board: 

	def __init__(self, size: int = 5): 
		"""
		Initializes the game board of a given size. 
		
		:param self: Game board object
		:param size: Tile-length of the square board
		:type size: int
		"""
		self.size = size
		
		# Making the main board
		self.grid = [[None for _ in range(size)] for _ in range(size)]

		# Making the premium squares
		# 'DL' = double letter, 'DW' = double word, None = normal
		self.premium_squares = self._initialize_premium_squares()

		# Tracking which squares are used for premium scoring
		self.premium_used = [[False for _ in range(size)] for _ in range(size)]


	def _initialize_premium_squares(self): 
		"""Set up premium square layour for mini-Scrabble board"""

		premiums = [[None for _ in range(self.size)] for _ in range(self.size)]

		# Simple symmetric pattern for 5x5
		# Center: double word
		premiums[2][2] = 'DW'

		# Corners: double letter
		premiums[0][0] = premiums[0][4] = 'DL'
		premiums[4][0] = premiums[4][4] = 'DL'

		# A few more double letters
		premiums[1][1] = premiums[1][3] = 'DL'
		premiums[3][1] = premiums[3][3] = 'DL'

		return premiums
	
	def place_word(self, word: str, row:int, col:int, direction:str) -> bool: 
		"""
		Places a word on the board. 
		
		:param self: Board object
		:param word: Word to place
		:type word: str
		:param row: Starting row
		:type row: int
		:param col: Starting column
		:type col: int
		:param direction: 'H' (horizontal) or 'V' (vertical)
		:type direction: str
		:return: True if successful placement, False if invalid placement
		:rtype: bool
		"""

		# Validate placement (detailed later)
		if not self._can_place_word(word, row, col, direction):
			return False

		# Place letters
		for i, letter in enumerate(word):
			if direction == 'H':
				self.grid[row][col + i] = letter
			else:  # 'V'
				self.grid[row + i][col] = letter

		return True
		
	def get_letter(self, row:int, col:int) -> Union[str, None]: 
		"""
		Get the letter at a specific position on the board
		
		:param self: Board object. 
		:param row: Row coordinate
		:type row: int
		:param col: Column coordinate
		:type col: int
		:return: Letter if occupied, None otherwise
		:rtype: str or None
		"""
		if 0 <= row < self.size and 0 <= col < self.size:
			return self.grid[row][col]
		return None
	
	def is_empty(self, row, col):
		"""Check if a square is empty."""
		return self.grid[row][col] is None

	def to_array(self):
		"""
		Convert board to numerical array for neural network input.

		Returns:
			numpy array of shape (size, size) with:
			0 = empty, 1-26 = letters A-Z
		"""
		import numpy as np

		array = np.zeros((self.size, self.size), dtype=np.int32)
		for i in range(self.size):
			for j in range(self.size):
				if self.grid[i][j] is not None:
					# A=1, B=2, ..., Z=26
					array[i][j] = ord(self.grid[i][j]) - ord('A') + 1

		return array

	def __str__(self):
		"""String representation for debugging/visualization."""
		lines = []
		for row in self.grid:
			line = ' '.join(cell if cell else '.' for cell in row)
			lines.append(line)
		return '\n'.join(lines)