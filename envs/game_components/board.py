import numpy as np
from typing import Union, Tuple
from dictionary import Dictionary

class Board: 

	def __init__(self, dictionary: Dictionary, size: int = 5): 
		"""
		Initializes the game board of a given size. 
		
		:param self: Game board object
		:param dictionary: Word dictionary
		:type dictionary: Dictionary
		:param size: Tile-length of the square board
		:type size: int
		
		"""
		self.size = size
		self.dictionary = dictionary
		
		# Making the main board
		self.grid = [[None for _ in range(size)] for _ in range(size)]

		# Making the premium squares
		# 'DL' = double letter, 'DW' = double word, None = normal
		self.premium_squares = self._initialize_premium_squares()

		# Tracking which squares are used for premium scoring
		self.premium_used = [[False for _ in range(size)] for _ in range(size)]


	def _initialize_premium_squares(self): 
		"""Set up premium square layour for mini-Scrabble board (for now, just double letter and double word)"""

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
	
	def is_empty(self, row, col) -> bool:
		"""
		Check if a square is empty.

		:param self: Board object
		:param row: Row coordinate
		:type row: int
		:param col: Column coordinate
		:type col: int
		:return: True if empty, False if occupied
		:rtype: bool
		"""
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
	
	def _can_place_word(self, word, row, col, direction, player_rack, is_first_move) -> Tuple[bool, str]:
		"""
		Validate if a word can be placed at the given position.

		Returns:
			(bool, str): (is_valid, reason_if_invalid)
		"""

		# Check 1: Word fits on board
		if direction == 'H':
			if col + len(word) > self.size:
				return False, "Word extends past board boundary"
		else:  # 'V'
			if row + len(word) > self.size:
				return False, "Word extends past board boundary"

		# Check 2: First move must pass through center
		if is_first_move:
			center = self.size // 2
			passes_through_center = False

			for i in range(len(word)):
				if direction == 'H':
					if row == center and col + i == center:
						passes_through_center = True
				else:
					if row + i == center and col == center:
						passes_through_center = True

			if not passes_through_center:
				return False, "First word must pass through center"

		# Check 3: Subsequent moves must connect to existing letters
		if not is_first_move:
			connects = False

			for i in range(len(word)):
				r = row if direction == 'H' else row + i
				c = col + i if direction == 'H' else col

				# Check if this position has an adjacent letter
				if self._has_adjacent_letter(r, c):
					connects = True

				# Or if we're placing on top of an existing letter
				if not self.is_empty(r, c):
					connects = True

			if not connects:
				return False, "Word must connect to existing letters"

		# Check 4: Can we form this word with our rack + board letters?
		letters_needed = []
		for i, letter in enumerate(word):
			r = row if direction == 'H' else row + i
			c = col + i if direction == 'H' else col

			if self.is_empty(r, c):
				# Need to place a new letter
				letters_needed.append(letter)
			else:
				# Letter already on board - must match
				if self.grid[r][c] != letter:
					return False, f"Letter mismatch at position ({r}, {c})"

		# Check if player has the needed letters in their rack
		rack_copy = player_rack.copy()
		for letter in letters_needed:
			if letter in rack_copy:
				rack_copy.remove(letter)
			else:
				# Check for blank tile
				if '_' in rack_copy:
					rack_copy.remove('_')
				else:
					return False, "Player doesn't have required letters"

		# Check 5: All perpendicular words formed must be valid
		perpendicular_words = self._get_perpendicular_words(word, row, col, direction)
		for perp_word, _, _ in perpendicular_words:
			if len(perp_word) > 1 and not self.dictionary.is_valid_word(perp_word):
				return False, f"Perpendicular word '{perp_word}' is invalid"

		return True, "Valid placement"

	def _has_adjacent_letter(self, row, col):
		"""Check if a position has any adjacent letters (not diagonal)."""
		adjacents = [
			(row - 1, col),  # above
			(row + 1, col),  # below
			(row, col - 1),  # left
			(row, col + 1),  # right
		]

		for r, c in adjacents:
			if 0 <= r < self.size and 0 <= c < self.size:
				if not self.is_empty(r, c):
					return True

		return False

	def _get_perpendicular_words(self, word, row, col, direction):
		"""
		Find all perpendicular words formed by placing this word.

		Returns:
			List of (word, start_row, start_col) tuples
		"""
		perpendicular_words = []

		# For each letter in the main word
		for i in range(len(word)):
			r = row if direction == 'H' else row + i
			c = col + i if direction == 'H' else col

			# Skip if this position already has a letter (we're not placing new)
			if not self.is_empty(r, c):
				continue

			# Look in perpendicular direction
			if direction == 'H':
				# Main word is horizontal, look vertically
				perp_word = self._extract_vertical_word(r, c, word[i])
			else:
				# Main word is vertical, look horizontally
				perp_word = self._extract_horizontal_word(r, c, word[i])

			if perp_word:
				perpendicular_words.append(perp_word)

		return perpendicular_words