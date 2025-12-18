import random

class TileBag: 
	"""Class that manages the collection of available tiles."""

	def __init__(self, use_simplified: bool = True):
		"""
		Initialize tile bag with either simplified or standard tile distribution.

		:param self: TileBag object
		:param use_simplified: If True, use simplified tile set for mini-Scrabble. If False, use standard Scrabble distribution
		:type use_simplified: bool
		"""
		if use_simplified: 
			self.tiles = self._create_simplified_tiles()
		else: 
			self.tiles = self._create_standard_tiles()

		self.remaining_tiles = self.tiles.copy()

	def _create_simplified_tiles(self):
		"""Create a simplified tile set for 5x5 board."""
		
		# For mini-Scrabble, you might have:
		# 20 total tiles (players start with 5 each, bag has 10)
		tiles = []

		# Common letters (more frequent)
		for letter in 'AEIOU':
			tiles.extend([letter] * 2)  # 2 of each vowel

		for letter in 'RSTLN':
			tiles.extend([letter] * 2)  # 2 of each common consonant

		# Less common letters
		for letter in 'BCDFGHJKMPQVWXYZ':
			tiles.append(letter)  # 1 of each

		# Blanks (wildcards)
		tiles.extend(['_'] * 2)

		return tiles
	
	def draw(self, n: int = 1) -> list[str]:
		"""
		Draw n tiles from the bag. Drawn tiles are removed from the bag.

		:param self: TileBag object
		:param n: Number of tiles to draw
		:type n: int
		:return: List of drawn tiles (letters as strings)
		:rtype: list[str]
		"""

		if n > len(self.remaining_tiles): # check that n is not more than the number of tiles left in the bag
			n = len(self.remaining_tiles)

		drawn = random.sample(self.remaining_tiles, n)
		for tile in drawn: 
			self.remaining_tiles.remove(tile)

		return drawn
	
	def is_empty(self) -> bool:
		"""
		Check if the tile bag is empty.

		:param self: TileBag object
		:return: True if no tiles remain, False otherwise
		:rtype: bool
		"""
		return len(self.remaining_tiles) == 0	

	def get_number_of_remaining_tiles(self) -> int:
		"""
		Get the number of remaining tiles in the bag.

		:param self: TileBag object
		:return: Number of tiles remaining
		:rtype: int
		"""
		return len(self.remaining_tiles)	