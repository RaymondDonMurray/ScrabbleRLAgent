import random

class TileBag: 
	"""Class that manages the collection of available tiles."""

	def __init__(self, use_simplified: bool = True): 
		"""
		Docstring for __init__
		
		:param self: TileBag object
		:param use_simplified: Boolean describing if the object should be a simple or standard bag of tiles
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
	
	def draw(self, n:int=1): 
		"""
		Drawing n tiles from the bag. Removing the drawn tiles from the bag.
		
		:param self: TileBag object
		:param n: Number of tiles to draw
		:type n: int
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
		"""
		return len(self.remaining_tiles) == 0	

	def get_number_of_remaining_tiles(self) -> int: 
		"""
		Get the number of remaining tiles in the bag.
		
		:param self: TileBag object
		"""
		return len(self.remaining_tiles)	