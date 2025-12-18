from typing import Union
from game_components.tile_bag import TileBag
import numpy as np

class Player: 

	def __init__(
			self,
			id: int,
			rack_size: int = 5,
			score: int = 0
	): 
		"""
		Player object
		
		:param self: Player object
		:param id: Player ID (either 0 or 1 for two-player scrabble)
		:type id: int
		:param rack_size: Size of the letter rack
		:type rack_size: int
		:param score: Player's score
		:type score: int
		"""

		self.id = id
		self.rack = []
		self.rack_size = rack_size
		self.score = score


	def draw_tiles(
			self, 
			tile_bag: TileBag, 
			n: int = None
	) -> None: 
		"""
		Drawing tiles from the bag and saving the player's rack
		
		:param self: Player object
		:param tile_bag: The game's bag of tiles
		:type tile_bag: TileBag
		:param n: Number of tiles to draw. If None, the n is computed to fill the rack
		:type n: Union[int, None]
		"""

		rack_copy = self.rack.copy()
		if n is None: # Compute the number needed to fill the rack
			n = self.rack_size - len(rack_copy)

		# Drawing tiles and placing into rack
		drawn_tiles = tile_bag.draw(n)
		rack_copy.extend(drawn_tiles)

		# Re-assigning rack at the end
		self.rack = rack_copy

	def use_tiles(
			self, 
			letters: list[str], 
			track_blanks: bool = True
	) -> list[bool] | None: 
		"""
		Take tiles from the rack to amke the attempted word, using blank tiles if needed. 
		
		:param self: Player object
		:param letters: List of letters trying to use
		:type letters: list[str]
		:param track_blanks: Boolean to track blanks used or not
		:type track_blanks: bool
		:return: List of booleans of where blanks were used
		:rtype: list[bool] | None
		"""

		rack_copy = self.rack.copy()
		blank_usage = []

		for letter in letters: 
			if letter in rack_copy: # Use letter if in the rack
				rack_copy.remove(letter)
				if track_blanks: 
					blank_usage.append(False)
			
			elif '_' in rack_copy: # Look for blanks if letter not in rack
				rack_copy.remove('_')
				if track_blanks: 
					blank_usage.append(True)
			
			else: # Don't have the letter or any more blanks
				raise ValueError(f"Cannot use '{letter}'")
			
		self.rack = rack_copy # Don't want to update rack unless word can be formed
			
		return blank_usage if track_blanks else None
	
	def add_score(
			self, 
			points: int
		) -> None: 
		"""
		Add points to the player's score
		
		:param self: Player object
		:param points: Points to add
		:type points: int
		"""

		if points < 0: 
			raise ValueError("Cannot subtract points in this game")
		
		self.score += points

	def rack_to_array(self) -> np.ndarray: 
		"""
		Converts the rack to an array for neural network input (count-based). The array length is 27 (english alphabet + blank space). Values in each index are the number of tiles of that letter or blank space. 
		
		:param self: Player object
		:return: Numerical array for neural network input
		:rtype: ndarray
		"""

		array = np.zeros(27, dtype= np.int32)

		for tile in self.rack:
			if tile is not None:
				if tile == '_':
					# blank tile goes in last position
					array[26] += 1
				else:
					# Use uppercase to match board convention
					array[ord(tile) - ord('A')] += 1

		return array
 
