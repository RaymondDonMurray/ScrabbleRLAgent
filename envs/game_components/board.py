import numpy as np

class Board: 

	def __init__(self, size: int = 5): 
		"""
		Initializes the game board of a given size. 
		
		:param self: Game board object
		:param size: Tile-length of the square board
		:type size: int
		"""
		self.size = size
		self.grid = np.zeros((self.size, self.size), dtype=int) # 0 = empty, 1 = occupied
		self.premium_squares = self._init_premium_squares()

	def place_word(self, word: str, row:int, col:int, direction:str) -> int: 
		""" Places a word on the board, returns the score"""
		pass

	
