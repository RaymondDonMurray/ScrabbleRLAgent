from game_components.board import Board


DEFAULT_LETTER_SCORES = {
	'a': 1, 'b': 3, 'c': 3, 'd': 2,
    'e': 1, 'f': 4, 'g': 2, 'h': 4,
    'i': 1, 'j': 8, 'k': 5, 'l': 1,
    'm': 3, 'n': 1, 'o': 1, 'p': 3,
    'q': 10, 'r': 1, 's': 1, 't': 1,
    'u': 1, 'v': 4, 'w': 4, 'x': 8,
    'y': 4, 'z': 10, '_': 0
}

class Scorer: 

	def __init__(
			self, 
			letter_scores: dict[str, int] = DEFAULT_LETTER_SCORES, 
		):
		"""
		Making the scorer object
		
		:param self: Scorer object
		:param letter_scores: Numerical scores for each letter
		:type letter_scores: dict[str, int]
		"""

		self.letter_scores = letter_scores

	def score_word_placement(
			self, 
			board: Board, 
			word: str, 
			row: int, 
			col: int, 
			direction: str, 
			letters_placed: list[tuple[int, int, str]]
	) -> int:
		"""
		Docstring for score_word_placement
		
		:param self: Scorer object
		:param board: Game board object
		:type board: Board
		:param word: Word formed during turn
		:type word: str
		:param row: Starting row coordinate
		:type row: int
		:param col: Starting column coordinate
		:type col: int
		:param direction: Direction of the word. Either 'H' (horizontal) or 'V' vertical
		:type direction: str
		:param letters_placed: The letters actually placed by the player. Letters already on the board do not get scored. Letters are stored in the form (row, col, letter)
		:type letters_placed: list[tuple[int, int, str]]
		:return: Score to add
		:rtype: int
		"""
		