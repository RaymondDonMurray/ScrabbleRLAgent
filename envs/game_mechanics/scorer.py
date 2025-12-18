from ..game_components.board import Board

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

	def _single_word_score(
			self, 
			board: Board, 
			word: str, 
			row: int, 
			col: int, 
			direction: str, 
			letters_placed: list[tuple[int, int, str]]
	) -> int:
		"""
		Score for a single word, not including perpendicular words
		
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
		:param letters_placed: The letters actually placed by the player. Letters are stored in the form (row, col, letter)
		:type letters_placed: list[tuple[int, int, str]]
		:return: Score to add
		:rtype: int
		"""

		# Initializing score parameters
		word_score = 0
		word_multiplier = 1

		# Defining which index changes with each letter
		direction = direction.upper()
		if direction == 'V': # 'row' changes with each letter
			idx_dynamic = 0 # First element in tuple is iterated
			val_dynamic = row # This value changes with each letter
			idx_static = 1 # Second elemtn in tuple is held static
			val_static = col # This value is held constant
		elif direction == 'H': # 'col' changes with each letter
			idx_dynamic = 1
			val_dynamic = col
			idx_static = 0
			val_static = row
		else: 
			raise ValueError("Direction must be either 'H' or 'V'")
		
		for letter in word:
			letter_multiplier = 1 # multiplier for the current letter

			# Making tuple of the current letter (keep uppercase for comparison)
			letter_list = [None, None, None]
			letter_list[2] = letter  # Keep uppercase
			letter_val = self.letter_scores[letter.lower()]  # Use lowercase for scoring lookup
			letter_list[idx_dynamic] = val_dynamic
			letter_list[idx_static] = val_static
			letter_tuple = tuple(letter_list)

			if letter_tuple in letters_placed: # Letter was one of the ones the player placed
				letter_row = letter_tuple[0]
				letter_col = letter_tuple[1]

				# Checking if the letter is in a premium square (either DL or DW)
				premium = board.premium_squares[letter_row][letter_col]
				premium_used = board.premium_used[letter_row][letter_col]
				if premium is not None and not premium_used: 
					match premium: 
						case 'DW': 
							word_multiplier *= 2
						case 'DL': 
							letter_multiplier = 2

					# Need to mark premium as used
					board.premium_used[letter_row][letter_col] = True

			# Adding letter score to word
			letter_score = letter_val * letter_multiplier
			word_score += letter_score

			# Iterating dynamic value
			val_dynamic += 1

		# Computing final word score
		word_score *= word_multiplier

		return word_score
				

	def score_word_placement(
			self, 
			board: Board, 
			word: str, 
			row: int, 
			col: int, 
			direction: str, 
			letters_placed: list[tuple[int, int, str]], 
			rack_size: int
	) -> int: 
		"""
		Full score generation for placed word, includin perpendicular words
		
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
		:param letters_placed: The letters actually placed by the player. Letters are stored in the form (row, col, letter)
		:type letters_placed: list[tuple[int, int, str]]
		:param rack_size: Size of the player's rack
		:type rack_size: int
		:return: Score to add
		:rtype: int
		"""
		
		# Check on initial word direction
		if direction == 'V': 
			perp_direction = 'H'
		elif direction == 'H': 
			perp_direction = 'V'
		else: 
			raise ValueError("Direction must be either 'H' or 'V'")
		
		total_score = 0
		
		# Getting score for the word actually placed
		word_placed_score = self._single_word_score(
			board, word, row, col, direction, letters_placed
			)
		total_score += word_placed_score
		
		# Adding scores for perpendicular words
		perpendicular_words = board._get_perpendicular_words(
			word, row, col, direction
		)
		for word_tuple in perpendicular_words: 
			perp_word = word_tuple[0]
			perp_row = word_tuple[1]
			perp_col = word_tuple[2]

			total_score += self._single_word_score(
				board, perp_word, perp_row, perp_col, perp_direction, letters_placed
			)

		# Checking for bingo bonus
		if len(letters_placed) == rack_size: 
			total_score += 10

		return total_score
