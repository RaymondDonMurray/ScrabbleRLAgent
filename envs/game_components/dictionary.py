class Dictionary: 
	
	def __init__(self): 
		
		self.words = set() # Use a set for O(1) lookup
		self.load_words()

	def load_words(self, file_path= 'envs/game_components/words_simple.txt'): 
		"""Load words from a file into the dictionary."""
		try: 
			with open(file_path, 'r') as file: 
				for line in file: 
					word = line.strip().lower() 
					if word: 
						self.words.add(word) 
		except FileNotFoundError: 
			print(f"Word list file not found: {file_path}")

	def is_valid_word(self, word: str) -> bool: 
		"""
		Docstring for is_valid_word
		
		:param self: Dictionary object
		:param word (str): word to check
		"""
		return word.lower() in self.words