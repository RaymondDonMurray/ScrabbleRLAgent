class Dictionary: 
	
	def __init__(self, word_list=None, file_path=None):
		"""
		Initialize dictionary from word list or file.
		
		:param word_list: List of words (optional)
		:param file_path: Path to word file (optional)
		"""
		self.words = set()

		if word_list is not None:
			self.words = set(word.strip().upper() for word in word_list if word.strip())
		elif file_path is not None:
			self._load_from_file(file_path)
		else:
			# Default minimal word list for testing
			self.words = {'CAT', 'DOG', 'HAT', 'BAT', 'RAT', 'MAT'}

	def _load_from_file(self, file_path):
		"""Load words from file (private method)."""
		try:
			with open(file_path, 'r') as f:
				self.words = set(
					line.strip().upper()
					for line in f
					if line.strip()
				)
		except FileNotFoundError:
			print(f"Warning: Word file not found: {file_path}")
			self.words = set()

	def is_valid_word(self, word: str) -> bool:
		"""Check if a word is in the dictionary."""
		return word.upper() in self.words