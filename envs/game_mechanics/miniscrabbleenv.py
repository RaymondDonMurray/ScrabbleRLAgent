import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game_components.board import Board
from game_components.tile_bag import TileBag
from game_components.dictionary import Dictionary
from game_mechanics.scorer import Scorer
from game_mechanics.player import Player

class MiniScrabbleEnv(gym.Env): 

	def __init__(
			self, 
			dictionary_path: str = None, 
			board_size: int = 5, 
			rack_size: int = 5
	): 
		"""
		Making the scrabble environment

		:param self: Environment object
		:param dictionary_path: Path to list of words, None by default
		:type dictionary_path: str
		:param board_size: Length of the sides of the square board
		:type board_size: int
		:param rack_size: Size of the players' tile racks
		:type rack_size: int
		"""
		
		super().__init__() # Initializing the gym base environment

		# Loading the dictionary
		if dictionary_path: 
			try:
				with open(dictionary_path, 'r') as f:
					words = [line.strip().upper() for line in f if line.strip()]
				self.dictionary = Dictionary(words)
			except FileNotFoundError:
				print(f"Warning: Could not find {dictionary_path}, using default word list")
				self.dictionary = Dictionary()
		else: 
			self.dictionary = Dictionary()

		# Storing game parameters / Mechanics
		self.board_size = board_size
		self.rack_size = rack_size
		self.scorer = Scorer()
		self.board = None
		self.tile_bag = None
		self.players = None
		self.current_player_idx = None
		self.is_first_move = None
		self.consecutive_passes = None

		# Action tracking
		self.valid_actions = []
		self.action_to_move = {}

		# Action space
		max_actions = 1 + (self.board_size ** 2) * 2 * len(self.dictionary.words)
		self.action_space = spaces.Discrete(max_actions)

		# Observation space
		self.observation_space = spaces.Dict({
              'board': spaces.Box(
                  low=0,
                  high=26,
                  shape=(self.board_size, self.board_size),
                  dtype=np.int32
              ),
              'rack': spaces.Box(
                  low=0,
                  high=self.rack_size,
                  shape=(27,),  # 26 letters + 1 blank
                  dtype=np.int32
              ),
              'action_mask': spaces.Box(
                  low=0,
                  high=1,
                  shape=(self.action_space.n,),  # One per action
                  dtype=np.int8  # Just 0 or 1, so int8 is efficient
              ),
              'score_self': spaces.Box(
                  low=0,
                  high=1000,
                  shape=(),  # Scalar
                  dtype=np.int32
              ),
              'score_opp': spaces.Box(
                  low=0,
                  high=1000,
                  shape=(),  # Scalar
                  dtype=np.int32
              )
          })
		
	def reset(
			self, 
			seed: int = None, 
			options = None
	) -> tuple: 
		"""
		Resetting the game environment
		
		:param self: Environment object
		:param seed: Random seed to use. None by default
		:type seed: int
		:param options: Specific options to set upon reset (not implemented yet)
		:return: Description
		:rtype: tuple
		"""

		super().reset(seed=seed)
		if seed: # Setting random seed if passed
			np.random.seed(seed)

		# Creating the new board
		self.board = Board(
			self.dictionary, 
			size= self.board_size
		)

		# New tile bag
		self.tile_bag = TileBag(
			use_simplified= True
		)

		# Two new players
		self.players = [
			Player(id= 0, rack_size= self.rack_size), 
			Player(id= 1, rack_size= self.rack_size), 	  
		]

		# Players are drawing initial tiles
		for player in self.players: 
			player.draw_tiles(self.tile_bag)

		# Setting game state
		self.current_player_idx = 0
		self.is_first_move = True
		self.consecutive_passes = 0

		# First player's valid actions
		self._generate_valid_actions()

		# Making the observation dictionary
		observation = self._get_observation()

		# Making the info dictionary
		info = {
			'player_id': self.current_player_idx, 
			'valid_action_count': len(self.valid_actions)
		}

		return (observation, info)
	
	def _generate_valid_actions(
			self
	) -> None:
		"""
		Iterating through all possible actions, making lists of all the possible valid actions
		
		:param self: Environment object
		"""
		
		# Clearing old action space
		self.valid_actions = []
		self.action_to_move = {}

		action_id = 0
		self.valid_actions.append(0) # Always have a 'PASS' option
		self.action_to_move[0] = 'PASS'
		action_id = 1

		# Getting the current player's rack
		current_player: Player = self.players[self.current_player_idx]
		player_rack = current_player.rack

		# Iterating through all possible words in all possible orientations/locations
		for row in range(self.board_size): 
			for col in range(self.board_size): 
				for direction in ['H', 'V']: 
					for word in self.dictionary.words: 
						
						# word boundary check
						if direction == 'H' and col + len(word) > self.board_size: continue
						if direction == 'V' and row + len(word) > self.board_size: continue

						# Actual validation
						valid, reason = self.board._can_place_word(
							word, row, col, direction, player_rack, self.is_first_move)
						if valid: 
							self.valid_actions.append(action_id)
							self.action_to_move[action_id] = (word, row, col, direction)
							action_id += 1


	def step(
			self,
			action: int
	) -> tuple: 
		"""
		Perform all actions necessary to move the game from one state to the next

		:param self: Game environment
		:param action: Action ID (integer index into action space)
		:type action: int
		:return: State of the game (observation, reward, terminated, truncated, info)
		:rtype: tuple
		"""
		
		# Initial check
		if action not in self.valid_actions: 
			return(self._get_observation(), -10, True, False, {'error': 'Invalid action'})
		
		# Initializing outputs
		reward = 0.0
		terminated = False
		truncated = False
		info = {}

		# Getting the action to perform
		action_perform = self.action_to_move[action]
		if action_perform == 'PASS': 
			self.consecutive_passes += 1
			if self.consecutive_passes >= 2: # Checkign if game ends due to both players passing
				terminated = True
				reward = self._compute_final_reward()
				info['termination_reason'] = 'both_passed'
			else: 
				reward = 0 # No reward for passing if game continues

		else: # Action is a tuple indicating word info
			word = action_perform[0]
			row_start = action_perform[1]
			col_start = action_perform[2]
			direction = action_perform[3]

			if direction == 'H': # Word is horizontal, row stays constant, iterate through columns
				idx_static = 0
				val_static = row_start
				idx_dynamic = 1
				val_dynamic = col_start
			elif direction == 'V': # Column stays static, iterate through tows
				idx_static = 1
				val_static = col_start
				idx_dynamic = 0
				val_dynamic = row_start
			else: 
				raise ValueError(f"Direction must be either 'V' or 'H', not {direction}")
			

			# Determining which letters were placed on the board
			letters_placed = []
			for letter in word: 
				letter_position = [None, None]
				letter_position[idx_static] = val_static
				letter_position[idx_dynamic] = val_dynamic

				# Checking if the board currently has a letter in that position
				if self.board.is_empty(letter_position[0], letter_position[1]): # board is empty in this position, add letter to letters_placed
					letters_placed.append((letter_position[0], letter_position[1], letter))

			# Placing the word on the board
			success = self.board.place_word(word, row_start, col_start, direction)
			if not success: 
				raise ValueError("Could not place word on board, check action generation methods")
			
			# getting the score for the word
			score = self.scorer.score_word_placement(
				self.board, word, row_start, col_start, direction, letters_placed, self.rack_size)
			reward = score / 10
			
			
			# Updating the player's state
			current_player: Player = self.players[self.current_player_idx]
			current_player.use_tiles([letter for _, _, letter in letters_placed])
			current_player.add_score(score)
			current_player.draw_tiles(self.tile_bag)

			# End of turn resets
			self.consecutive_passes = 0
			self.is_first_move = False
			if self._is_game_over(): 
				terminated = True
				reward = self._compute_final_reward()
				info['termination_reason'] = 'game_over'

			# Switching to next player
			self.current_player_idx = 1 - self.current_player_idx
			self._generate_valid_actions()
			

			# Adding more debugging info to the info dict
			info['player_id'] = self.current_player_idx
			info['valid_action_count'] = len(self.valid_actions)
			info['move'] = action_perform
		
		return (self._get_observation(), reward, terminated, truncated, info)


	def _get_observation(
			self
		) -> dict: 
		"""
		Getting the agent's observation space
		
		:param self: Environment object
		:return: The observation space the agent can know
		:rtype: dict
		"""

		# Getting the players
		current_player: Player = self.players[self.current_player_idx]
		opponent: Player = self.players[1 - self.current_player_idx]

		# board + rack info
		board_array = self.board.to_array()
		rack_array = current_player.rack_to_array()

		# Action mask
		action_mask = np.zeros(self.action_space.n, dtype=np.int8)
		action_mask[self.valid_actions] = 1  # Mark valid actions as 1

		# Building the return dict
		observation = {
			'board': board_array, 
			'rack': rack_array, 
			'action_mask': action_mask, 
			'score_self': np.int32(current_player.score), 
			'score_opp': np.int32(opponent.score)
		}

		return observation
	
	def _is_game_over(
			self
	) -> bool: 
		"""
		Checks if the game is over
		
		:param self: Environment
		:return: Boolean if game is over or not
		:rtype: bool
		"""
		
		if self.consecutive_passes >= 2: # both players passed 
			return True
		elif self.tile_bag.is_empty(): # Check players' racks if tile bag is empty
			empty_racks = []
			for player in self.players: 
				empty_racks.append(
					all(tile is None for tile in player.rack) 
					or len(player.rack) == 0
					)

			return all(empty_racks)
		else: 
			return False
		

	def render(
			self, 
			mode: str = 'human'
	) -> None: 
		
		print(f"MINI-SCRABBLE - Player {self.current_player_idx}'s turn")
		print("Board:")
		print(str(self.board))
		print("Player info:")
		for player in self.players: 
			if player.id == self.current_player_idx: 
				indicator = ">>> "
			else: 
				indicator = ""

			print(indicator + f"Player {player.id}, {player.score} Points")
			print(f"Rack: {player.rack}")

		print("Game State:")
		print(f"Tiles Remaining: {self.tile_bag.get_number_of_remaining_tiles()}")
		print(f"First Move: {'YES' if self.is_first_move else 'NO'}")		
		print(f"Number of viable actions: {len(self.valid_actions)}")


	def _compute_final_reward(
			self
	) -> float: 
		"""
		Computes the player's final score
		
		:param self: Game environment
		:return: Final reward
		:rtype: float
		"""

		# Setting player references (THESE ARE NOT COPIES, BUT POINTERS TO ORIGINALS)
		current_player: Player = self.players[self.current_player_idx]
		opponent: Player = self.players[1 - self.current_player_idx]

		# Applying end-game scoring
		# The sum of the point values of the tiles still in the player's rack is subtracted from their current score
		for player in self.players: 
			remaining_value = sum(self.scorer.letter_scores[tile.lower()] for tile in player.rack if tile is not None)
			player.score -= remaining_value

		# Appropriately rewarding players based on if they won or lost
		score_diff = current_player.score - opponent.score
		if score_diff > 0: 
			reward = 10.0 + score_diff / 10.0
		elif score_diff < 0: 
			reward = -10.0 + score_diff / 10
		else: 
			reward = 0.0

		return reward
		