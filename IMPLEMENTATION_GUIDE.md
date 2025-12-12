# Scrabble RL Agent - Detailed Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the early stages of the Scrabble RL Agent project. Each section includes detailed descriptions of what to build, why it matters, and how to code it up.

---

## Part 1: Environment Foundation (Days 1-3)

Before any reinforcement learning happens, you need a working Scrabble game. This is the most critical part - if the environment is buggy, everything built on top will fail.

### 1.1 Dictionary and Word Validation

**What it is:** A data structure that stores valid words and can quickly check if a word is valid.

**Why it matters:** The environment needs to validate every word placed on the board. This happens thousands of times during training, so it must be fast and correct.

**Implementation approach:**

```python
class Dictionary:
    """Manages the dictionary of valid words for the game."""

    def __init__(self, word_list):
        # Store words in a set for O(1) lookup
        self.valid_words = set(word.upper() for word in word_list)

        # Optional: Build a trie for prefix checking (helpful later)
        # This lets you check "is 'QUA' a valid prefix?" efficiently
        self.trie = self._build_trie(self.valid_words)

    def is_valid_word(self, word):
        """Check if a word exists in the dictionary."""
        return word.upper() in self.valid_words

    def is_valid_prefix(self, prefix):
        """Check if any word starts with this prefix."""
        # Using trie: O(len(prefix))
        # Without trie: iterate through all words (slow)
        pass

    def get_words_by_length(self, length):
        """Return all words of a specific length."""
        return [w for w in self.valid_words if len(w) == length]
```

**What to code:**
1. Create a file `dictionary.py` with the `Dictionary` class
2. For Phase 1, start with a simple list of 500-1000 common 3-5 letter words (you can hardcode these or load from a text file)
3. Implement `is_valid_word()` using a Python set (O(1) lookup)
4. Skip the trie for now - implement it only if word generation becomes too slow
5. Write tests: check that known words return True, gibberish returns False

**Key decisions:**
- Start with a small dictionary (500-1000 words) for mini-Scrabble
- All words uppercase to avoid case issues
- Don't worry about performance optimization yet

---

### 1.2 Tile Bag and Letter Management

**What it is:** A bag containing all letter tiles, tracking which tiles remain available.

**Why it matters:** Players draw tiles from the bag, and the game ends when the bag is empty and one player uses all tiles. The tile distribution affects game strategy.

**Implementation approach:**

```python
class TileBag:
    """Manages the collection of available tiles."""

    def __init__(self, use_simplified=True):
        if use_simplified:
            # Simplified distribution for mini-Scrabble
            # Fewer tiles, simpler counts
            self.tiles = self._create_simplified_tiles()
        else:
            # Full Scrabble distribution (use later)
            self.tiles = self._create_standard_tiles()

        self.remaining = self.tiles.copy()

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

    def draw(self, n=1):
        """Draw n tiles from the bag."""
        import random

        if n > len(self.remaining):
            n = len(self.remaining)

        drawn = random.sample(self.remaining, n)
        for tile in drawn:
            self.remaining.remove(tile)

        return drawn

    def is_empty(self):
        """Check if bag is empty."""
        return len(self.remaining) == 0

    def tiles_remaining(self):
        """Get count of remaining tiles."""
        return len(self.remaining)
```

**What to code:**
1. Create `tile_bag.py` with the `TileBag` class
2. Define a simplified tile distribution (maybe 30-40 total tiles for mini-Scrabble)
3. Implement `draw()` to randomly select tiles and remove them from the bag
4. Add `is_empty()` to check game end condition
5. Consider tile values (A=1 point, Q=10 points, etc.) - store as a separate dict
6. Write tests: draw all tiles, verify counts, check randomness

**Key decisions:**
- Use simplified distribution (fewer tiles than standard Scrabble)
- For mini-Scrabble, maybe 30-40 total tiles instead of 100
- Include 1-2 blank tiles (wildcards)
- Standard tile values to start: common letters = 1-2 pts, rare letters = 8-10 pts

---

### 1.3 Board Representation and State

**What it is:** A 2D grid representing the board, tracking which letters are placed where, and which squares have premium bonuses.

**Why it matters:** The board is the core game state. The RL agent observes the board to decide what move to make. It must be efficiently represented for neural network input.

**Implementation approach:**

```python
class Board:
    """Represents the Scrabble board state."""

    def __init__(self, size=5):
        self.size = size

        # Main board: 2D array of characters
        # None = empty square, 'A' = letter A placed there
        self.grid = [[None for _ in range(size)] for _ in range(size)]

        # Premium squares: 2D array of bonuses
        # 'DL' = double letter, 'DW' = double word, None = normal
        self.premium_squares = self._initialize_premium_squares()

        # Track which squares have been used (for premium scoring)
        # Premiums only apply on first use
        self.premium_used = [[False for _ in range(size)] for _ in range(size)]

    def _initialize_premium_squares(self):
        """Set up premium square layout for mini-Scrabble."""
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

    def place_word(self, word, row, col, direction):
        """
        Place a word on the board.

        Args:
            word: The word to place (string)
            row, col: Starting position
            direction: 'H' (horizontal) or 'V' (vertical)

        Returns:
            True if successful, False if invalid placement
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

    def get_letter(self, row, col):
        """Get the letter at a position (or None if empty)."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.grid[row][col]
        return None

    def is_empty(self, row, col):
        """Check if a square is empty."""
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
```

**What to code:**
1. Create `board.py` with the `Board` class
2. Initialize a 5x5 2D array (list of lists or numpy array)
3. Define premium square positions (hardcode a simple pattern)
4. Implement `place_word()` - places letters on the board
5. Implement `to_array()` - converts board to neural network input format
6. Add `__str__()` for pretty printing during debugging
7. Write tests: place words, check positions, verify array conversion

**Key decisions:**
- Use None for empty squares, single characters for placed letters
- Premium squares: keep it simple (1 double word, 4-6 double letters)
- Encoding: 0=empty, 1-26=A-Z, or use one-hot encoding (decide based on NN architecture)

---

### 1.4 Word Placement Validation

**What it is:** Logic to check if a proposed word placement is legal according to Scrabble rules.

**Why it matters:** This is the most complex part of the environment. Invalid moves must be rejected. The validation must catch all edge cases.

**Scrabble placement rules to implement:**
1. **First move**: Must pass through center square
2. **Subsequent moves**: Must connect to at least one existing letter
3. **Bounds**: Word must fit within board boundaries
4. **Overlap**: Can overlap existing letters only if they match
5. **Perpendicular words**: Any perpendicular words formed must be valid
6. **Player rack**: Player must have the required letters in their rack

**Implementation approach:**

```python
def _can_place_word(self, word, row, col, direction, player_rack, is_first_move):
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
```

**What to code:**
1. Add `_can_place_word()` method to `Board` class
2. Implement each validation check separately (easier to debug)
3. Check boundaries first (simplest)
4. Check center square for first move
5. Check adjacency for subsequent moves
6. Check letter availability in rack
7. Check perpendicular word validity (most complex)
8. Write extensive tests for each rule:
   - Valid placements should pass
   - Each invalid case should fail with correct reason

**Key decisions:**
- Return both boolean and reason string (helpful for debugging)
- Validate incrementally - fail fast on easy checks
- Perpendicular word checking is tricky - test thoroughly

---

## Part 2: Game Mechanics (Days 4-5)

Now that you have the board and validation, implement the actual game flow.

### 2.1 Player State and Rack Management

**What it is:** Each player has a rack (hand) of tiles and a score. The rack needs to be refilled after each turn.

**Implementation approach:**

```python
class Player:
    """Represents a player in the game."""

    def __init__(self, player_id, rack_size=5):
        self.player_id = player_id
        self.rack_size = rack_size
        self.rack = []  # List of letter characters
        self.score = 0

    def draw_tiles(self, tile_bag, n=None):
        """Draw tiles from bag to fill rack."""
        if n is None:
            n = self.rack_size - len(self.rack)

        new_tiles = tile_bag.draw(n)
        self.rack.extend(new_tiles)

    def use_tiles(self, letters):
        """
        Remove used letters from rack.

        Args:
            letters: List of letters to remove
        """
        for letter in letters:
            if letter in self.rack:
                self.rack.remove(letter)
            else:
                # Check for blank
                if '_' in self.rack:
                    self.rack.remove('_')

    def add_score(self, points):
        """Add points to player's score."""
        self.score += points

    def rack_to_array(self):
        """Convert rack to numerical array for NN input."""
        import numpy as np

        # One-hot encoding: 27 dimensions (26 letters + blank)
        rack_array = np.zeros(27, dtype=np.int32)
        for letter in self.rack:
            if letter == '_':
                rack_array[26] += 1
            else:
                idx = ord(letter) - ord('A')
                rack_array[idx] += 1

        return rack_array
```

**What to code:**
1. Create `player.py` with `Player` class
2. Initialize with empty rack and score=0
3. Implement `draw_tiles()` to refill rack from tile bag
4. Implement `use_tiles()` to remove played letters
5. Add `rack_to_array()` for neural network observation
6. Write tests: draw tiles, use tiles, check scoring

---

### 2.2 Scoring System

**What it is:** Calculate the score for a word placement, including letter values, premium squares, and bonuses.

**Scrabble scoring rules:**
1. Each letter has a point value
2. Double/triple letter squares multiply that letter's value
3. Double/triple word squares multiply the entire word's value
4. Premium squares only count on first use
5. Using all tiles in your rack = +10 bonus ("bingo")

**Implementation approach:**

```python
class Scorer:
    """Handles score calculation for word placements."""

    def __init__(self):
        # Standard Scrabble letter values (can simplify for Phase 1)
        self.letter_values = {
            'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1, 'L': 1, 'N': 1,
            'R': 1, 'S': 1, 'T': 1,
            'D': 2, 'G': 2,
            'B': 3, 'C': 3, 'M': 3, 'P': 3,
            'F': 4, 'H': 4, 'V': 4, 'W': 4, 'Y': 4,
            'K': 5,
            'J': 8, 'X': 8,
            'Q': 10, 'Z': 10,
            '_': 0  # Blank tile
        }

    def score_word_placement(self, board, word, row, col, direction,
                            letters_placed):
        """
        Calculate score for placing a word.

        Args:
            board: The Board object
            word: The word being placed
            row, col: Starting position
            direction: 'H' or 'V'
            letters_placed: List of (row, col, letter) for new letters

        Returns:
            int: Total score for this placement
        """
        word_multiplier = 1
        word_score = 0

        # Score each letter in the main word
        for i, letter in enumerate(word):
            r = row if direction == 'H' else row + i
            c = col + i if direction == 'H' else col

            letter_score = self.letter_values[letter]
            letter_multiplier = 1

            # Apply premium if this is a newly placed letter
            if (r, c, letter) in letters_placed:
                premium = board.premium_squares[r][c]
                if not board.premium_used[r][c]:
                    if premium == 'DL':
                        letter_multiplier = 2
                    elif premium == 'TL':
                        letter_multiplier = 3
                    elif premium == 'DW':
                        word_multiplier *= 2
                    elif premium == 'TW':
                        word_multiplier *= 3

                    board.premium_used[r][c] = True

            word_score += letter_score * letter_multiplier

        word_score *= word_multiplier

        # Add scores for perpendicular words formed
        perp_score = 0
        for r, c, letter in letters_placed:
            # Check if this letter forms a perpendicular word
            if direction == 'H':
                perp_word = board._extract_vertical_word(r, c, letter)
            else:
                perp_word = board._extract_horizontal_word(r, c, letter)

            if perp_word and len(perp_word[0]) > 1:
                perp_score += self._score_perpendicular_word(
                    board, perp_word, (r, c, letter)
                )

        total_score = word_score + perp_score

        # Bingo bonus: used all tiles in rack
        if len(letters_placed) == 5:  # rack_size
            total_score += 10

        return total_score
```

**What to code:**
1. Create `scorer.py` with `Scorer` class
2. Define letter values (use simplified values to start)
3. Implement `score_word_placement()`:
   - Sum letter values
   - Apply letter multipliers (DL, TL)
   - Apply word multipliers (DW, TW)
   - Add perpendicular word scores
   - Add bingo bonus if applicable
4. Track which premium squares have been used
5. Write tests: score simple words, words with premiums, bingo bonus

---

### 2.3 Putting It All Together: The Game Environment

**What it is:** A Gymnasium-compatible environment that orchestrates all the components above into a playable game.

**Why it matters:** This is what the RL agent interacts with. It must follow the Gym API (reset, step, render).

**Implementation approach:**

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MiniScrabbleEnv(gym.Env):
    """
    Mini-Scrabble environment for reinforcement learning.

    Observation space:
        Dict with:
        - 'board': (5, 5) array of integers (0=empty, 1-26=letters)
        - 'rack': (27,) array of counts for each letter type
        - 'action_mask': (action_space_size,) boolean array of valid actions
        - 'score_self': int, current player's score
        - 'score_opp': int, opponent's score

    Action space:
        Discrete(N) where N = number of possible word placements
        Action encoding: Flatten (row, col, direction, word_index)
    """

    def __init__(self, dictionary_path=None, board_size=5):
        super().__init__()

        self.board_size = board_size

        # Initialize components
        self.dictionary = Dictionary(self._load_dictionary(dictionary_path))
        self.board = Board(size=board_size)
        self.tile_bag = TileBag(use_simplified=True)
        self.scorer = Scorer()

        self.players = [
            Player(player_id=0, rack_size=5),
            Player(player_id=1, rack_size=5)
        ]
        self.current_player_idx = 0

        # Game state
        self.is_first_move = True
        self.consecutive_passes = 0

        # Define observation space
        self.observation_space = spaces.Dict({
            'board': spaces.Box(
                low=0, high=26, shape=(board_size, board_size), dtype=np.int32
            ),
            'rack': spaces.Box(
                low=0, high=5, shape=(27,), dtype=np.int32
            ),
            'action_mask': spaces.Box(
                low=0, high=1, shape=(self._compute_action_space_size(),),
                dtype=np.bool_
            ),
            'score_self': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            'score_opp': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
        })

        # Define action space
        self.action_space = spaces.Discrete(self._compute_action_space_size())

        # Action mapping (populated during valid action generation)
        self.action_to_move = {}  # action_id -> (row, col, direction, word)
        self.valid_actions = []

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset all components
        self.board = Board(size=self.board_size)
        self.tile_bag = TileBag(use_simplified=True)
        self.players = [
            Player(player_id=0, rack_size=5),
            Player(player_id=1, rack_size=5)
        ]

        # Draw initial tiles
        for player in self.players:
            player.draw_tiles(self.tile_bag)

        self.current_player_idx = 0
        self.is_first_move = True
        self.consecutive_passes = 0

        # Generate valid actions
        self._generate_valid_actions()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Integer action ID

        Returns:
            observation, reward, terminated, truncated, info
        """
        current_player = self.players[self.current_player_idx]
        opponent = self.players[1 - self.current_player_idx]

        # Decode action
        if action >= len(self.valid_actions):
            # Invalid action - penalize and end episode
            reward = -10.0
            terminated = True
            obs = self._get_observation()
            info = {'invalid_action': True}
            return obs, reward, terminated, False, info

        move = self.action_to_move[action]

        # Special case: pass action
        if move == 'PASS':
            self.consecutive_passes += 1
            reward = 0.0

            if self.consecutive_passes >= 2:
                # Both players passed - game over
                terminated = True
            else:
                terminated = False

            # Switch player
            self.current_player_idx = 1 - self.current_player_idx
            self._generate_valid_actions()

            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, terminated, False, info

        # Regular move: place word
        row, col, direction, word = move

        # Place word on board
        letters_placed = self._place_word_and_track(word, row, col, direction)

        # Calculate score
        score = self.scorer.score_word_placement(
            self.board, word, row, col, direction, letters_placed
        )

        # Update player
        current_player.use_tiles([letter for _, _, letter in letters_placed])
        current_player.add_score(score)
        current_player.draw_tiles(self.tile_bag)

        # Reset consecutive passes
        self.consecutive_passes = 0
        self.is_first_move = False

        # Calculate reward (score differential or just score)
        reward = score / 10.0  # Normalize

        # Check if game is over
        terminated = self._is_game_over()

        if terminated:
            # Final scoring adjustments
            if len(current_player.rack) == 0:
                # Subtract opponent's remaining tiles from opponent
                leftover = sum(self.scorer.letter_values.get(l, 0)
                              for l in opponent.rack)
                opponent.score -= leftover
                current_player.score += leftover

            # Bonus reward for winning
            if current_player.score > opponent.score:
                reward += 10.0
            elif current_player.score < opponent.score:
                reward -= 10.0

        # Switch player
        self.current_player_idx = 1 - self.current_player_idx

        # Generate valid actions for next player
        self._generate_valid_actions()

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, False, info

    def _generate_valid_actions(self):
        """
        Generate all valid actions for the current player.

        This is the computationally intensive part.
        """
        self.valid_actions = []
        self.action_to_move = {}
        action_id = 0

        current_player = self.players[self.current_player_idx]

        # Always allow passing
        self.valid_actions.append(action_id)
        self.action_to_move[action_id] = 'PASS'
        action_id += 1

        # Generate all possible word placements
        for row in range(self.board_size):
            for col in range(self.board_size):
                for direction in ['H', 'V']:
                    # Try words from dictionary
                    for word in self.dictionary.valid_words:
                        # Quick check: does word fit?
                        if direction == 'H' and col + len(word) > self.board_size:
                            continue
                        if direction == 'V' and row + len(word) > self.board_size:
                            continue

                        # Detailed validation
                        is_valid, _ = self.board._can_place_word(
                            word, row, col, direction,
                            current_player.rack, self.is_first_move
                        )

                        if is_valid:
                            self.valid_actions.append(action_id)
                            self.action_to_move[action_id] = (row, col, direction, word)
                            action_id += 1

        # If no valid moves, must pass
        if len(self.valid_actions) == 1:  # Only PASS
            pass  # That's OK, passing is valid

    def _get_observation(self):
        """Build observation dict for current player."""
        current_player = self.players[self.current_player_idx]
        opponent = self.players[1 - self.current_player_idx]

        # Create action mask
        action_mask = np.zeros(self.action_space.n, dtype=np.bool_)
        for action_id in self.valid_actions:
            action_mask[action_id] = True

        obs = {
            'board': self.board.to_array(),
            'rack': current_player.rack_to_array(),
            'action_mask': action_mask,
            'score_self': np.array([current_player.score], dtype=np.int32),
            'score_opp': np.array([opponent.score], dtype=np.int32),
        }

        return obs

    def _is_game_over(self):
        """Check if game has ended."""
        # Game ends if:
        # 1. Tile bag empty and one player used all tiles
        # 2. Both players passed consecutively

        if self.consecutive_passes >= 2:
            return True

        if self.tile_bag.is_empty():
            for player in self.players:
                if len(player.rack) == 0:
                    return True

        return False

    def render(self):
        """Print current game state (for debugging)."""
        print("\n" + "="*40)
        print("MINI SCRABBLE")
        print("="*40)
        print(self.board)
        print()
        print(f"Player 0: Score={self.players[0].score}, Rack={self.players[0].rack}")
        print(f"Player 1: Score={self.players[1].score}, Rack={self.players[1].rack}")
        print(f"Tiles remaining: {self.tile_bag.tiles_remaining()}")
        print(f"Current player: {self.current_player_idx}")
        print(f"Valid actions: {len(self.valid_actions)}")
        print("="*40)
```

**What to code:**
1. Create `mini_scrabble_env.py` with `MiniScrabbleEnv` class
2. Inherit from `gym.Env`
3. Define observation and action spaces
4. Implement `reset()` - initializes game
5. Implement `step()` - executes one action
6. Implement `_generate_valid_actions()` - finds all legal moves
7. Implement `_get_observation()` - builds observation dict
8. Implement `render()` - prints game state (for debugging)
9. Test: play random vs random for 100 games

---

## Part 3: Testing and Baseline Agents (Days 6-7)

Before implementing RL, verify everything works with simple agents.

### 3.1 Random Agent

**What it is:** Agent that picks uniformly from valid actions.

**Implementation:**

```python
def random_agent(observation):
    """Select a random valid action."""
    action_mask = observation['action_mask']
    valid_actions = np.where(action_mask)[0]
    return np.random.choice(valid_actions)

# Test
env = MiniScrabbleEnv()
obs, info = env.reset()

for _ in range(100):  # 100 games
    terminated = False
    while not terminated:
        action = random_agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    env.render()  # Print final state
    env.reset()
```

**Expected results:** Games should complete without errors. Scores will be low (most moves will be short words).

---

### 3.2 Greedy Agent

**What it is:** Agent that picks the highest-scoring valid move.

**Implementation:**

```python
def greedy_agent(env, observation):
    """Select the highest-scoring valid action."""
    action_mask = observation['action_mask']
    valid_action_ids = np.where(action_mask)[0]

    best_action = None
    best_score = -1

    for action_id in valid_action_ids:
        move = env.action_to_move[action_id]

        if move == 'PASS':
            score = 0
        else:
            row, col, direction, word = move
            # Simulate scoring (without actually placing)
            score = env.scorer.score_word_placement(
                env.board, word, row, col, direction,
                env._get_letters_to_place(word, row, col, direction)
            )

        if score > best_score:
            best_score = score
            best_action = action_id

    return best_action
```

**Expected results:** Greedy should beat random consistently (>80% win rate). Scores should be higher.

---

## Part 4: What's Next?

After completing Parts 1-3, you'll have:
- A working mini-Scrabble environment
- Validated game mechanics
- Baseline agents for comparison

**Next steps:**
1. Implement PPO agent (Part 5 - separate document)
2. Set up training loop with self-play
3. Add logging and visualization
4. Train and evaluate

---

## Key Takeaways

1. **Environment correctness is critical** - spend time getting rules right
2. **Test incrementally** - test each component before moving on
3. **Start simple** - don't optimize prematurely
4. **Validate with baselines** - random and greedy agents catch bugs
5. **Log everything** - you'll need to debug during training

---

## Common Issues and Solutions

**Issue**: Action generation is very slow
- **Solution**: Start with small dictionary (500 words), optimize later with trie

**Issue**: Valid action mask has bugs
- **Solution**: Add logging to print all generated actions, manually verify a few

**Issue**: Scoring doesn't match expected
- **Solution**: Write unit tests for each scoring rule separately

**Issue**: Games don't terminate
- **Solution**: Add max_steps limit (e.g., 30 turns), check termination conditions

**Issue**: Neural network gets NaN loss
- **Solution**: Normalize rewards, check for -inf in masked logits

---

This should give you a solid foundation for the first phase of the project. Take it step by step, test frequently, and don't hesitate to simplify if you get stuck!
