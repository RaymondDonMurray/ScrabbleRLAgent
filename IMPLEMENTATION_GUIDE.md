# Scrabble RL Agent - Detailed Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the early stages of the Scrabble RL Agent project. Each section describes what components to build, why they matter, and what each method should accomplish - but leaves the implementation details to you.

---

## Part 1: Environment Foundation (Days 1-3)

Before any reinforcement learning happens, you need a working Scrabble game. This is the most critical part - if the environment is buggy, everything built on top will fail.

### 1.1 Dictionary and Word Validation

**What it is:** A data structure that stores valid words and can quickly check if a word is valid.

**Why it matters:** The environment needs to validate every word placed on the board. This happens thousands of times during training, so it must be fast and correct.

**Class: `Dictionary`**

**Attributes to store:**
- A collection of valid words (consider what data structure gives O(1) lookup)
- Optional: A trie structure for prefix checking (implement later if needed)

**Methods to implement:**

1. **`__init__(self, word_list)`**
   - Takes a list of words as input
   - Store them in an efficient data structure for lookup
   - Consider case handling (uppercase/lowercase)

2. **`is_valid_word(self, word)`**
   - Input: A string word
   - Output: Boolean (True if word exists in dictionary)
   - Should be very fast (O(1) or close to it)

3. **`is_valid_prefix(self, prefix)`** (optional, for optimization)
   - Input: A string prefix
   - Output: Boolean (True if any word starts with this prefix)
   - Helps optimize action generation later

4. **`get_words_by_length(self, length)`** (optional, for optimization)
   - Input: Integer length
   - Output: List of all words with that length
   - Can help filter dictionary during action generation

**What to code:**
1. Create a file `dictionary.py` with the `Dictionary` class
2. For Phase 1, start with 500-1000 common 3-5 letter words from `words_simple.txt` or create your own list
3. Choose an appropriate data structure (hint: sets are fast for membership testing)
4. Implement at minimum `__init__` and `is_valid_word()`
5. Write tests: check that known words return True, gibberish returns False

**Key decisions:**
- Start with a small dictionary (500-1000 words) for mini-Scrabble
- All words uppercase to avoid case issues
- Don't worry about performance optimization yet

---

### 1.2 Tile Bag and Letter Management

**What it is:** A bag containing all letter tiles, tracking which tiles remain available.

**Why it matters:** Players draw tiles from the bag, and the game ends when the bag is empty and one player uses all tiles. The tile distribution affects game strategy.

**Class: `TileBag`**

**Attributes to store:**
- Initial tile distribution (all tiles at start of game)
- Remaining tiles (what's left in the bag)
- Letter values (A=1 point, Q=10 points, etc.)

**Methods to implement:**

1. **`__init__(self, use_simplified=True)`**
   - Create the initial tile distribution
   - For simplified: ~30-40 total tiles (suitable for mini-Scrabble)
   - For standard: ~100 tiles (implement later)
   - Initialize remaining tiles to match initial distribution

2. **`_create_simplified_tiles(self)`**
   - Returns a list of letter characters representing all tiles
   - Design your distribution: more common letters (A, E, I, O, U, R, S, T, L, N), fewer rare letters (Q, Z, X)
   - Include 1-2 blank tiles (represented as '_')
   - Total should support 2 players with 5-tile racks

3. **`draw(self, n=1)`**
   - Input: Number of tiles to draw
   - Output: List of drawn tiles (letters)
   - Randomly select n tiles from remaining tiles
   - Remove drawn tiles from the bag
   - Handle edge case: requesting more tiles than available

4. **`is_empty(self)`**
   - Output: Boolean (True if no tiles left)

5. **`tiles_remaining(self)`**
   - Output: Integer count of tiles left in bag

**What to code:**
1. Create `tile_bag.py` with the `TileBag` class
2. Design a simplified tile distribution (30-40 total tiles)
3. Implement random drawing (remove tiles after drawing)
4. Consider storing letter values in a separate dictionary or class
5. Write tests: draw all tiles, verify counts, check that drawing reduces remaining

**Key decisions:**
- Use simplified distribution (fewer tiles than standard Scrabble)
- For mini-Scrabble: 30-40 total tiles instead of 100
- Include 1-2 blank tiles (wildcards)
- Standard tile values: common letters = 1-2 pts, rare letters = 8-10 pts

---

### 1.3 Board Representation and State

**What it is:** A 2D grid representing the board, tracking which letters are placed where, and which squares have premium bonuses.

**Why it matters:** The board is the core game state. The RL agent observes the board to decide what move to make. It must be efficiently represented for neural network input.

**Class: `Board`**

**Attributes to store:**
- The main grid (5x5 2D array) - stores letter characters or None for empty
- Premium square locations (which squares have DL, DW, etc.)
- Premium squares usage tracking (premiums only apply once)
- Board size parameter

**Methods to implement:**

1. **`__init__(self, size=5)`**
   - Initialize empty grid (5x5 2D array)
   - Set up premium square layout
   - Initialize premium usage tracking (all False initially)

2. **`_initialize_premium_squares(self)`**
   - Returns a 2D array same size as board
   - Mark which squares have premiums: 'DL' (double letter), 'DW' (double word), None (normal)
   - Design a simple symmetric pattern (e.g., center = DW, corners = DL)
   - Keep it simple for mini-Scrabble: 1 DW, 4-6 DL squares

3. **`place_word(self, word, row, col, direction)`**
   - Input: word string, starting position (row, col), direction ('H' or 'V')
   - Validate the placement first (call `_can_place_word`)
   - If valid: place each letter of the word on the board grid
   - Horizontal: increment column; Vertical: increment row
   - Output: Boolean success/failure

4. **`get_letter(self, row, col)`**
   - Input: Position (row, col)
   - Output: Letter character at that position, or None if empty
   - Handle out-of-bounds gracefully

5. **`is_empty(self, row, col)`**
   - Input: Position (row, col)
   - Output: Boolean (True if no letter placed there)

6. **`to_array(self)`**
   - Output: Numerical numpy array representation of board
   - For neural network input
   - Encoding scheme: 0 = empty, 1 = A, 2 = B, ..., 26 = Z
   - Shape: (5, 5) with integers

7. **`__str__(self)`**
   - Output: Human-readable string representation
   - For debugging and visualization
   - Use '.' for empty squares, letter characters for placed tiles

**What to code:**
1. Create `board.py` with the `Board` class
2. Use a 2D list or numpy array for the grid
3. Define a simple premium square layout (hardcode positions)
4. Implement basic accessors (get_letter, is_empty)
5. Implement board-to-array conversion for neural networks
6. Add string representation for debugging
7. Write tests: place words, check positions, verify array conversion

**Key decisions:**
- Use None for empty squares, single characters ('A'-'Z') for placed letters
- Premium squares: keep it simple (1 double word in center, 4-6 double letters)
- Encoding: 0=empty, 1-26=A-Z (or consider one-hot encoding later)

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

**Methods to add to `Board` class:**

1. **`_can_place_word(self, word, row, col, direction, player_rack, is_first_move)`**
   - Input: word, position, direction, player's rack (list of letters), whether this is the first move
   - Output: (Boolean valid, String reason_if_invalid)
   - Implement all 6 validation checks listed above
   - Return False early if any check fails (fail fast)
   - Check in this order: bounds → center/adjacency → letter availability → perpendicular words

   **Logic to implement:**
   - Check 1: Calculate if word extends past board boundary
   - Check 2: If first move, check if any position of the word passes through center square (size//2, size//2)
   - Check 3: If not first move, check if word connects to existing letters (adjacent or overlapping)
   - Check 4: For each position in word, check if board already has a letter there and if it matches
   - Check 5: Count which letters need to be placed (not already on board), verify player has them in rack
   - Check 6: Find all perpendicular words formed, validate each one with dictionary

2. **`_has_adjacent_letter(self, row, col)`**
   - Input: Position (row, col)
   - Output: Boolean (True if any orthogonally adjacent square has a letter)
   - Check 4 directions: up, down, left, right (not diagonals)
   - Handle board boundaries

3. **`_get_perpendicular_words(self, word, row, col, direction)`**
   - Input: The word being placed, position, direction
   - Output: List of perpendicular words formed (each as tuple: word, start_row, start_col)
   - For each letter position in the main word:
     - If we're placing a new letter there (not using existing letter)
     - Look in the perpendicular direction
     - Extract any word formed (including adjacent letters on board)
   - This is complex - start simple and test thoroughly

4. **`_extract_vertical_word(self, row, col, center_letter)`** (helper)
   - Input: Center position and letter being placed there
   - Output: The complete vertical word formed (including board letters above and below)
   - Walk up from position while letters exist on board
   - Walk down from position while letters exist on board
   - Combine: above + center_letter + below

5. **`_extract_horizontal_word(self, row, col, center_letter)`** (helper)
   - Similar to above but walks left and right

**What to code:**
1. Add `_can_place_word()` method to `Board` class
2. Implement each validation check separately (easier to debug)
3. Implement helper methods for adjacency checking and perpendicular word extraction
4. Return both boolean and reason string (helpful for debugging)
5. Write extensive tests for each rule:
   - Valid placements should pass
   - Each invalid case should fail with correct reason
   - Test edge cases: corners, board boundaries, overlaps

**Key decisions:**
- Validate incrementally - fail fast on simple checks first
- Perpendicular word checking is the trickiest part - test thoroughly
- Keep a reference to the Dictionary in the Board class for validation

---

## Part 2: Game Mechanics (Days 4-5)

Now that you have the board and validation, implement the actual game flow.

### 2.1 Player State and Rack Management

**What it is:** Each player has a rack (hand) of tiles and a score. The rack needs to be refilled after each turn.

**Class: `Player`**

**Attributes to store:**
- Player ID (0 or 1)
- Rack (list of letter characters)
- Rack size (how many tiles to hold, default 5 for mini-Scrabble)
- Score (integer)

**Methods to implement:**

1. **`__init__(self, player_id, rack_size=5)`**
   - Initialize empty rack
   - Set score to 0
   - Store player ID and rack size

2. **`draw_tiles(self, tile_bag, n=None)`**
   - Input: TileBag object, optional number to draw (default: fill rack)
   - If n not specified: calculate how many needed to fill rack
   - Draw tiles from the bag
   - Add them to player's rack

3. **`use_tiles(self, letters)`**
   - Input: List of letters to remove from rack
   - Remove each letter from rack
   - Handle blank tiles: if letter not in rack, use a blank ('_') instead
   - Raise error or handle gracefully if player doesn't have required tiles

4. **`add_score(self, points)`**
   - Input: Points to add
   - Add to player's score

5. **`rack_to_array(self)`**
   - Output: Numerical array representation of rack for neural network
   - Use count encoding: array of size 27 (26 letters + 1 blank)
   - rack_array[0] = count of 'A' in rack, rack_array[1] = count of 'B', etc.
   - rack_array[26] = count of blanks

**What to code:**
1. Create `player.py` with `Player` class
2. Initialize with empty rack and score=0
3. Implement tile drawing (calls tile_bag.draw())
4. Implement tile removal (be careful with list.remove())
5. Add rack-to-array conversion for neural network observation
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

**Class: `Scorer`**

**Attributes to store:**
- Letter values dictionary (maps 'A' → 1, 'Q' → 10, etc.)

**Methods to implement:**

1. **`__init__(self)`**
   - Create letter values dictionary
   - Use standard Scrabble values or simplified version
   - Common letters: 1-2 points
   - Uncommon: 3-5 points
   - Rare (Q, Z, X, J): 8-10 points
   - Blanks: 0 points

2. **`score_word_placement(self, board, word, row, col, direction, letters_placed)`**
   - Input: Board object, word placed, position, direction, list of newly placed letters (as tuples: row, col, letter)
   - Output: Integer score for this placement

   **Logic to implement:**
   - Initialize word_score = 0 and word_multiplier = 1
   - For each letter in the word:
     - Get base letter value
     - If this letter was newly placed (in letters_placed list):
       - Check if position has a premium square
       - If premium not yet used:
         - Apply letter multipliers (DL = ×2, TL = ×3)
         - Accumulate word multipliers (DW = ×2, TW = ×3)
         - Mark premium as used
     - Add letter_value × letter_multiplier to word_score
   - Multiply word_score by word_multiplier
   - Calculate scores for any perpendicular words formed (similar process)
   - Check for bingo: if len(letters_placed) == rack_size, add +10 bonus
   - Return total score

3. **`_score_perpendicular_word(self, board, perp_word_info, letter_placed)`** (helper)
   - Input: Board, perpendicular word info (word, start_row, start_col), the letter you just placed
   - Output: Score for that perpendicular word
   - Similar to main word scoring but only the newly placed letter gets premium bonuses

**What to code:**
1. Create `scorer.py` with `Scorer` class
2. Define letter values (use simplified values to start)
3. Implement main scoring logic:
   - Sum letter values
   - Apply letter multipliers (DL, TL)
   - Apply word multipliers (DW, TW)
   - Add perpendicular word scores
   - Add bingo bonus if applicable
4. Track which premium squares have been used
5. Write tests: score simple words, words with premiums, bingo bonus, perpendicular words

---

### 2.3 Putting It All Together: The Game Environment

**What it is:** A Gymnasium-compatible environment that orchestrates all the components above into a playable game.

**Why it matters:** This is what the RL agent interacts with. It must follow the Gym API (reset, step, render).

**How the environment works in practice:**

The Gymnasium environment provides a standard interface for reinforcement learning. Think of it as a game engine that agents interact with using only two methods:

1. **`reset()`**: Starts a new game episode. Returns the initial game state (observation).
2. **`step(action)`**: Executes one action, updates the game state, and returns (new_state, reward, done, info).

Here's how a typical game plays out:

```python
env = MiniScrabbleEnv()

# Start new game
observation, info = env.reset()

done = False
while not done:
    # Agent looks at observation and picks action
    action = agent.choose_action(observation)  # Agent's decision

    # Environment executes the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if game ended
    done = terminated or truncated
```

**Key concept: Two-player alternating turns**
After each `step()`, the environment automatically switches to the other player. The observation returned shows the board state and rack for the player whose turn it is next. This means:
- Call `reset()` → Player 0's turn, get Player 0's observation
- Call `step(action)` → Player 0 makes move, environment switches to Player 1, returns Player 1's observation
- Call `step(action)` → Player 1 makes move, environment switches to Player 0, returns Player 0's observation
- And so on...

The RL agent will play as both players (self-play), so it needs to handle observations from either player's perspective.

**Class: `MiniScrabbleEnv(gym.Env)`**

**Attributes to store:**
- Board object
- TileBag object
- Dictionary object
- Scorer object
- Two Player objects
- Current player index (0 or 1)
- Game state flags (is_first_move, consecutive_passes)
- Action space and observation space (Gymnasium spaces)
- Action mapping (dict: action_id → move)
- Valid actions list

**Observation Space Structure:**
- Dictionary with keys:
  - 'board': Box of integers, shape (5, 5), range [0, 26] where 0=empty, 1-26=A-Z
  - 'rack': Box of integers, shape (27,), range [0, rack_size] counting each letter (26 letters + 1 blank)
  - 'action_mask': Box of int8, shape (action_space_size,), 0=invalid action, 1=valid action
  - 'score_self': Box scalar int, current player's score
  - 'score_opp': Box scalar int, opponent's score

**Action Space:**
- Discrete(N) where N = maximum possible actions
- N = 1 + (board_size × board_size × 2 directions × dictionary_size)
- Each action is an integer from 0 to N-1 that maps to either:
  - 'PASS' (always action 0)
  - A word placement: (word, row, col, direction) tuple

**Methods to implement:**

1. **`__init__(self, dictionary_path=None, board_size=5, rack_size=5)`**
   - Call `super().__init__()` to initialize the Gymnasium base class
   - Store configuration parameters: `self.board_size`, `self.rack_size`
   - Load dictionary: If path provided, read word list from file and create `Dictionary` object. Otherwise use a small hardcoded list for testing (e.g., ['CAT', 'DOG', 'HAT', 'BAT', 'RAT', 'MAT'])
   - Create Scorer object: `self.scorer = Scorer()` (persistent across episodes)
   - Initialize game state variables to None: `self.board`, `self.tile_bag`, `self.players`, `self.current_player_idx`, `self.is_first_move`, `self.consecutive_passes`
   - Initialize action tracking: `self.valid_actions = []` and `self.action_to_move = {}` (empty initially)
   - Define observation space using `spaces.Dict()` with keys: 'board' (Box shape (5,5) dtype int32), 'rack' (Box shape (27,) dtype int32), 'action_mask' (Box shape (action_space_n,) dtype int8), 'score_self' (Box shape () dtype int32), 'score_opp' (Box shape () dtype int32)
   - Calculate action space size: `max_actions = 1 + (self.board_size ** 2) * 2 * len(self.dictionary.words)`
   - Define action space: `self.action_space = spaces.Discrete(max_actions)`

2. **`reset(self, seed=None, options=None)`**
   - Call `super().reset(seed=seed)` for proper Gymnasium seeding
   - If seed is provided, also call `np.random.seed(seed)` for NumPy operations
   - Create fresh Board object: `self.board = Board(self.dictionary, size=self.board_size)` (don't reuse from previous episode)
   - Create fresh TileBag object: `self.tile_bag = TileBag(use_simplified=True)`
   - Create two fresh Player objects: `self.players = [Player(id=0, rack_size=self.rack_size), Player(id=1, rack_size=self.rack_size)]`
   - Have both players draw initial tiles: `for player in self.players: player.draw_tiles(self.tile_bag)`
   - Initialize game state: `self.current_player_idx = 0`, `self.is_first_move = True`, `self.consecutive_passes = 0`
   - Generate valid actions for first player: `self._generate_valid_actions()` (populates self.valid_actions and self.action_to_move)
   - Build observation dict: `observation = self._get_observation()` (for Player 0)
   - Create info dict with debugging info: `info = {'player_id': self.current_player_idx, 'valid_action_count': len(self.valid_actions)}`
   - Return tuple: `(observation, info)`

3. **`step(self, action)`**
   - Input: Integer action ID (from 0 to action_space.n - 1)
   - Output: Tuple of (observation, reward, terminated, truncated, info)

   **Logic to implement:**
   - **Validate action**: Check `if action not in self.valid_actions`. If invalid, return `(self._get_observation(), -10.0, True, False, {'error': 'Invalid action'})` to penalize and end episode
   - **Decode action**: Look up move in `self.action_to_move[action]` dictionary
   - **Initialize return values**: `reward = 0.0`, `terminated = False`, `truncated = False`, `info = {}`
   - **Handle PASS action** (if `move == 'PASS'`):
     - Increment `self.consecutive_passes += 1`
     - If `self.consecutive_passes >= 2`: set `terminated = True`, calculate `reward = self._compute_final_reward()`, add to info: `{'termination_reason': 'both_passed'}`
     - Otherwise: `reward = 0.0` (no reward for passing)
   - **Handle word placement** (if move is tuple `(word, row, col, direction)`):
     - Determine which letters are being placed vs already on board: Loop through word positions, check `self.board.is_empty(r, c)`, build `letters_placed = [(r, c, letter), ...]` list
     - Place word on board: `success = self.board.place_word(word, row, col, direction)`. If not success, return error (shouldn't happen if action generation correct)
     - Get current player reference: `current_player = self.players[self.current_player_idx]`
     - Calculate score: `score = self.scorer.score_word_placement(board, word, row, col, direction, letters_placed, rack_size)`
     - Update player state: `current_player.use_tiles([letter for _, _, letter in letters_placed])`, then `current_player.add_score(score)`, then `current_player.draw_tiles(self.tile_bag)`
     - Set `reward = score / 10.0` (normalize score to reasonable range)
     - Reset consecutive passes: `self.consecutive_passes = 0`
     - Mark first move complete: `self.is_first_move = False`
     - Check if game over: `if self._is_game_over(): terminated = True, reward = self._compute_final_reward(), info['termination_reason'] = 'game_over'`
   - **Switch players**: `self.current_player_idx = 1 - self.current_player_idx` (toggles between 0 and 1)
   - **Generate valid actions for next player**: Call `self._generate_valid_actions()` to populate valid actions for the player whose turn is next
   - **Build observation for next player**: `observation = self._get_observation()` (returns observation from perspective of next player)
   - **Add debugging info**: Add to info dict: `{'player_id': self.current_player_idx, 'valid_action_count': len(self.valid_actions), 'move': move}`
   - **Return**: `(observation, reward, terminated, truncated, info)`

4. **`_generate_valid_actions(self)`**
   - **Clear previous actions**: `self.valid_actions = []` and `self.action_to_move = {}`
   - **Initialize action ID counter**: `action_id = 0`
   - **Always add PASS action**: `self.valid_actions.append(0)`, `self.action_to_move[0] = 'PASS'`, then `action_id = 1`
   - **Get current player's rack**: `current_player = self.players[self.current_player_idx]`, `player_rack = current_player.rack`
   - **Triple nested loop** (this is the computational bottleneck):
     - For `row in range(self.board_size)`:
       - For `col in range(self.board_size)`:
         - For `direction in ['H', 'V']`:
           - For `word in self.dictionary.words`:
             - **Quick boundary check**: If `direction == 'H' and col + len(word) > self.board_size: continue`. If `direction == 'V' and row + len(word) > self.board_size: continue`
             - **Detailed validation**: `valid, reason = self.board._can_place_word(word, row, col, direction, player_rack, self.is_first_move)`
             - **If valid**: Add to valid actions list: `self.valid_actions.append(action_id)`, map to move: `self.action_to_move[action_id] = (word, row, col, direction)`, increment counter: `action_id += 1`
   - **Note**: This generates O(board_size² × 2 × dictionary_size) checks per turn. For 5×5 with 500 words = ~25,000 checks. Optimization comes later.

5. **`_get_observation(self)`**
   - **Get current player and opponent references**: `current_player = self.players[self.current_player_idx]`, `opponent = self.players[1 - self.current_player_idx]`
   - **Get board array**: `board_array = self.board.to_array()` (returns numpy array shape (5, 5) with integers 0-26)
   - **Get rack array**: `rack_array = current_player.rack_to_array()` (returns numpy array shape (27,) with letter counts)
   - **Build action mask**: Create numpy zeros array: `action_mask = np.zeros(self.action_space.n, dtype=np.int8)`. Set valid positions to 1: `action_mask[self.valid_actions] = 1`
   - **Build observation dict**: `observation = {'board': board_array, 'rack': rack_array, 'action_mask': action_mask, 'score_self': np.int32(current_player.score), 'score_opp': np.int32(opponent.score)}`
   - **Return**: `observation` dictionary

6. **`_is_game_over(self)`**
   - **Check consecutive passes**: `if self.consecutive_passes >= 2: return True` (both players passed)
   - **Check tile bag empty + player out of tiles**: `if self.tile_bag.is_empty():` then loop through `for player in self.players:` and check if all rack tiles are None: `if all(tile is None for tile in player.rack): return True`
   - **Otherwise**: `return False` (game continues)

7. **`render(self, mode='human')`**
   - **Print header**: Print separator line and title with current player info: `"MINI-SCRABBLE - Player {self.current_player_idx}'s turn"`
   - **Print board**: Print label "Board:", then print `self.board` (uses Board's `__str__` method showing grid with letters or dots)
   - **Print player info**: Loop through both players, for each print: player ID, score, rack tiles. Mark current player with ">>>" indicator
   - **Print game state**: Print tiles remaining in bag: `self.tile_bag.tiles_remaining()`, print `self.is_first_move` flag, print `self.consecutive_passes` count
   - **Print action info**: Print number of valid actions: `len(self.valid_actions)`. Optionally, if small number (<=10), print the actual moves available

8. **`_compute_final_reward(self)` (helper)**
   - **Purpose**: Calculate final reward when game ends, based on score differential and win/loss
   - **Get player references**: `current_player = self.players[self.current_player_idx]`, `opponent = self.players[1 - self.current_player_idx]`
   - **Apply end-game scoring** (Scrabble rule): For each player, subtract value of tiles remaining in rack from their score. Loop: `for player in self.players:` calculate `remaining_value = sum(self.scorer.letter_scores[tile.lower()] for tile in player.rack if tile is not None)`, then `player.score -= remaining_value`
   - **Calculate score differential**: `score_diff = current_player.score - opponent.score`
   - **Calculate reward**: If won (score_diff > 0): `reward = 10.0 + score_diff / 10.0`. If lost (score_diff < 0): `reward = -10.0 + score_diff / 10.0`. If tied: `reward = 0.0`
   - **Return**: `reward` (float)

**What to code:**
1. Create `mini_scrabble_env.py` with `MiniScrabbleEnv` class
2. Inherit from `gym.Env`
3. Define observation and action spaces carefully
4. Implement `reset()` - initializes game to starting state
5. Implement `step()` - executes one action and returns next state
6. Implement `_generate_valid_actions()` - the key computational challenge
7. Implement `_get_observation()` - builds observation dict
8. Implement `render()` - prints game state for debugging
9. Test: play random vs random for 100 games, verify they complete without errors

**Key decisions:**
- Action space size: use upper bound, actual valid actions are a small subset
- Reward structure: normalize scores (divide by 10?), add win/loss bonus at end
- Action generation is slow: accept this for now, optimize later
- Pass action: always include it (action 0)

---

## Part 3: Testing and Baseline Agents (Days 6-7)

Before implementing RL, verify everything works with simple agents.

### 3.1 Random Agent

**What it is:** Agent that picks uniformly from valid actions.

**What to code:**
- Function that takes observation as input
- Extracts action_mask from observation
- Finds all valid action IDs (where mask is True)
- Returns random choice from valid actions

**Testing:**
- Create environment
- Run 100 complete games with two random agents
- Verify games complete without errors
- Check that scores are reasonable (not negative, not absurdly high)
- Look at a few rendered games to see if moves make sense

**Expected results:** Games should complete without errors. Scores will be low (most moves will be short words).

---

### 3.2 Greedy Agent

**What it is:** Agent that picks the highest-scoring valid move.

**What to code:**
- Function that takes environment and observation as input
- Extracts valid action IDs from action mask
- For each valid action:
  - Decode the move (use env.action_to_move)
  - If PASS: score = 0
  - If word placement: calculate what score it would get (call env.scorer.score_word_placement)
- Select action with highest score
- Return that action ID

**Testing:**
- Run 100 games: greedy vs random
- Track win rate (greedy should win >80% of the time)
- Compare average scores (greedy should score higher)
- Render a few games to see strategy differences

**Expected results:** Greedy should beat random consistently (>80% win rate). Scores should be higher than random agent.

---

## Part 4: What's Next?

After completing Parts 1-3, you'll have:
- A working mini-Scrabble environment following the Gymnasium API
- Validated game mechanics (board, scoring, rules)
- Baseline agents for comparison

**Next steps (separate implementation guide):**
1. **Neural network architecture** - Design actor and critic networks
2. **PPO agent** - Implement PPO algorithm for action selection and learning
3. **Training loop** - Set up self-play training with data collection
4. **Logging and visualization** - Track training progress, plot learning curves
5. **Evaluation** - Test trained agent against baselines

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

**Issue**: Perpendicular word validation is confusing
- **Solution**: Draw examples on paper, test with simple cases first

**Issue**: Player doesn't have required letters but validation passed
- **Solution**: Check your letter counting logic in `_can_place_word`

---

## Deep RL Connection

Remember: you're building this environment to train a **deep RL agent**. The key connections:

- **Observation space** → What the neural network sees as input
- **Action space** → What the neural network chooses from
- **Reward signal** → What teaches the network "good" vs "bad"
- **Action masking** → Critical for handling variable valid actions (a key deep RL challenge)

The environment handles the game logic. The neural network will learn *strategy* through trial and error (self-play), generalizing patterns across millions of unique game states it's never seen before. This generalization ability is why we need **deep** RL (neural networks) rather than traditional RL (tables).

---

This should give you a solid foundation for the first phase of the project. Take it step by step, test frequently, and don't hesitate to simplify if you get stuck!
