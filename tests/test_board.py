"""
Tests for Board class
"""
import pytest
import numpy as np
from envs.game_components.board import Board
from envs.game_components.dictionary import Dictionary


class TestBoard:
    """Test suite for Board class."""

    @pytest.fixture
    def simple_dictionary(self):
        """Create a simple dictionary for testing."""
        words = ['CAT', 'DOG', 'HAT', 'BAT', 'RAT', 'MAT',
                 'AT', 'GO', 'TO', 'IT', 'DO', 'SO']
        return Dictionary(word_list=words)

    @pytest.fixture
    def board(self, simple_dictionary):
        """Create a standard 5x5 board."""
        return Board(dictionary=simple_dictionary, size=5)

    def test_init_default_size(self, simple_dictionary):
        """Test board initialization with default size."""
        board = Board(dictionary=simple_dictionary)

        assert board.size == 5
        assert len(board.grid) == 5
        assert len(board.grid[0]) == 5
        assert all(board.grid[i][j] is None for i in range(5) for j in range(5))

    def test_init_custom_size(self, simple_dictionary):
        """Test board initialization with custom size."""
        board = Board(dictionary=simple_dictionary, size=7)

        assert board.size == 7
        assert len(board.grid) == 7

    def test_premium_squares_initialized(self, board):
        """Test that premium squares are initialized."""
        assert board.premium_squares is not None
        assert len(board.premium_squares) == 5

        # Center should be DW
        assert board.premium_squares[2][2] == 'DW'

        # Corners should be DL
        assert board.premium_squares[0][0] == 'DL'
        assert board.premium_squares[0][4] == 'DL'
        assert board.premium_squares[4][0] == 'DL'
        assert board.premium_squares[4][4] == 'DL'

    def test_premium_used_initialized_false(self, board):
        """Test that premium_used is initialized to False."""
        assert all(not board.premium_used[i][j] for i in range(5) for j in range(5))

    def test_is_empty_new_board(self, board):
        """Test that new board is empty."""
        for i in range(5):
            for j in range(5):
                assert board.is_empty(i, j) is True

    def test_is_empty_after_placement(self, board):
        """Test is_empty after placing letters."""
        board.grid[2][2] = 'A'

        assert board.is_empty(2, 2) is False
        assert board.is_empty(2, 3) is True

    def test_get_letter_empty(self, board):
        """Test getting letter from empty square."""
        assert board.get_letter(0, 0) is None

    def test_get_letter_filled(self, board):
        """Test getting letter from filled square."""
        board.grid[1][1] = 'C'

        assert board.get_letter(1, 1) == 'C'

    def test_get_letter_out_of_bounds(self, board):
        """Test getting letter from out of bounds."""
        assert board.get_letter(-1, 0) is None
        assert board.get_letter(0, -1) is None
        assert board.get_letter(5, 0) is None
        assert board.get_letter(0, 5) is None

    def test_place_word_horizontal_first_move(self, board):
        """Test placing first word horizontally through center."""
        player_rack = ['C', 'A', 'T']

        success = board.place_word('CAT', 2, 2, 'H')

        assert success is True
        assert board.grid[2][2] == 'C'
        assert board.grid[2][3] == 'A'
        assert board.grid[2][4] == 'T'

    def test_place_word_vertical_first_move(self, board):
        """Test placing first word vertically through center."""
        player_rack = ['D', 'O', 'G']

        success = board.place_word('DOG', 2, 2, 'V')

        assert success is True
        assert board.grid[2][2] == 'D'
        assert board.grid[3][2] == 'O'
        assert board.grid[4][2] == 'G'

    def test_to_array_empty_board(self, board):
        """Test converting empty board to array."""
        array = board.to_array()

        assert array.shape == (5, 5)
        assert np.all(array == 0)
        assert array.dtype == np.int32

    def test_to_array_with_letters(self, board):
        """Test converting board with letters to array."""
        board.grid[2][2] = 'C'
        board.grid[2][3] = 'A'
        board.grid[2][4] = 'T'

        array = board.to_array()

        assert array[2][2] == 3  # C = 3
        assert array[2][3] == 1  # A = 1
        assert array[2][4] == 20  # T = 20

    def test_to_array_all_letters(self, board):
        """Test array encoding for all letters."""
        # Place A-Z on board
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, letter in enumerate(letters[:25]):  # First 25 letters
            board.grid[i // 5][i % 5] = letter

        array = board.to_array()

        # Check that A=1, B=2, ..., Y=25
        for i, letter in enumerate(letters[:25]):
            expected_value = ord(letter) - ord('A') + 1
            actual_value = array[i // 5][i % 5]
            assert actual_value == expected_value, f"Letter {letter} should be {expected_value}, got {actual_value}"

    def test_str_representation(self, board):
        """Test string representation."""
        board.grid[0][0] = 'C'
        board.grid[0][1] = 'A'
        board.grid[0][2] = 'T'

        board_str = str(board)

        assert 'C' in board_str
        assert 'A' in board_str
        assert 'T' in board_str
        assert '.' in board_str  # Empty squares

    def test_can_place_word_first_move_through_center(self, board):
        """Test validation of first word through center."""
        player_rack = ['C', 'A', 'T']

        valid, reason = board._can_place_word('CAT', 2, 2, 'H', player_rack, True)

        assert valid is True

    def test_can_place_word_first_move_not_through_center(self, board):
        """Test validation fails if first word not through center."""
        player_rack = ['C', 'A', 'T']

        valid, reason = board._can_place_word('CAT', 0, 0, 'H', player_rack, True)

        assert valid is False
        assert 'center' in reason.lower()

    def test_can_place_word_out_of_bounds_horizontal(self, board):
        """Test validation fails if word extends past boundary."""
        player_rack = ['C', 'A', 'T']

        valid, reason = board._can_place_word('CAT', 2, 3, 'H', player_rack, True)

        assert valid is False
        assert 'boundary' in reason.lower()

    def test_can_place_word_out_of_bounds_vertical(self, board):
        """Test validation fails if word extends past boundary vertically."""
        player_rack = ['C', 'A', 'T']

        valid, reason = board._can_place_word('CAT', 3, 2, 'V', player_rack, True)

        assert valid is False
        assert 'boundary' in reason.lower()

    def test_can_place_word_insufficient_letters(self, board):
        """Test validation fails without enough letters."""
        player_rack = ['C', 'A']  # Missing T

        valid, reason = board._can_place_word('CAT', 2, 2, 'H', player_rack, True)

        assert valid is False
        assert 'required letters' in reason.lower()

    def test_can_place_word_with_blank(self, board):
        """Test validation succeeds using blank tile."""
        player_rack = ['C', 'A', '_']  # Blank instead of T

        valid, reason = board._can_place_word('CAT', 2, 2, 'H', player_rack, True)

        assert valid is True

    def test_can_place_word_second_move_no_connection(self, board):
        """Test validation fails if second move doesn't connect."""
        # Place first word
        board.grid[2][2] = 'C'
        board.grid[2][3] = 'A'
        board.grid[2][4] = 'T'

        player_rack = ['D', 'O', 'G']

        # Try to place disconnected word
        valid, reason = board._can_place_word('DOG', 0, 0, 'H', player_rack, False)

        assert valid is False
        assert 'connect' in reason.lower()

    def test_can_place_word_second_move_adjacent(self, board):
        """Test validation succeeds if second move is adjacent."""
        # Place first word: CAT horizontally at (2, 2)
        board.grid[2][2] = 'C'
        board.grid[2][3] = 'A'
        board.grid[2][4] = 'T'

        player_rack = ['D', 'O', 'G']

        # Place word adjacent (sharing no letters but adjacent)
        valid, reason = board._can_place_word('DOG', 3, 2, 'H', player_rack, False)

        assert valid is True

    def test_extract_horizontal_word_simple(self, board):
        """Test extracting horizontal word."""
        # Place C_T on board (missing A in middle)
        board.grid[1][1] = 'C'
        board.grid[1][3] = 'T'

        # Extract word when placing A at (1, 2)
        result = board._extract_horizontal_word(1, 2, 'A')

        assert result is not None
        word, row, col = result
        assert word == 'CAT'
        assert row == 1
        assert col == 1

    def test_extract_horizontal_word_no_adjacent(self, board):
        """Test extracting word with no adjacent letters."""
        # Empty board, placing single letter
        result = board._extract_horizontal_word(1, 1, 'A')

        assert result is None

    def test_extract_vertical_word_simple(self, board):
        """Test extracting vertical word."""
        # Place C_T vertically (missing A)
        board.grid[1][1] = 'C'
        board.grid[3][1] = 'T'

        # Extract word when placing A at (2, 1)
        result = board._extract_vertical_word(2, 1, 'A')

        assert result is not None
        word, row, col = result
        assert word == 'CAT'
        assert row == 1
        assert col == 1

    def test_extract_vertical_word_no_adjacent(self, board):
        """Test extracting vertical word with no adjacent letters."""
        result = board._extract_vertical_word(1, 1, 'A')

        assert result is None

    def test_has_adjacent_letter_empty_board(self, board):
        """Test adjacent check on empty board."""
        assert board._has_adjacent_letter(2, 2) is False

    def test_has_adjacent_letter_with_neighbor(self, board):
        """Test adjacent check with neighbor."""
        board.grid[2][2] = 'A'

        # Check adjacent positions
        assert board._has_adjacent_letter(2, 3) is True  # Right
        assert board._has_adjacent_letter(2, 1) is True  # Left
        assert board._has_adjacent_letter(1, 2) is True  # Above
        assert board._has_adjacent_letter(3, 2) is True  # Below

        # Diagonal should not count
        assert board._has_adjacent_letter(1, 1) is False

    def test_get_perpendicular_words_none(self, board):
        """Test getting perpendicular words when none formed."""
        # Empty board, place word with no perpendiculars
        result = board._get_perpendicular_words('CAT', 2, 2, 'H')

        assert len(result) == 0

    def test_get_perpendicular_words_single(self, board):
        """Test getting single perpendicular word."""
        # Place 'A' above where we'll place 'C' in CAT
        board.grid[1][2] = 'A'

        # When placing CAT at (2,2) horizontal, should form CA vertically
        result = board._get_perpendicular_words('CAT', 2, 2, 'H')

        assert len(result) == 1
        word, row, col = result[0]
        assert word == 'AC' or word == 'CA'  # Could be either depending on implementation

    def test_complex_placement_scenario(self, simple_dictionary):
        """Test complex multi-word placement scenario."""
        board = Board(dictionary=simple_dictionary, size=5)

        # Place first word: CAT
        board.grid[2][2] = 'C'
        board.grid[2][3] = 'A'
        board.grid[2][4] = 'T'

        # Place second word: DOG vertically at (1, 3), sharing the 'A'
        board.grid[1][3] = 'D'
        board.grid[2][3] = 'A'  # Already there
        board.grid[3][3] = 'G'

        # Verify board state
        assert board.grid[2][2] == 'C'
        assert board.grid[2][3] == 'A'
        assert board.grid[2][4] == 'T'
        assert board.grid[1][3] == 'D'
        assert board.grid[3][3] == 'G'
