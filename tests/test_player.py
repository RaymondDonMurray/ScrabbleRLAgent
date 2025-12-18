"""
Tests for Player class
"""
import pytest
import numpy as np
from envs.game_mechanics.player import Player
from envs.game_components.tile_bag import TileBag


class TestPlayer:
    """Test suite for Player class."""

    def test_init_default(self):
        """Test default initialization."""
        player = Player(id=0)

        assert player.id == 0
        assert player.rack_size == 5
        assert player.score == 0
        assert player.rack == []

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        player = Player(id=1, rack_size=7, score=100)

        assert player.id == 1
        assert player.rack_size == 7
        assert player.score == 100
        assert player.rack == []

    def test_draw_tiles_fill_empty_rack(self):
        """Test drawing tiles to fill an empty rack."""
        player = Player(id=0, rack_size=5)
        bag = TileBag(use_simplified=True)

        player.draw_tiles(bag)

        assert len(player.rack) == 5
        assert all(tile is not None for tile in player.rack)

    def test_draw_tiles_specific_number(self):
        """Test drawing a specific number of tiles."""
        player = Player(id=0, rack_size=7)
        bag = TileBag(use_simplified=True)

        player.draw_tiles(bag, n=3)

        assert len(player.rack) == 3

    def test_draw_tiles_partial_fill(self):
        """Test drawing tiles to fill a partially empty rack."""
        player = Player(id=0, rack_size=5)
        bag = TileBag(use_simplified=True)

        # Fill rack initially
        player.draw_tiles(bag)
        assert len(player.rack) == 5

        # Use some tiles
        player.rack = player.rack[:2]  # Keep only 2 tiles

        # Draw to fill
        player.draw_tiles(bag)

        assert len(player.rack) == 5

    def test_draw_tiles_from_limited_bag(self):
        """Test drawing when bag doesn't have enough tiles."""
        player = Player(id=0, rack_size=5)
        bag = TileBag(use_simplified=True)

        # Exhaust most of the bag
        total_tiles = bag.get_number_of_remaining_tiles()
        bag.draw(total_tiles - 2)  # Leave only 2

        player.draw_tiles(bag)

        # Should only get 2 tiles
        assert len(player.rack) == 2
        assert bag.is_empty()

    def test_use_tiles_simple(self):
        """Test using tiles that are in the rack."""
        player = Player(id=0)
        player.rack = ['C', 'A', 'T', 'D', 'O']

        blank_usage = player.use_tiles(['C', 'A', 'T'])

        assert len(player.rack) == 2
        assert 'C' not in player.rack
        assert 'A' not in player.rack
        assert 'T' not in player.rack
        assert blank_usage == [False, False, False]

    def test_use_tiles_with_blank(self):
        """Test using a blank tile."""
        player = Player(id=0)
        player.rack = ['C', 'A', '_', 'D', 'O']

        blank_usage = player.use_tiles(['C', 'A', 'T'])  # T not in rack, use blank

        assert len(player.rack) == 2
        assert '_' not in player.rack
        assert blank_usage == [False, False, True]

    def test_use_tiles_multiple_blanks(self):
        """Test using multiple blank tiles."""
        player = Player(id=0)
        player.rack = ['_', '_', 'A']

        blank_usage = player.use_tiles(['X', 'Y', 'A'])

        assert len(player.rack) == 0
        assert blank_usage == [True, True, False]

    def test_use_tiles_insufficient_letters(self):
        """Test error when not enough letters."""
        player = Player(id=0)
        player.rack = ['C', 'A', 'D']

        with pytest.raises(ValueError, match="Cannot use 'T'"):
            player.use_tiles(['C', 'A', 'T'])

    def test_use_tiles_no_blanks_available(self):
        """Test error when missing letter and no blanks."""
        player = Player(id=0)
        player.rack = ['C', 'A', 'D', 'O', 'G']

        with pytest.raises(ValueError):
            player.use_tiles(['C', 'A', 'T'])

    def test_use_tiles_track_blanks_false(self):
        """Test not tracking blank usage."""
        player = Player(id=0)
        player.rack = ['C', 'A', '_']

        result = player.use_tiles(['C', 'A', 'T'], track_blanks=False)

        assert result is None
        assert len(player.rack) == 0

    def test_use_tiles_duplicate_letters(self):
        """Test using duplicate letters."""
        player = Player(id=0)
        player.rack = ['A', 'A', 'T', 'C', 'D']

        player.use_tiles(['A', 'A', 'T'])

        assert len(player.rack) == 2
        assert 'A' not in player.rack
        assert 'C' in player.rack
        assert 'D' in player.rack

    def test_use_tiles_all_rack(self):
        """Test using all tiles in rack (bingo)."""
        player = Player(id=0, rack_size=5)
        player.rack = ['C', 'A', 'T', 'D', 'O']

        player.use_tiles(['C', 'A', 'T', 'D', 'O'])

        assert len(player.rack) == 0

    def test_add_score_positive(self):
        """Test adding positive score."""
        player = Player(id=0, score=10)

        player.add_score(5)

        assert player.score == 15

    def test_add_score_zero(self):
        """Test adding zero score."""
        player = Player(id=0, score=10)

        player.add_score(0)

        assert player.score == 10

    def test_add_score_negative_raises_error(self):
        """Test that negative scores raise error."""
        player = Player(id=0, score=10)

        with pytest.raises(ValueError, match="Cannot subtract points"):
            player.add_score(-5)

    def test_add_score_multiple_times(self):
        """Test adding score multiple times."""
        player = Player(id=0)

        player.add_score(10)
        player.add_score(20)
        player.add_score(5)

        assert player.score == 35

    def test_rack_to_array_empty_rack(self):
        """Test converting empty rack to array."""
        player = Player(id=0)
        player.rack = []

        array = player.rack_to_array()

        assert array.shape == (27,)
        assert np.all(array == 0)
        assert array.dtype == np.int32

    def test_rack_to_array_simple(self):
        """Test converting simple rack to array."""
        player = Player(id=0)
        player.rack = ['C', 'A', 'T']

        array = player.rack_to_array()

        assert array.shape == (27,)
        assert array[0] == 1  # A
        assert array[2] == 1  # C
        assert array[19] == 1  # T
        assert array[26] == 0  # No blanks

    def test_rack_to_array_with_duplicates(self):
        """Test rack with duplicate letters."""
        player = Player(id=0)
        player.rack = ['A', 'A', 'A', 'B', 'C']

        array = player.rack_to_array()

        assert array[0] == 3  # 3 A's
        assert array[1] == 1  # 1 B
        assert array[2] == 1  # 1 C

    def test_rack_to_array_with_blank(self):
        """Test rack with blank tile."""
        player = Player(id=0)
        player.rack = ['A', '_', 'C']

        array = player.rack_to_array()

        assert array[0] == 1  # A
        assert array[2] == 1  # C
        assert array[26] == 1  # Blank at index 26

    def test_rack_to_array_multiple_blanks(self):
        """Test rack with multiple blanks."""
        player = Player(id=0)
        player.rack = ['_', '_', 'A']

        array = player.rack_to_array()

        assert array[0] == 1  # A
        assert array[26] == 2  # 2 blanks

    def test_rack_to_array_all_letters(self):
        """Test rack to array is permutation invariant."""
        player1 = Player(id=0)
        player2 = Player(id=1)

        player1.rack = ['C', 'A', 'T']
        player2.rack = ['T', 'A', 'C']  # Same letters, different order

        array1 = player1.rack_to_array()
        array2 = player2.rack_to_array()

        # Should be identical (permutation invariant)
        assert np.array_equal(array1, array2)

    def test_rack_to_array_dtype(self):
        """Test that array has correct dtype."""
        player = Player(id=0)
        player.rack = ['A', 'B', 'C']

        array = player.rack_to_array()

        assert array.dtype == np.int32

    def test_full_workflow(self):
        """Test complete player workflow."""
        player = Player(id=0, rack_size=5)
        bag = TileBag(use_simplified=True)

        # Draw initial tiles
        player.draw_tiles(bag)
        assert len(player.rack) == 5

        # Convert to array
        array = player.rack_to_array()
        assert array.shape == (27,)

        # Use some tiles (if possible)
        if 'C' in player.rack and 'A' in player.rack and 'T' in player.rack:
            player.use_tiles(['C', 'A', 'T'])
            assert len(player.rack) == 2

            # Add score
            player.add_score(15)
            assert player.score == 15

            # Refill rack
            player.draw_tiles(bag)
            assert len(player.rack) == 5
