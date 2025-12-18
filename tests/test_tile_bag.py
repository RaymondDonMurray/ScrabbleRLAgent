"""
Tests for TileBag class
"""
import pytest
from envs.game_components.tile_bag import TileBag


class TestTileBag:
    """Test suite for TileBag class."""

    def test_init_simplified(self):
        """Test initialization with simplified tile set."""
        bag = TileBag(use_simplified=True)

        assert len(bag.tiles) > 0
        assert len(bag.remaining_tiles) == len(bag.tiles)
        # Check that vowels are present
        assert 'A' in bag.tiles
        assert 'E' in bag.tiles
        # Check blanks are present
        assert '_' in bag.tiles

    def test_simplified_tile_distribution(self):
        """Test that simplified distribution is correct."""
        bag = TileBag(use_simplified=True)

        # Count specific letters
        a_count = bag.tiles.count('A')
        blank_count = bag.tiles.count('_')

        assert a_count == 2  # Should have 2 of each vowel
        assert blank_count == 2  # Should have 2 blanks

    def test_draw_single_tile(self):
        """Test drawing a single tile."""
        bag = TileBag(use_simplified=True)
        initial_count = len(bag.remaining_tiles)

        tiles = bag.draw(1)

        assert len(tiles) == 1
        assert len(bag.remaining_tiles) == initial_count - 1
        assert tiles[0] not in bag.remaining_tiles or bag.remaining_tiles.count(tiles[0]) < bag.tiles.count(tiles[0])

    def test_draw_multiple_tiles(self):
        """Test drawing multiple tiles."""
        bag = TileBag(use_simplified=True)
        initial_count = len(bag.remaining_tiles)

        tiles = bag.draw(5)

        assert len(tiles) == 5
        assert len(bag.remaining_tiles) == initial_count - 5

    def test_draw_default_parameter(self):
        """Test that draw defaults to 1 tile."""
        bag = TileBag(use_simplified=True)
        initial_count = len(bag.remaining_tiles)

        tiles = bag.draw()  # No parameter

        assert len(tiles) == 1
        assert len(bag.remaining_tiles) == initial_count - 1

    def test_draw_more_than_available(self):
        """Test drawing more tiles than available."""
        bag = TileBag(use_simplified=True)
        total_tiles = len(bag.remaining_tiles)

        # Try to draw more than available
        tiles = bag.draw(total_tiles + 10)

        # Should only get what's available
        assert len(tiles) == total_tiles
        assert len(bag.remaining_tiles) == 0

    def test_draw_until_empty(self):
        """Test drawing all tiles."""
        bag = TileBag(use_simplified=True)
        total_tiles = len(bag.remaining_tiles)

        tiles = bag.draw(total_tiles)

        assert len(tiles) == total_tiles
        assert len(bag.remaining_tiles) == 0
        assert bag.is_empty() is True

    def test_draw_from_empty_bag(self):
        """Test drawing from an empty bag."""
        bag = TileBag(use_simplified=True)
        # Empty the bag
        bag.draw(len(bag.remaining_tiles))

        tiles = bag.draw(5)

        assert len(tiles) == 0
        assert bag.is_empty() is True

    def test_is_empty_initially_false(self):
        """Test that bag is not empty initially."""
        bag = TileBag(use_simplified=True)

        assert bag.is_empty() is False

    def test_is_empty_after_exhaustion(self):
        """Test that bag is empty after drawing all tiles."""
        bag = TileBag(use_simplified=True)
        bag.draw(len(bag.remaining_tiles))

        assert bag.is_empty() is True

    def test_get_number_of_remaining_tiles_initial(self):
        """Test getting remaining tile count initially."""
        bag = TileBag(use_simplified=True)

        count = bag.get_number_of_remaining_tiles()

        assert count == len(bag.remaining_tiles)
        assert count > 0

    def test_get_number_of_remaining_tiles_after_draw(self):
        """Test remaining tile count after drawing."""
        bag = TileBag(use_simplified=True)
        initial_count = bag.get_number_of_remaining_tiles()

        bag.draw(5)

        assert bag.get_number_of_remaining_tiles() == initial_count - 5

    def test_get_number_of_remaining_tiles_empty(self):
        """Test remaining tile count when empty."""
        bag = TileBag(use_simplified=True)
        bag.draw(len(bag.remaining_tiles))

        assert bag.get_number_of_remaining_tiles() == 0

    def test_drawn_tiles_are_removed(self):
        """Test that drawn tiles are actually removed from bag."""
        bag = TileBag(use_simplified=True)

        # Draw all 'A' tiles
        drawn_as = []
        while 'A' in bag.remaining_tiles:
            tiles = bag.draw(1)
            if tiles and tiles[0] == 'A':
                drawn_as.append(tiles[0])

        # Should have drawn 2 A's (simplified has 2 of each vowel)
        assert len(drawn_as) == 2
        assert 'A' not in bag.remaining_tiles

    def test_draw_returns_valid_tiles(self):
        """Test that all drawn tiles are valid."""
        bag = TileBag(use_simplified=True)

        tiles = bag.draw(10)

        for tile in tiles:
            # Should be uppercase letter or blank
            assert tile.isupper() or tile == '_'
            assert len(tile) == 1

    def test_randomness_of_draw(self):
        """Test that draws are random (not deterministic)."""
        bag1 = TileBag(use_simplified=True)
        bag2 = TileBag(use_simplified=True)

        tiles1 = bag1.draw(10)
        tiles2 = bag2.draw(10)

        # Very unlikely (but possible) that two random draws are identical
        # This test might occasionally fail due to randomness
        # If it fails frequently, there's an issue
        assert tiles1 != tiles2 or len(tiles1) < 3  # Allow same if very few tiles

    def test_original_tiles_unchanged(self):
        """Test that original tiles list is unchanged after draws."""
        bag = TileBag(use_simplified=True)
        original_count = len(bag.tiles)

        bag.draw(10)

        # Original tiles should be unchanged
        assert len(bag.tiles) == original_count
