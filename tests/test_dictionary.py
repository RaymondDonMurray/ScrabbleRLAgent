"""
Tests for Dictionary class
"""
import pytest
from envs.game_components.dictionary import Dictionary


class TestDictionary:
    """Test suite for Dictionary class."""

    def test_init_with_word_list(self):
        """Test initialization with a word list."""
        words = ['CAT', 'DOG', 'HAT']
        dictionary = Dictionary(word_list=words)

        assert len(dictionary.words) == 3
        assert 'CAT' in dictionary.words
        assert 'DOG' in dictionary.words
        assert 'HAT' in dictionary.words

    def test_init_with_lowercase_words(self):
        """Test that lowercase words are converted to uppercase."""
        words = ['cat', 'dog', 'hat']
        dictionary = Dictionary(word_list=words)

        assert 'CAT' in dictionary.words
        assert 'DOG' in dictionary.words
        assert 'cat' not in dictionary.words  # Should be uppercase

    def test_init_with_mixed_case(self):
        """Test initialization with mixed case words."""
        words = ['CaT', 'DoG', 'hAt']
        dictionary = Dictionary(word_list=words)

        assert 'CAT' in dictionary.words
        assert 'DOG' in dictionary.words
        assert 'HAT' in dictionary.words

    def test_init_default(self):
        """Test default initialization (hardcoded word list)."""
        dictionary = Dictionary()

        assert len(dictionary.words) > 0
        assert 'CAT' in dictionary.words  # Should have default words

    def test_init_with_empty_list(self):
        """Test initialization with empty word list."""
        dictionary = Dictionary(word_list=[])

        assert len(dictionary.words) == 0

    def test_init_strips_whitespace(self):
        """Test that whitespace is stripped from words."""
        words = ['  CAT  ', '\tDOG\n', ' HAT']
        dictionary = Dictionary(word_list=words)

        assert 'CAT' in dictionary.words
        assert 'DOG' in dictionary.words
        assert '  CAT  ' not in dictionary.words

    def test_init_filters_empty_strings(self):
        """Test that empty strings are filtered out."""
        words = ['CAT', '', '   ', 'DOG', '\n']
        dictionary = Dictionary(word_list=words)

        assert len(dictionary.words) == 2
        assert 'CAT' in dictionary.words
        assert 'DOG' in dictionary.words

    def test_is_valid_word_uppercase(self):
        """Test word validation with uppercase input."""
        dictionary = Dictionary(word_list=['CAT', 'DOG'])

        assert dictionary.is_valid_word('CAT') is True
        assert dictionary.is_valid_word('DOG') is True
        assert dictionary.is_valid_word('HAT') is False

    def test_is_valid_word_lowercase(self):
        """Test word validation with lowercase input."""
        dictionary = Dictionary(word_list=['CAT', 'DOG'])

        # Should work with lowercase input (gets converted)
        assert dictionary.is_valid_word('cat') is True
        assert dictionary.is_valid_word('dog') is True
        assert dictionary.is_valid_word('hat') is False

    def test_is_valid_word_mixed_case(self):
        """Test word validation with mixed case input."""
        dictionary = Dictionary(word_list=['CAT', 'DOG'])

        assert dictionary.is_valid_word('CaT') is True
        assert dictionary.is_valid_word('DoG') is True

    def test_is_valid_word_empty_string(self):
        """Test validation of empty string."""
        dictionary = Dictionary(word_list=['CAT'])

        assert dictionary.is_valid_word('') is False

    def test_duplicate_words(self):
        """Test that duplicate words are handled (sets deduplicate)."""
        words = ['CAT', 'CAT', 'DOG', 'dog']
        dictionary = Dictionary(word_list=words)

        assert len(dictionary.words) == 2  # Set should deduplicate

    def test_init_from_file(self, tmp_path):
        """Test initialization from file."""
        # Create temporary word file
        word_file = tmp_path / "words.txt"
        word_file.write_text("CAT\nDOG\nHAT\n")

        dictionary = Dictionary(file_path=str(word_file))

        assert len(dictionary.words) == 3
        assert 'CAT' in dictionary.words
        assert 'DOG' in dictionary.words
        assert 'HAT' in dictionary.words

    def test_init_from_nonexistent_file(self):
        """Test initialization from non-existent file."""
        dictionary = Dictionary(file_path='/nonexistent/path/words.txt')

        # Should create empty dictionary with warning
        assert len(dictionary.words) == 0

    def test_large_word_list(self):
        """Test with large word list."""
        words = [f'WORD{i}' for i in range(10000)]
        dictionary = Dictionary(word_list=words)

        assert len(dictionary.words) == 10000
        assert 'WORD0' in dictionary.words
        assert 'WORD9999' in dictionary.words
        assert dictionary.is_valid_word('WORD5000') is True
        assert dictionary.is_valid_word('WORD99999') is False
