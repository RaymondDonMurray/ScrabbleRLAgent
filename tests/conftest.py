"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path so we can import envs
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_word_list():
    """Shared test word list for all tests."""
    return ['CAT', 'DOG', 'HAT', 'BAT', 'RAT', 'MAT',
            'AT', 'GO', 'TO', 'IT', 'DO', 'SO',
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT']


@pytest.fixture(scope="session")
def extended_word_list():
    """Extended word list for more complex tests."""
    words = [
        # 2-letter words
        'AT', 'IT', 'TO', 'GO', 'DO', 'SO', 'NO', 'OR',
        # 3-letter words
        'CAT', 'DOG', 'HAT', 'BAT', 'RAT', 'MAT', 'SAT', 'PAT',
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
        # 4-letter words
        'CATS', 'DOGS', 'HATS', 'BATS', 'RATS', 'MATS',
        'THAT', 'WITH', 'HAVE', 'THIS', 'WILL', 'YOUR',
        # 5-letter words
        'ABOUT', 'WHICH', 'THEIR', 'WOULD', 'THERE', 'COULD',
    ]
    return words


# Configure pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
