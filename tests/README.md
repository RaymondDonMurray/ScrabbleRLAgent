# Test Suite for Scrabble RL Environment

Comprehensive test suite covering all components of the mini-Scrabble reinforcement learning environment.

## Test Structure

```
tests/
├── __init__.py           # Package initialization
├── conftest.py           # Pytest configuration and shared fixtures
├── test_dictionary.py    # Dictionary class tests
├── test_tile_bag.py      # TileBag class tests
├── test_board.py         # Board class tests
├── test_player.py        # Player class tests
├── test_scorer.py        # Scorer class tests
└── test_environment.py   # MiniScrabbleEnv integration tests
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_dictionary.py
pytest tests/test_environment.py
```

### Run specific test class
```bash
pytest tests/test_dictionary.py::TestDictionary
```

### Run specific test
```bash
pytest tests/test_dictionary.py::TestDictionary::test_init_with_word_list
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with very verbose output (show each test)
```bash
pytest tests/ -vv
```

### Run tests matching a pattern
```bash
pytest tests/ -k "board"  # Run all tests with "board" in the name
```

### Run with coverage report
```bash
# Install pytest-cov first: pip install pytest-cov
pytest tests/ --cov=envs --cov-report=html --cov-report=term
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS
```

### Run only fast tests (skip slow ones)
```bash
pytest tests/ -m "not slow"
```

### Run with output capture disabled (see print statements)
```bash
pytest tests/ -s
```

### Stop on first failure
```bash
pytest tests/ -x
```

### Run last failed tests only
```bash
pytest tests/ --lf
```

## Test Categories

### Unit Tests
Test individual components in isolation:
- `test_dictionary.py` - Dictionary word validation
- `test_tile_bag.py` - Tile drawing and management
- `test_player.py` - Player state management
- `test_board.py` - Board placement and validation
- `test_scorer.py` - Word scoring logic

### Integration Tests
Test complete environment behavior:
- `test_environment.py` - Full environment with all components

## Test Coverage

### Dictionary Tests (21 tests)
- Initialization (from list, file, default)
- Word validation (case insensitivity)
- Whitespace handling
- Edge cases (empty, duplicates, large lists)

### TileBag Tests (17 tests)
- Tile drawing (single, multiple, default)
- Bag depletion
- Randomness verification
- Edge cases (empty bag, overdraw)

### Board Tests (30 tests)
- Initialization (default, custom size)
- Letter placement (horizontal, vertical)
- Word validation (boundaries, connections, first move)
- Premium squares (DL, DW)
- Word extraction (perpendicular words)
- Array conversion for neural networks

### Player Tests (20 tests)
- Initialization
- Tile drawing (full, partial, limited bag)
- Tile usage (simple, with blanks, errors)
- Score management
- Rack-to-array conversion (permutation invariance)

### Scorer Tests (18 tests)
- Single word scoring
- Premium multipliers (DL, DW)
- Premium usage tracking
- Perpendicular word scoring
- Bingo bonus (+10 for using all tiles)
- Case insensitivity

### Environment Tests (35+ tests)
- Initialization and configuration
- Reset functionality (with/without seed)
- Step functionality (valid/invalid actions)
- Action generation and masking
- Observation structure and validity
- Player switching
- Score updates
- Game termination conditions
- Rendering
- Complete episode playthrough
- Gymnasium API compliance

## Requirements

```bash
pip install pytest
pip install pytest-cov  # Optional, for coverage reports
pip install numpy
pip install gymnasium
```

## Writing New Tests

### Test Structure
Follow the existing pattern:

```python
import pytest
from envs import YourClass


class TestYourClass:
    """Test suite for YourClass."""

    @pytest.fixture
    def your_object(self):
        """Create test object."""
        return YourClass()

    def test_basic_functionality(self, your_object):
        """Test basic functionality."""
        result = your_object.do_something()
        assert result == expected_value

    def test_edge_case(self, your_object):
        """Test edge case."""
        with pytest.raises(ValueError):
            your_object.invalid_operation()
```

### Testing Guidelines

1. **One concept per test**: Each test should verify one specific behavior
2. **Clear names**: Test names should describe what they test
3. **Arrange-Act-Assert**: Structure tests in three phases:
   - Arrange: Set up test data
   - Act: Execute the code being tested
   - Assert: Verify the results
4. **Use fixtures**: Share common setup code via fixtures
5. **Test edge cases**: Empty inputs, boundary values, invalid inputs
6. **Test error cases**: Verify exceptions are raised appropriately

## Common Issues

### Import Errors
If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/ScrabbleRLAgent
pytest tests/
```

### Random Test Failures
Some tests involve randomness. If a test fails:
1. Check if it's marked as potentially flaky
2. Run it multiple times to verify
3. Use fixed seeds for reproducibility

### Slow Tests
Some integration tests may take time. Skip them during development:
```bash
pytest tests/ -m "not slow"
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=envs --cov-report=xml
```

## Test Results Interpretation

### Success
All tests pass - environment is working correctly!

### Failures
If tests fail:
1. Read the failure message carefully
2. Check which assertion failed
3. Look at the test code to understand what's expected
4. Fix the bug in the implementation
5. Re-run tests to verify fix

### Common Failure Patterns

**AttributeError**: Missing method or property
- Check that all required methods are implemented

**AssertionError**: Unexpected value
- Check calculation logic or expected values

**TypeError**: Wrong type
- Check type hints and conversions

**ValueError**: Invalid value
- Check input validation logic

## Next Steps

After all tests pass:
1. Run with coverage to identify untested code
2. Add tests for any uncovered edge cases
3. Test with actual RL training loops
4. Profile for performance bottlenecks

## Contact

For issues or questions about tests, check:
- Implementation guide: `IMPLEMENTATION_GUIDE.md`
- Code review: `CODE_REVIEW.md`
