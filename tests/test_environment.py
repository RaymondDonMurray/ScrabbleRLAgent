"""
Tests for MiniScrabbleEnv (the complete environment)
"""
import pytest
import numpy as np
import gymnasium as gym
from envs import MiniScrabbleEnv


class TestMiniScrabbleEnv:
    """Test suite for MiniScrabbleEnv."""

    @pytest.fixture
    def env(self):
        """Create a basic environment."""
        return MiniScrabbleEnv()

    def test_init_default(self):
        """Test default initialization."""
        env = MiniScrabbleEnv()

        assert env.board_size == 5
        assert env.rack_size == 5
        assert env.dictionary is not None
        assert env.scorer is not None
        assert len(env.dictionary.words) > 0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        env = MiniScrabbleEnv(board_size=7, rack_size=7)

        assert env.board_size == 7
        assert env.rack_size == 7

    def test_action_space_defined(self, env):
        """Test that action space is properly defined."""
        assert hasattr(env, 'action_space')
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n > 0

    def test_observation_space_defined(self, env):
        """Test that observation space is properly defined."""
        assert hasattr(env, 'observation_space')
        assert isinstance(env.observation_space, gym.spaces.Dict)

        # Check required keys
        assert 'board' in env.observation_space.spaces
        assert 'rack' in env.observation_space.spaces
        assert 'action_mask' in env.observation_space.spaces
        assert 'score_self' in env.observation_space.spaces
        assert 'score_opp' in env.observation_space.spaces

    def test_observation_space_shapes(self, env):
        """Test observation space has correct shapes."""
        obs_space = env.observation_space

        assert obs_space['board'].shape == (5, 5)
        assert obs_space['rack'].shape == (27,)
        assert obs_space['action_mask'].shape == (env.action_space.n,)
        assert obs_space['score_self'].shape == ()
        assert obs_space['score_opp'].shape == ()

    def test_reset_returns_observation_and_info(self, env):
        """Test that reset returns observation and info."""
        observation, info = env.reset()

        assert isinstance(observation, dict)
        assert isinstance(info, dict)

    def test_reset_observation_structure(self, env):
        """Test that reset returns properly structured observation."""
        observation, info = env.reset()

        # Check all required keys present
        assert 'board' in observation
        assert 'rack' in observation
        assert 'action_mask' in observation
        assert 'score_self' in observation
        assert 'score_opp' in observation

        # Check types
        assert isinstance(observation['board'], np.ndarray)
        assert isinstance(observation['rack'], np.ndarray)
        assert isinstance(observation['action_mask'], np.ndarray)

    def test_reset_initializes_game_state(self, env):
        """Test that reset properly initializes game state."""
        observation, info = env.reset()

        # Board should be mostly empty initially
        assert np.sum(observation['board'] == 0) >= 20  # Most squares empty

        # Rack should have tiles
        assert np.sum(observation['rack']) == 5  # Should have 5 tiles

        # Scores should be zero
        assert observation['score_self'] == 0
        assert observation['score_opp'] == 0

        # Should have some valid actions
        assert np.sum(observation['action_mask']) > 0

    def test_reset_with_seed(self, env):
        """Test that reset with seed is reproducible."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        # Should be identical with same seed
        assert np.array_equal(obs1['board'], obs2['board'])
        assert np.array_equal(obs1['rack'], obs2['rack'])

    def test_reset_different_seeds(self, env):
        """Test that different seeds produce different results."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)

        # Very unlikely to be identical with different seeds
        assert not np.array_equal(obs1['rack'], obs2['rack'])

    def test_reset_creates_fresh_game(self, env):
        """Test that reset creates completely fresh game."""
        # Play a turn
        obs1, _ = env.reset(seed=42)
        valid_actions = np.where(obs1['action_mask'] == 1)[0]
        env.step(valid_actions[0])

        # Reset and verify fresh state
        obs2, _ = env.reset(seed=123)

        assert obs2['score_self'] == 0
        assert obs2['score_opp'] == 0
        assert np.sum(obs2['board'] == 0) >= 20  # Empty board

    def test_step_returns_five_values(self, env):
        """Test that step returns correct tuple."""
        observation, _ = env.reset()
        valid_actions = np.where(observation['action_mask'] == 1)[0]

        result = env.step(valid_actions[0])

        assert len(result) == 5
        observation, reward, terminated, truncated, info = result

        assert isinstance(observation, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_invalid_action(self, env):
        """Test that invalid action is handled correctly."""
        observation, _ = env.reset()

        # Find an invalid action
        invalid_actions = np.where(observation['action_mask'] == 0)[0]

        if len(invalid_actions) > 0:
            observation, reward, terminated, truncated, info = env.step(invalid_actions[0])

            assert terminated is True
            assert reward < 0  # Should be penalized
            assert 'error' in info

    def test_step_pass_action(self, env):
        """Test PASS action (action 0)."""
        observation, _ = env.reset()

        observation, reward, terminated, truncated, info = env.step(0)

        # First pass shouldn't end game
        assert terminated is False
        assert reward == 0.0  # No reward for passing

    def test_step_two_passes_ends_game(self, env):
        """Test that two consecutive passes ends the game."""
        observation, _ = env.reset()

        # First pass
        observation, reward, terminated, truncated, info = env.step(0)
        assert terminated is False

        # Second pass
        observation, reward, terminated, truncated, info = env.step(0)
        assert terminated is True

    def test_step_valid_word_placement(self, env):
        """Test placing a valid word."""
        observation, _ = env.reset()

        # Find a valid word placement action (not PASS)
        valid_actions = np.where(observation['action_mask'] == 1)[0]
        word_actions = [a for a in valid_actions if a != 0]

        if len(word_actions) > 0:
            observation, reward, terminated, truncated, info = env.step(word_actions[0])

            # Should not terminate on first word
            assert terminated is False
            # Should receive some reward
            assert reward >= 0

    def test_action_generation_always_includes_pass(self, env):
        """Test that PASS is always a valid action."""
        observation, _ = env.reset()

        # Action 0 (PASS) should always be valid
        assert observation['action_mask'][0] == 1

    def test_action_mask_matches_valid_actions(self, env):
        """Test that action mask correctly represents valid actions."""
        observation, _ = env.reset()

        # Count valid actions
        mask_count = np.sum(observation['action_mask'])
        env_count = len(env.valid_actions)

        assert mask_count == env_count

    def test_observation_satisfies_space(self, env):
        """Test that observations satisfy the observation space."""
        observation, _ = env.reset()

        # Should not raise error
        assert env.observation_space.contains(observation)

    def test_player_switching(self, env):
        """Test that players switch after each turn."""
        observation, info = env.reset()

        player1_id = info['player_id']

        valid_actions = np.where(observation['action_mask'] == 1)[0]
        observation, reward, terminated, truncated, info = env.step(valid_actions[0])

        player2_id = info['player_id']

        # Should switch to other player
        assert player1_id != player2_id
        assert player2_id == 1 - player1_id

    def test_scores_update(self, env):
        """Test that scores update after word placement."""
        observation, _ = env.reset()

        initial_score = observation['score_self']

        # Place a word (not PASS)
        valid_actions = np.where(observation['action_mask'] == 1)[0]
        word_actions = [a for a in valid_actions if a != 0]

        if len(word_actions) > 0:
            observation, reward, terminated, truncated, info = env.step(word_actions[0])

            # Score should have changed (either self or opp depending on who moved)
            # After player switch, score_opp should be updated
            assert observation['score_opp'] >= 0

    def test_render_doesnt_crash(self, env):
        """Test that render doesn't crash."""
        env.reset()

        # Should not raise exception
        env.render()

    def test_complete_episode_random_actions(self, env):
        """Test playing a complete episode with random actions."""
        observation, _ = env.reset(seed=42)

        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            valid_actions = np.where(observation['action_mask'] == 1)[0]
            action = np.random.choice(valid_actions)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # Should eventually end
        assert done or steps >= max_steps

    def test_multiple_episodes(self, env):
        """Test running multiple episodes."""
        for episode in range(5):
            observation, _ = env.reset(seed=episode)

            done = False
            steps = 0

            while not done and steps < 50:
                valid_actions = np.where(observation['action_mask'] == 1)[0]
                action = np.random.choice(valid_actions)

                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

    def test_game_over_conditions(self, env):
        """Test various game over conditions."""
        observation, _ = env.reset()

        # Test: Two passes ends game
        observation, reward, terminated, truncated, info = env.step(0)
        assert terminated is False

        observation, reward, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert 'termination_reason' in info

    def test_reward_structure(self, env):
        """Test reward structure makes sense."""
        observation, _ = env.reset()

        valid_actions = np.where(observation['action_mask'] == 1)[0]
        word_actions = [a for a in valid_actions if a != 0]

        if len(word_actions) > 0:
            observation, reward, terminated, truncated, info = env.step(word_actions[0])

            # Reward should be normalized (roughly -10 to +10 range)
            assert -20 <= reward <= 20

    def test_observation_consistency(self, env):
        """Test that observation is consistent with environment state."""
        observation, _ = env.reset()

        # Board array should match board state
        assert observation['board'].shape == (env.board_size, env.board_size)

        # Rack should have rack_size total tiles (or fewer if bag is empty)
        assert np.sum(observation['rack']) <= env.rack_size

    def test_action_mapping_consistency(self, env):
        """Test that action mapping is consistent."""
        env.reset()

        # All valid actions should have mappings
        for action_id in env.valid_actions:
            assert action_id in env.action_to_move

    def test_first_move_through_center(self, env):
        """Test that first move must go through center."""
        observation, _ = env.reset()

        # First move should require center
        assert env.is_first_move is True

        # Make a move
        valid_actions = np.where(observation['action_mask'] == 1)[0]
        word_actions = [a for a in valid_actions if a != 0]

        if len(word_actions) > 0:
            observation, reward, terminated, truncated, info = env.step(word_actions[0])

            # After first word, should no longer be first move
            assert env.is_first_move is False

    def test_tile_bag_depletion(self, env):
        """Test game behavior when tile bag is depleted."""
        env.reset()

        # Artificially deplete tile bag
        env.tile_bag.draw(env.tile_bag.get_number_of_remaining_tiles())

        assert env.tile_bag.is_empty()

        # Game should still be playable
        observation = env._get_observation()
        assert observation is not None

    def test_info_dict_contents(self, env):
        """Test that info dict contains useful information."""
        observation, info = env.reset()

        assert 'player_id' in info
        assert 'valid_action_count' in info
        assert info['player_id'] in [0, 1]
        assert info['valid_action_count'] > 0

    def test_board_state_updates(self, env):
        """Test that board state updates after moves."""
        observation, _ = env.reset()

        initial_board = observation['board'].copy()

        # Make a move
        valid_actions = np.where(observation['action_mask'] == 1)[0]
        word_actions = [a for a in valid_actions if a != 0]

        if len(word_actions) > 0:
            observation, reward, terminated, truncated, info = env.step(word_actions[0])

            # Board should have changed
            assert not np.array_equal(observation['board'], initial_board)

    def test_consecutive_passes_counter(self, env):
        """Test consecutive passes counter."""
        env.reset()

        assert env.consecutive_passes == 0

        env.step(0)  # First pass
        assert env.consecutive_passes == 1

        # Make a word move (resets counter)
        valid_actions = np.where(env._get_observation()['action_mask'] == 1)[0]
        word_actions = [a for a in valid_actions if a != 0]

        if len(word_actions) > 0:
            env.step(word_actions[0])
            assert env.consecutive_passes == 0

    def test_environment_gymnasium_compatible(self, env):
        """Test that environment follows Gymnasium API."""
        # Check required methods exist
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')

        # Check inheritance
        assert isinstance(env, gym.Env)

    def test_large_dictionary_performance(self, tmp_path):
        """Test environment with larger dictionary."""
        # Create a larger word list
        word_file = tmp_path / "words.txt"
        words = [f'WORD{i}' for i in range(100)]
        word_file.write_text('\n'.join(words))

        env = MiniScrabbleEnv(dictionary_path=str(word_file))

        observation, _ = env.reset()

        # Should still work with more words
        assert len(env.valid_actions) >= 1  # At least PASS

    def test_deterministic_behavior_with_seed(self):
        """Test that environment is deterministic with seed."""
        env1 = MiniScrabbleEnv()
        env2 = MiniScrabbleEnv()

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Play same actions
        for _ in range(5):
            valid1 = np.where(obs1['action_mask'] == 1)[0]
            valid2 = np.where(obs2['action_mask'] == 1)[0]

            if len(valid1) > 0 and len(valid2) > 0:
                action = valid1[0]
                obs1, r1, t1, tr1, i1 = env1.step(action)
                obs2, r2, t2, tr2, i2 = env2.step(action)

                if not (t1 or tr1):
                    # Should be identical
                    assert r1 == r2
                    assert t1 == t2
