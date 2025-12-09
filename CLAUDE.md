# Claude Code Development Guide - Scrabble RL Agent

This document provides context and implementation guidance for working on this project with Claude Code.

## Project Context

The developer has:
- Coursework knowledge in RL and Deep Learning
- Previously implemented Boggle solver using graph search
- No prior experience combining RL + DL (this is their first deep RL project)
- Wants to start with mini-Scrabble (5×5 board) and progressively scale

This is a learning project focused on understanding deep RL fundamentals while tackling a complex discrete action space problem.

## Implementation Philosophy

### Start Simple, Scale Gradually
- Phase 1 is about getting the fundamentals right: environment, self-play loop, basic PPO
- Don't over-engineer early - get something training first, optimize later
- Use simple neural networks initially (MLPs are fine for mini-Scrabble)
- Add complexity only when current version is working

### Code Quality Priorities
1. **Correctness**: Environment must follow Scrabble rules correctly
2. **Debuggability**: Extensive logging, visualization, sanity checks
3. **Reproducibility**: Seed everything, save configs with checkpoints
4. **Modularity**: Clean separation between env, agent, training loop

## Phase 1 Implementation Plan

### Step 1: Mini-Scrabble Environment (Week 1)
Create a Gymnasium-compatible environment that's correct and testable.

**Key Components:**
```python
class MiniScrabbleEnv(gym.Env):
    # State: board (5x5), rack (5 tiles), tile_bag, scores
    # Action: (row, col, direction, word_index) or masked word placements
    # Observation: Dict with board, rack, valid_actions_mask
```

**Critical Implementation Details:**
- **Dictionary loading**: Start with a hardcoded list of ~1000 common 3-5 letter words
  - Later: Load from file, build trie for efficient prefix lookup
- **Action generation**: 
  - For each empty/partially filled board position
  - For each direction (horizontal/vertical)
  - Check which words from dictionary can be formed with current rack
  - Validate against existing board letters
  - This is the computational bottleneck - optimize later
- **Action masking**: Return boolean mask of valid actions with observation
- **Tile bag**: Simplified distribution (fewer of each letter than full Scrabble)
- **Premium squares**: Simple pattern (2-3 double letter, 1 double word)

**Testing Strategy:**
- Unit tests for word placement validation
- Test that all generated actions are actually valid
- Test that invalid actions are properly masked
- Play random vs random for 100 games - should complete without errors

### Step 2: Baseline Agents (Week 1-2)
Implement simple baselines to validate environment and establish benchmarks.

**Random Agent**: Samples uniformly from valid actions
**Greedy Agent**: Picks highest-scoring valid move
**Heuristic Agent**: Greedy + bonus for using all tiles, premium squares

These give you sanity checks: RL agent should at minimum beat random consistently.

### Step 3: PPO Agent (Week 2-3)
Implement or adapt PPO for this environment.

**Network Architecture (Start Simple):**
```python
# Encoder
board_embed = MLP(board_flattened)  # 5x5 = 25 squares -> 128
rack_embed = MLP(rack_one_hot)      # 5 tiles -> 64
state_embed = concat([board_embed, rack_embed]) -> 192

# Actor head (policy)
logits = MLP(state_embed) -> action_space_size
masked_logits = logits.masked_fill(~action_mask, -inf)
action_probs = softmax(masked_logits)

# Critic head (value)
value = MLP(state_embed) -> 1
```

**Key PPO Considerations:**
- **Action masking is critical**: Invalid actions must have -inf logits before softmax
- **Reward normalization**: Scores vary widely; normalize or clip
- **Episode length**: Games can be 10-30 moves; adjust GAE lambda accordingly
- **Learning rate**: Start with 3e-4, tune based on policy entropy
- **Batch size**: 2048 steps works well for discrete actions

**Common Pitfalls:**
- Forgetting to mask actions → agent tries invalid moves → crashes
- Not normalizing rewards → unstable training
- Too high learning rate → policy collapse (all weight on one action)
- Not enough exploration → agent finds suboptimal strategy and gets stuck

### Step 4: Self-Play Training Loop (Week 3-4)
The agent plays against itself or past versions.

**Simple Approach (Start Here):**
- Agent plays against itself (same weights for both players)
- Collect trajectories, update policy with PPO
- Problem: Can get stuck in local optima (both players learn same bad strategy)

**Better Approach (Implement Second):**
- Agent plays against pool of past versions
- Every N updates, save current policy to pool
- Sample opponent from pool (recent + some older versions)
- This prevents "forgetting" and gives more diverse experience

**Training Hyperparameters (Starting Point):**
```yaml
episodes_per_iteration: 100
ppo_epochs: 4
batch_size: 2048
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
entropy_coefficient: 0.01  # Encourage exploration
value_coefficient: 0.5
```

**What to Log:**
- Episode length, total scores (both players)
- Win rate (if playing against fixed opponent)
- Policy entropy (should decrease over time but not collapse)
- Value loss, policy loss
- Average vocabulary size used
- Board coverage metrics

### Step 5: Evaluation & Debugging (Ongoing)
Continuously evaluate to catch issues early.

**Evaluation Protocol:**
- Every 10 training iterations:
  - Play 100 games vs random agent
  - Play 100 games vs greedy agent
  - Log win rates, average scores
- Visualize a few games to spot strange behaviors
- Check policy entropy - if near zero, add entropy bonus

**Debugging Checklist:**
- Is reward scaling reasonable? (Print mean/std of rewards)
- Is action distribution diverse? (Log action entropy)
- Are episodes terminating naturally? (Check for early stops)
- Is value function learning? (Compare predicted vs actual returns)
- Are there any runtime errors during training? (Add try-catch + logging)

## Technical Gotchas

### Action Space Management
The biggest challenge in this project. For a 5×5 board with 1000 words:
- Theoretical actions: ~5×5×2×1000 = 50,000
- Valid actions per turn: Usually 50-200
- Must generate valid actions efficiently and mask invalid ones

**Optimization Ideas (Implement Later):**
- Build trie of dictionary for O(n) prefix lookup
- Cache action generation based on board state hash
- Pre-compute common rack → valid words mapping

### Reward Shaping
Pure win/loss is very sparse. Consider:
```python
# Simple shaping
reward = score_gained_this_turn / 10.0  # Normalize
if game_over:
    reward += 10.0 if won else -10.0

# More sophisticated
reward = (my_score - opp_score) / 50.0  # Score differential
reward += 0.1 * tiles_used / 5  # Bonus for rack turnover
reward += 0.2 * premium_squares_used  # Bonus for strategy
```

Start simple, add shaping only if needed.

### Memory Management
Self-play generates lots of trajectories. Monitor memory usage:
- Clear replay buffers between PPO updates
- Don't keep full episode history in memory
- Use generators for data loading if possible

## Phase 2 & 3 Considerations

When scaling to larger boards:

**Transfer Learning:**
- Load Phase 1 weights, continue training on 7×7
- May need to adjust network architecture (larger board → more features)
- Consider using CNNs instead of MLPs for board encoding

**Computational Scaling:**
- Action space grows quadratically with board size
- May need distributed training or better hardware
- Consider Monte Carlo Tree Search (MCTS) for action selection

**Dictionary Scaling:**
- Full dictionary: ~100,000+ words
- Need very efficient lookup (trie + caching essential)
- May need to filter to common words to keep action space manageable

## Development Workflow

1. **Environment first**: Get the Scrabble game logic perfect before any RL
2. **Baselines**: Validate environment with simple agents
3. **Simple RL**: Get PPO training on the simplest possible setup
4. **Iterate**: Add complexity only when current system is working
5. **Ablations**: When stuck, remove features to isolate the problem

## Recommended Tools

- **Debugging**: Use `gym.make(...).render()` to visualize games
- **Logging**: TensorBoard for training curves, custom logs for game analysis
- **Testing**: pytest for unit tests, especially environment logic
- **Profiling**: cProfile if action generation is slow

## Success Metrics for Phase 1

You'll know Phase 1 is successful when:
- Agent consistently beats random opponent (>90% win rate)
- Agent beats greedy opponent (>60% win rate)
- Training is stable (losses decrease, entropy controlled)
- Agent uses diverse vocabulary (not just one word repeatedly)
- Games look reasonable when visualized (strategic moves, not random)

## Common Questions

**Q: Should I use Stable-Baselines3 or implement PPO from scratch?**
A: Start with SB3 to get something working quickly. Implement from scratch only if you want deeper understanding or need custom modifications.

**Q: How many training episodes before I see results?**
A: For mini-Scrabble, expect 10,000-50,000 episodes to beat random, 100,000+ to beat greedy.

**Q: The action space is huge. How do I handle this?**
A: Action masking is your friend. The agent only sees valid actions, so effective action space is ~50-200 per turn, not 50,000.

**Q: Training is unstable. What should I check?**
A: (1) Reward scaling, (2) Learning rate, (3) Policy entropy (add entropy bonus if collapsed), (4) Action masking (make sure it's working).

## Resources

- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- Gymnasium docs: https://gymnasium.farama.org/
- PPO paper: https://arxiv.org/abs/1707.06347
- Action masking in SB3: Check their MaskablePPO implementation

Good luck! Remember: start simple, validate often, scale gradually.