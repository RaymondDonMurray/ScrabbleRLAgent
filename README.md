# Deep RL Scrabble Agent

A reinforcement learning agent that learns to play Scrabble through self-play, starting with a simplified mini-Scrabble and scaling to the full game.

## Project Overview

This project implements a deep reinforcement learning agent for Scrabble, taking a staged approach:
- **Phase 1**: Mini-Scrabble (5×5 board, 3-5 letter words, 5-tile rack)
- **Phase 2**: Progressive scaling (7×7, 11×11 boards with expanded dictionaries)
- **Phase 3**: Full Scrabble (15×15 board, complete dictionary, 7-tile rack)

The goal is to understand how deep RL handles complex, discrete action spaces with partial observability and strategic depth.

## Why Scrabble?

Scrabble presents unique challenges for RL:
- **Massive action space**: Thousands of possible word placements per turn
- **Strategic depth**: Trade-offs between immediate points and board control
- **Partial observability**: Unknown opponent tiles and tile bag composition
- **Vocabulary knowledge**: Must integrate linguistic constraints with RL
- **Long-term planning**: Setup moves, blocking, rack management

## Technical Approach

### State Representation
- Board state: 2D grid encoding (letter values + premium squares)
- Rack: Current available tiles
- Game state: Scores, remaining tiles, turn count
- History: Recent opponent moves (for learning patterns)

### Action Space
- Pre-computed valid word placements using dictionary
- Action masking to exclude invalid moves
- Possibly factored: (position, direction, word) to reduce dimensionality

### Algorithm Candidates
- **PPO** (Proximal Policy Optimization): Handles action masking well, stable
- **AlphaZero-style**: MCTS + neural network for strategic depth
- **Self-play**: Essential for learning strategic elements

### Reward Structure
- Primary: Win/loss outcome (+1/-1)
- Shaping: Per-turn score differential
- Exploration bonuses: Encourage diverse vocabulary usage

## Implementation Phases

### Phase 1: Mini-Scrabble (Current Focus)
- 5×5 board with simplified premium squares
- Filtered dictionary: 1000 most common 3-5 letter words
- 5-tile racks
- Simplified tile distribution
- Goal: Achieve >50% win rate against random baseline

### Phase 2: Intermediate Scaling
- Expand to 7×7, then 11×11 boards
- Increase dictionary size progressively
- Transfer learning from smaller models
- Goal: Maintain learning efficiency as complexity increases

### Phase 3: Full Game
- Standard 15×15 board
- Full OSPD/TWL dictionary (with efficient lookup)
- 7-tile racks, standard tile distribution
- Goal: Competitive play against rule-based agents

## Evaluation Metrics

- Win rate vs baselines (random, greedy, rule-based)
- Average score per game
- Vocabulary diversity (unique words played)
- Board control metrics (premium square usage)
- Training sample efficiency

## Tech Stack

- **Framework**: PyTorch
- **RL Library**: Stable-Baselines3 or custom PPO implementation
- **Dictionary**: NLTK or custom word list
- **Visualization**: Pygame or web interface for watching games
- **Compute**: GPU recommended for Phase 2+

## Project Structure

```
scrabble-rl/
├── envs/
│   ├── scrabble_env.py          # Gymnasium environment
│   ├── board.py                 # Board logic
│   ├── tile_bag.py              # Tile management
│   └── dictionary.py            # Word validation
├── agents/
│   ├── ppo_agent.py             # PPO implementation
│   ├── network.py               # Neural network architectures
│   └── baselines.py             # Random, greedy agents
├── training/
│   ├── self_play.py             # Self-play training loop
│   ├── evaluation.py            # Evaluation against baselines
│   └── utils.py                 # Training utilities
├── config/
│   ├── mini_scrabble.yaml       # Phase 1 config
│   ├── medium_scrabble.yaml     # Phase 2 config
│   └── full_scrabble.yaml       # Phase 3 config
└── tests/
    └── test_env.py              # Environment tests
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train mini-Scrabble agent
python train.py --config config/mini_scrabble.yaml

# Evaluate agent
python evaluate.py --checkpoint checkpoints/best_model.pt

# Watch agent play
python play.py --checkpoint checkpoints/best_model.pt --visualize
```

## Challenges & Considerations

- **Action space explosion**: Need efficient action generation and masking
- **Sparse rewards**: Win/loss is delayed; reward shaping crucial
- **Sample efficiency**: Self-play requires many episodes
- **Strategic depth**: Must learn opening theory, endgame, rack management
- **Vocabulary coverage**: Balance between common and high-value words

## Future Extensions

- Multi-agent tournament play
- Transfer learning across board sizes
- Integration with Scrabble strategy heuristics
- Sim-to-real: Play against human players via online interface
- Explainability: Visualize learned strategies and value estimates

## References

- Silver et al. (2017) - Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- Schulman et al. (2017) - Proximal Policy Optimization Algorithms
- Mnih et al. (2013) - Playing Atari with Deep Reinforcement Learning

## License

MIT