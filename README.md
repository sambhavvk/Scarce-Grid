# Scarce-Grid

A multi-agent reinforcement learning environment where agents compete and cooperate to collect scarce rewards on a grid world. Built on [Gymnasium](https://gymnasium.farama.org/) with support for tabular Q-Learning, Deep Q-Networks (DQN), game-theoretic analysis, and reward shaping experiments.

## Overview

Scarce-Grid places two agents on an `N × N` grid populated with a limited number of rewards — one **super-reward** (10 pts) and three **minor rewards** (2 pts each). Agents move in four directions (up, right, down, left), collect rewards by stepping on them, and the episode ends when all rewards are collected or the maximum step limit is reached. The environment models resource scarcity and supports studying cooperation vs. competition dynamics between agents.

## Project Structure

```
Scarce-Grid/
├── src/
│   ├── __init__.py          # Package exports
│   ├── environment.py       # ScarceGridEnv (Gymnasium environment)
│   ├── agents.py            # QLearningAgent, GPUQLearningAgent, MultiAgentQLearning
│   ├── game_theory.py       # RewardShapingStrategy, GameTheoryAnalyzer, Nash equilibrium
│   └── utils.py             # Logger, MetricsTracker, ConfigManager, seed utilities
├── scripts/
│   ├── train.py             # Training script with multi-agent Q-learning
│   └── evaluate.py          # Evaluation script with performance reporting
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_agent_training_dynamics.ipynb
│   ├── 03_game_theory_analysis.ipynb
│   ├── 04_reward_shaping_experiments.ipynb
│   └── 05_multi_agent_strategies.ipynb
├── tests/
│   ├── test_environment.py  # Environment unit tests
│   └── test_agents.py       # Agent unit tests
├── config.json              # Default configuration
├── requirements.txt         # Python dependencies
├── check_environment.py     # Environment dependency checker
└── test.py                  # CUDA/GPU availability check
```

## Installation

### Prerequisites

- Python 3.9+
- (Optional) NVIDIA GPU with CUDA for GPU-accelerated DQN training

### Setup

```bash
# Clone the repository
git clone https://github.com/sambhavvk/Scarce-Grid.git
cd Scarce-Grid

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python check_environment.py
python test.py               # Check CUDA/GPU availability
```

## Configuration

All training and environment parameters are managed through `config.json`:

```json
{
    "environment": {
        "grid_size": 5,
        "reward_count": 4
    },
    "agent": {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "initial_exploration_rate": 1.0,
        "min_exploration_rate": 0.01,
        "exploration_decay_rate": 0.995
    },
    "training": {
        "episodes": 1000,
        "max_steps_per_episode": 200,
        "num_agents": 2,
        "seed": 42
    },
    "game_theory": {
        "strategies": ["cooperative", "competitive", "mixed"],
        "reward_shaping": {
            "cooperation_bonus": 1.5,
            "competition_penalty": 0.5
        }
    }
}
```

## Usage

### Training

```bash
python scripts/train.py
```

Trains two Q-learning agents in the Scarce-Grid environment using multi-agent reinforcement learning with optional reward shaping. Results and plots are saved to a `results/` directory.

### Evaluation

```bash
python scripts/evaluate.py
```

Evaluates trained agents over 100 episodes, generates performance visualizations, and writes an evaluation report to `results/evaluation_report.txt`.

### Programmatic Usage

```python
from src.environment import ScarceGridEnv
from src.agents import QLearningAgent, MultiAgentQLearning

# Create environment
env = ScarceGridEnv(grid_size=5, max_steps=200)

# Single-agent interaction
state, info = env.reset()
action = env.action_space.sample()
next_state, rewards, done, truncated, info = env.step([action, action])

# Multi-agent training
multi_agent = MultiAgentQLearning(env, num_agents=2, learning_rate=0.1, discount_factor=0.95)
reward_history = multi_agent.train(num_episodes=500)
```

### Game Theory Analysis

```python
from src.game_theory import GameTheoryAnalyzer, RewardShapingStrategy

# Analyze Nash equilibria
analyzer = GameTheoryAnalyzer(num_agents=2)
strategies = ['cooperative', 'competitive', 'mixed']
payoff_matrix = analyzer.create_payoff_matrix(strategies)
nash_equilibria = analyzer.nash_equilibrium(payoff_matrix)
cooperation_score = analyzer.cooperation_potential(payoff_matrix)

# Reward shaping
shaper = RewardShapingStrategy({
    'base_reward': 1.0,
    'cooperation_bonus': 1.5,
    'competition_penalty': 0.5
})
shaped = shaper.shape_reward([agent1_state, agent2_state], base_reward=5.0)
```

## Testing

```bash
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python -m unittest tests.test_environment -v
python -m unittest tests.test_agents -v
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for interactive exploration:

| Notebook | Description |
|---|---|
| `01_environment_exploration.ipynb` | Environment setup, grid visualization, and basic interactions |
| `02_agent_training_dynamics.ipynb` | Training curves, exploration decay, and convergence analysis |
| `03_game_theory_analysis.ipynb` | Payoff matrices, Nash equilibria, and cooperation metrics |
| `04_reward_shaping_experiments.ipynb` | Reward shaping strategies and their impact on agent behavior |
| `05_multi_agent_strategies.ipynb` | Comparative analysis of cooperative, competitive, and mixed strategies |

## Key Features

- **Gymnasium-compatible environment** with configurable grid size, reward structure, and step limits
- **Multi-agent Q-Learning** with tabular Q-learning (CPU) and optional Deep Q-Network (GPU) support
- **Game theory toolkit** — payoff matrices, Nash equilibrium computation, cooperation potential analysis
- **Reward shaping** — dynamic reward modification based on agent interaction patterns (cooperative, competitive, neutral)
- **Collision detection** preventing agents from occupying the same cell
- **Reproducibility** via configurable random seeds
- **Comprehensive test suite** covering environment dynamics, agent behavior, and multi-agent interactions

## License

This project is for academic and research purposes.