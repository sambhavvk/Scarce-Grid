import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class ScarseGridEnv(gym.Env):  # Note the spelling: ScarseGridEnv, not ScarceGridEnv
    """
    ScarseGrid Environment for Multi-Agent Reinforcement Learning
    """
    def __init__(self, grid_size=5, seed=None):
        """
        Initialize the ScarseGrid environment
        
        Args:
            grid_size (int): Size of the grid
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__()
        
        # Environment configuration
        self.grid_size = grid_size
        self.seed_value = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reward configuration
        self.super_reward = 10.0
        self.minor_rewards = [2.0, 2.0, 2.0]
        
        # Action space (Up, Right, Down, Left)
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(grid_size, grid_size), 
            dtype=np.float32
        )
        
        # Agent positions
        self.agent_positions = [None, None]
        
        # Grid state
        self.grid = None
    
    def reset(self, seed=None):
        """
        Reset the environment to initial state
        
        Args:
            seed (int, optional): Seed for random number generation
        
        Returns:
            tuple: Initial state and additional info
        """
        # Use provided seed or class seed
        super().reset(seed=seed or self.seed_value)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Place rewards
        self._place_rewards()
        
        # Initialize agent positions
        self._place_agents()
        
        return self.grid, {}
    
    def _place_rewards(self):
        """
        Randomly place rewards on the grid
        """
        # Total number of rewards
        total_rewards = 1 + len(self.minor_rewards)
        
        # Get unique random positions
        reward_positions = random.sample(
            list(np.ndindex(self.grid.shape)), 
            total_rewards
        )
        
        # Place super reward
        self.grid[reward_positions[0]] = self.super_reward
        
        # Place minor rewards
        for pos in reward_positions[1:]:
            self.grid[pos] = random.choice(self.minor_rewards)
    
    def _place_agents(self):
        """
        Randomly place agents on the grid
        """
        # Ensure agents are on different cells
        while True:
            positions = [
                (random.randint(0, self.grid_size-1), 
                 random.randint(0, self.grid_size-1)) 
                for _ in range(2)
            ]
            
            if positions[0] != positions[1]:
                self.agent_positions = positions
                break
    
    def step(self, actions):
        """
        Perform a step in the environment
        
        Args:
            actions (list): Actions for each agent
        
        Returns:
            tuple: Next state, rewards, done flag, truncated, info
        """
        # Validate actions
        if len(actions) != 2:
            raise ValueError("Exactly 2 actions must be provided")
        
        if any(action not in range(4) for action in actions):
            raise ValueError("Invalid action. Must be 0-3.")
        
        # Movement directions (Up, Right, Down, Left)
        directions = [
            (-1, 0),   # Up
            (0, 1),    # Right
            (1, 0),    # Down
            (0, -1)    # Left
        ]
        
        # Move agents
        rewards = [0.0, 0.0]
        for i, (action, pos) in enumerate(zip(actions, self.agent_positions)):
            # Calculate new position
            dy, dx = directions[action]
            new_y = max(0, min(self.grid_size-1, pos[0] + dy))
            new_x = max(0, min(self.grid_size-1, pos[1] + dx))
            
            # Update agent position
            self.agent_positions[i] = (new_y, new_x)
            
            # Check for rewards
            rewards[i] = self.grid[new_y, new_x]
            
            # Remove collected reward
            self.grid[new_y, new_x] = 0
        
        # Check if all rewards are collected
        done = np.sum(self.grid) == 0
        
        return self.grid, rewards, done, False, {}

# Optional: If you want to test the environment directly
if __name__ == "__main__":
    env = ScarseGridEnv()
    state, _ = env.reset()
    print("Initial State:")
    print(state)
    
    # Example step
    actions = [0, 1]  # First agent goes up, second goes right
    next_state, rewards, done, _, _ = env.step(actions)
    print("\nNext State:")
    print(next_state)
    print("\nRewards:", rewards)