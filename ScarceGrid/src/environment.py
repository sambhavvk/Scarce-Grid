import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class ScarceGridEnv(gym.Env):
    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size), 
            dtype=np.float32
        )
        
        # Reward configuration
        self.super_reward = 10
        self.minor_rewards = [2, 2, 2]
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self._place_rewards()
        
        # Initial agent positions
        self.agent_positions = {
            'agent1': self._random_position(),
            'agent2': self._random_position()
        }
        
        return self.grid, {}
    
    def _random_position(self):
        return (
            random.randint(0, self.grid_size-1),
            random.randint(0, self.grid_size-1)
        )
    
    def _place_rewards(self):
        # Randomly place rewards
        reward_positions = random.sample(
            list(np.ndindex(self.grid.shape)), 
            len(self.minor_rewards) + 1
        )
        
        # Mark reward positions
        for pos in reward_positions:
            self.grid[pos] = 1