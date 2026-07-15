import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class ScarceGridEnv(gym.Env):
    def __init__(self, grid_size=5, seed_value=None, max_steps=200):
        """
        Initialize the ScarceGrid environment
        
        Args:
            grid_size (int): Size of the grid
            seed_value (int, optional): Random seed for reproducibility
            max_steps (int, optional): Maximum steps before truncation
        """
        super().__init__()
        
        # Environment configuration
        self.grid_size = grid_size
        self.seed_value = seed_value  # Store seed value
        self.max_steps = max_steps
        
        # Set random seed if provided
        if seed_value is not None:
            np.random.seed(seed_value)
            random.seed(seed_value)
        
        # Reward configuration
        self.super_reward = 10.0
        self.minor_rewards = [2.0, 2.0, 2.0]
        
        # Action space (Up, Right, Down, Left)
        self.action_space = spaces.Discrete(4)
        
        # Observation space (BUG 8 FIX: high matches actual max reward value)
        self.observation_space = spaces.Box(
            low=0, 
            high=self.super_reward, 
            shape=(grid_size, grid_size), 
            dtype=np.float32
        )
        
        # Agent positions
        self.agent_positions = [None, None]
        
        # Grid state
        self.grid = None
        self.steps_taken = 0
    
    def reset(self, seed=None):
        """
        Reset the environment to initial state
        
        Args:
            seed (int, optional): Seed for random number generation
        
        Returns:
            tuple: Initial state and additional info
        """
        # BUG 14 FIX: Use explicit None check instead of falsy `or` to handle seed=0
        super().reset(seed=seed if seed is not None else self.seed_value)
        
        # Apply seeding
        effective_seed = seed if seed is not None else self.seed_value
        if effective_seed is not None:
            np.random.seed(effective_seed)
            random.seed(effective_seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.steps_taken = 0
        
        # Place rewards
        self._place_rewards()
        
        # Initialize agent positions
        self._place_agents()
        
        # BUG FIX: Return a copy to prevent external mutation of internal state
        return self.grid.copy(), {}
    
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
        
        # Store reward positions for agent placement
        self._reward_positions = set(reward_positions)
    
    def _place_agents(self):
        """
        Randomly place agents on the grid
        BUG 13 FIX: Ensure agents do not spawn on reward positions
        """
        # Find non-reward positions
        non_reward_positions = [
            (r, c) for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0
        ]
        
        if len(non_reward_positions) >= 2:
            positions = random.sample(non_reward_positions, 2)
        else:
            # Fallback if grid is mostly rewards
            all_positions = [
                (r, c) for r in range(self.grid_size)
                for c in range(self.grid_size)
            ]
            positions = random.sample(all_positions, 2)
        
        self.agent_positions = positions
    
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
        
        # Increment step counter
        self.steps_taken += 1
        
        # Calculate new positions
        new_positions = []
        for i, (action, pos) in enumerate(zip(actions, self.agent_positions)):
            dy, dx = directions[action]
            new_y = max(0, min(self.grid_size-1, pos[0] + dy))
            new_x = max(0, min(self.grid_size-1, pos[1] + dx))
            new_positions.append((new_y, new_x))
        
        # BUG 12 FIX: Collision detection - ensure agents never end up on same cell
        if new_positions[0] == new_positions[1]:
            new_positions[1] = self.agent_positions[1]
        # Re-check: if agent 0 moved to where agent 1 just stayed
        if new_positions[0] == new_positions[1]:
            new_positions[0] = self.agent_positions[0]
        
        # Move agents and collect rewards
        rewards = [0.0, 0.0]
        for i, new_pos in enumerate(new_positions):
            self.agent_positions[i] = new_pos
            rewards[i] = self.grid[new_pos]
            # Remove collected reward
            self.grid[new_pos] = 0
        
        # Check if all rewards are collected (cast to Python bool)
        done = bool(np.sum(self.grid) == 0)
        
        # Check truncation (max steps exceeded)
        truncated = bool(self.steps_taken >= self.max_steps)
        
        # Cast rewards to Python floats
        rewards = [float(r) for r in rewards]
        
        # Return a copy to prevent external mutation
        return self.grid.copy(), rewards, done, truncated, {}
