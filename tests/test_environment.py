import os
import sys
import numpy as np
import unittest
import gymnasium as gym

# Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# BUG 1 FIX: Import correct class name ScarceGridEnv
from src.environment import ScarceGridEnv

class TestScarceGridEnvironment(unittest.TestCase):
    def setUp(self):
        """
        Set up the environment before each test
        """
        self.env = ScarceGridEnv(grid_size=5)
    
    def test_environment_creation(self):
        """
        Test basic environment initialization
        """
        # Check if environment is created correctly
        self.assertIsInstance(self.env, ScarceGridEnv)
        
        # Verify grid size
        self.assertEqual(self.env.grid_size, 5)
    
    def test_observation_space(self):
        """
        Test observation space characteristics
        """
        # Check observation space type
        self.assertIsInstance(
            self.env.observation_space, 
            gym.spaces.Box
        )
        
        # Verify observation space dimensions
        self.assertEqual(
            self.env.observation_space.shape, 
            (5, 5)
        )
        
        # Check observation space bounds
        self.assertEqual(
            self.env.observation_space.low.min(), 
            0
        )
        # BUG 8 FIX: Observation space high now matches max reward (10.0)
        self.assertEqual(
            self.env.observation_space.high.max(), 
            10.0
        )
    
    def test_action_space(self):
        """
        Test action space characteristics
        """
        # Check action space type
        self.assertIsInstance(
            self.env.action_space, 
            gym.spaces.Discrete
        )
        
        # Verify number of actions
        self.assertEqual(self.env.action_space.n, 4)
    
    def test_reset_method(self):
        """
        Test environment reset functionality
        """
        # Reset the environment
        initial_state, info = self.env.reset()
        
        # Check state type and shape
        self.assertIsInstance(initial_state, np.ndarray)
        self.assertEqual(initial_state.shape, (5, 5))
        
        # Verify state is within bounds
        self.assertTrue(np.all(initial_state >= 0))
        # BUG 9 FIX: Bounds check now uses actual max reward value (10.0)
        self.assertTrue(np.all(initial_state <= 10.0))
    
    def test_reward_placement(self):
        """
        Test reward placement mechanism
        """
        # Reset environment
        initial_state, _ = self.env.reset()
        
        # Count rewards
        reward_count = np.sum(initial_state)
        
        # Verify correct number of rewards
        self.assertEqual(
            reward_count, 
            self.env.super_reward + sum(self.env.minor_rewards)
        )
    
    def test_step_method(self):
        """
        Test environment step functionality
        """
        # Reset environment
        state, _ = self.env.reset()
        
        # Perform a step
        action = self.env.action_space.sample()
        next_state, reward, done, truncated, info = self.env.step([action, action])
        
        # Verify return types
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, list)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        
        # Check state shape and bounds
        self.assertEqual(next_state.shape, (5, 5))
        self.assertTrue(np.all(next_state >= 0))
        # BUG 9 FIX: Bounds check now uses actual max reward value (10.0)
        self.assertTrue(np.all(next_state <= 10.0))
    
    def test_multiple_episodes(self):
        """
        Test multiple episode resets produce varied states
        """
        episode_states = []
        
        for _ in range(5):
            # Reset environment
            state, _ = self.env.reset()
            episode_states.append(state)
        
        # BUG 30 FIX: Check that not all states are identical
        # (rather than asserting every pair is different, which is flaky)
        all_same = all(
            np.array_equal(episode_states[0], s) 
            for s in episode_states[1:]
        )
        self.assertFalse(all_same, "All 5 reset states were identical - extremely unlikely with random placement")
    
    def test_agent_movement_constraints(self):
        """
        Test agent movement within grid boundaries
        """
        # Reset environment
        state, _ = self.env.reset()
        
        # Test all possible actions
        actions = [
            [0, 0],  # Up for both agents
            [1, 1],  # Right for both agents
            [2, 2],  # Down for both agents
            [3, 3]   # Left for both agents
        ]
        
        for action_pair in actions:
            state, rewards, done, truncated, info = self.env.step(action_pair)
            
            # Verify state remains within grid
            self.assertEqual(state.shape, (5, 5))
    
    def test_reward_collection(self):
        """
        Test reward collection mechanism
        """
        # Multiple episode test
        for _ in range(10):
            # Reset environment
            state, _ = self.env.reset()
            
            # Track initial rewards
            initial_rewards = np.sum(state)
            
            # Perform some steps
            for _ in range(5):
                action = self.env.action_space.sample()
                state, rewards, done, _, _ = self.env.step([action, action])
            
            # Verify reward mechanism
            current_rewards = np.sum(state)
            
            # Rewards should not increase arbitrarily
            self.assertLessEqual(current_rewards, initial_rewards)
    
    def test_invalid_actions(self):
        """
        Test handling of invalid actions
        """
        # Reset environment
        self.env.reset()
        
        # Test out-of-bound actions
        with self.assertRaises(ValueError):
            self.env.step([4, 4])  # Action beyond action space
    
    def test_reproducibility(self):
        """
        BUG 7 FIX: Test environment reproducibility with same seed_value parameter
        """
        # First episode with seed_value parameter
        env1 = ScarceGridEnv(grid_size=5, seed_value=42)
        state1, _ = env1.reset()
        
        # Second episode with same seed_value
        env2 = ScarceGridEnv(grid_size=5, seed_value=42)
        state2, _ = env2.reset()
        
        # States should be identical
        np.testing.assert_array_equal(state1, state2)
    
    def test_truncation(self):
        """
        Test that episodes truncate after max_steps
        """
        env = ScarceGridEnv(grid_size=5, max_steps=10)
        state, _ = env.reset()
        
        for step in range(15):
            action = env.action_space.sample()
            state, rewards, done, truncated, info = env.step([action, action])
            if step >= 9:
                # After max_steps, truncated should be True
                self.assertTrue(truncated)
                break
            if done:
                break  # All rewards collected before truncation
    
    def test_collision_detection(self):
        """
        BUG 12 FIX: Test that agents don't occupy the same cell after step
        """
        for _ in range(20):
            state, _ = self.env.reset()
            
            # Both agents move in the same direction repeatedly
            for _ in range(10):
                action = self.env.action_space.sample()
                state, rewards, done, _, _ = self.env.step([action, action])
                
                # Agents should not be on the same cell
                pos0 = self.env.agent_positions[0]
                pos1 = self.env.agent_positions[1]
                self.assertNotEqual(
                    pos0, pos1,
                    f"Agents collided at position {pos0}"
                )
                
                if done:
                    break
    
    def test_agents_not_on_rewards(self):
        """
        BUG 13 FIX: Test that agents don't spawn on reward positions
        """
        for _ in range(20):
            state, _ = self.env.reset()
            
            for i, pos in enumerate(self.env.agent_positions):
                self.assertEqual(
                    state[pos], 0.0,
                    f"Agent {i} spawned on reward at {pos} (value={state[pos]})"
                )

def main():
    """
    Run all tests
    """
    unittest.main()

if __name__ == "__main__":
    main()