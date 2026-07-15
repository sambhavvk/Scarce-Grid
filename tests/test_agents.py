import os
import sys
import numpy as np
import unittest

# Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the agents and environment to test
# BUG 1 FIX: Use correct class name ScarceGridEnv
from src.environment import ScarceGridEnv
# BUG 2 FIX: Import QLearningAgent (now exists in agents.py)
from src.agents import QLearningAgent, MultiAgentQLearning

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        """
        Set up the environment and agent before each test
        """
        self.env = ScarceGridEnv(grid_size=5)
        self.agent = QLearningAgent(
            action_space=self.env.action_space,
            observation_space=self.env.observation_space
        )
    
    def test_agent_initialization(self):
        """
        Test Q-Learning agent initialization
        """
        # Check if agent is created correctly
        self.assertIsInstance(self.agent, QLearningAgent)
        
        # Check default hyperparameters
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.95)
        self.assertGreaterEqual(self.agent.exploration_rate, 0)
        self.assertLessEqual(self.agent.exploration_rate, 1)
    
    def test_action_selection(self):
        """
        Test agent action selection mechanism
        """
        # Reset environment
        state, _ = self.env.reset()
        
        # Choose action
        action = self.agent.choose_action(state)
        
        # Verify action is valid
        self.assertIn(action, range(self.env.action_space.n))
    
    def test_q_value_initialization(self):
        """
        Test Q-value initialization for new states
        """
        # Reset environment
        state, _ = self.env.reset()
        
        # Get state key
        state_key = self.agent.get_state_key(state)
        
        # Initialize Q-values
        self.agent.initialize_q_value(state)
        
        # Check Q-table entry
        self.assertIn(state_key, self.agent.q_table)
        self.assertEqual(
            len(self.agent.q_table[state_key]), 
            self.env.action_space.n
        )
        self.assertTrue(
            np.all(self.agent.q_table[state_key] == 0)
        )
    
    def test_q_learning_update(self):
        """
        Test Q-Learning update mechanism
        """
        # Reset environment
        state, _ = self.env.reset()
        
        # Force exploitation to get predictable action
        self.agent.exploration_rate = 0.0
        action = self.agent.choose_action(state)
        
        # Simulate step
        next_state, reward, done, _, _ = self.env.step([action, action])
        
        # Manually set a known Q-value so the update is detectable
        state_key = self.agent.get_state_key(state)
        self.agent.q_table[state_key][action] = 1.0
        initial_q_value = 1.0
        
        # Use a known non-zero reward to ensure the update changes the value
        self.agent.update(state, action, 5.0, next_state, done)
        
        # Check Q-value changed
        updated_q_value = self.agent.q_table[state_key][action]
        self.assertNotEqual(initial_q_value, updated_q_value)
    
    def test_exploration_decay(self):
        """
        Test exploration rate decay
        """
        # Store initial exploration rate
        initial_exploration_rate = self.agent.exploration_rate
        
        # Decay exploration
        self.agent.decay_exploration()
        
        # Check exploration rate
        self.assertLess(
            self.agent.exploration_rate, 
            initial_exploration_rate
        )
        self.assertGreaterEqual(
            self.agent.exploration_rate, 
            self.agent.min_exploration_rate
        )

class TestMultiAgentQLearning(unittest.TestCase):
    def setUp(self):
        """
        Set up multi-agent environment
        """
        # BUG 1 FIX: Use correct class name ScarceGridEnv
        self.env = ScarceGridEnv(grid_size=5)
        self.multi_agent_learning = MultiAgentQLearning(
            env=self.env, 
            num_agents=2
        )
    
    def test_multi_agent_initialization(self):
        """
        Test multi-agent system initialization
        """
        # Check number of agents
        self.assertEqual(
            len(self.multi_agent_learning.agents), 
            2
        )
        
        # BUG 2 FIX: Verify each agent is a QLearningAgent
        for agent in self.multi_agent_learning.agents:
            self.assertIsInstance(agent, QLearningAgent)
    
    def test_independent_learning(self):
        """
        Test that agents learn independently
        """
        # BUG 4 FIX: Properly unpack reset() return value
        state, _ = self.env.reset()
        
        # Get initial Q-tables
        initial_q_tables = [
            agent.q_table.copy() 
            for agent in self.multi_agent_learning.agents
        ]
        
        # BUG 18 FIX: Both agents see the same grid state
        actions = [
            agent.choose_action(state) 
            for agent in self.multi_agent_learning.agents
        ]
        
        next_state, rewards, done, _, _ = self.env.step(actions)
        
        # BUG 18 FIX: Both agents see the same grid state for update
        for i, agent in enumerate(self.multi_agent_learning.agents):
            agent.update(
                state, 
                actions[i], 
                rewards[i], 
                next_state, 
                done
            )
        
        # Check that Q-tables have changed
        for i, agent in enumerate(self.multi_agent_learning.agents):
            self.assertNotEqual(
                initial_q_tables[i], 
                agent.q_table, 
                f"Agent {i} Q-table did not update"
            )
    
    def test_multi_agent_training_loop(self):
        """
        Test basic multi-agent training loop
        """
        # BUG 11 FIX: train() now returns per-episode rewards
        agent_rewards = self.multi_agent_learning.train(
            num_episodes=10
        )
        
        # Verify rewards list structure
        self.assertEqual(len(agent_rewards), 2)
        self.assertEqual(len(agent_rewards[0]), 10)
        self.assertEqual(len(agent_rewards[1]), 10)
        
        # Check reward types
        for agent_reward_list in agent_rewards:
            for reward in agent_reward_list:
                self.assertIsInstance(reward, (int, float))

def main():
    """
    Run all tests
    """
    unittest.main()

if __name__ == "__main__":
    main()