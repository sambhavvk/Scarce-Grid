import numpy as np
import random

class QLearningAgent:
    def __init__(self, 
                 action_space, 
                 observation_space, 
                 learning_rate=0.1, 
                 discount_factor=0.95, 
                 exploration_rate=1.0, 
                 min_exploration_rate=0.01,
                 exploration_decay_rate=0.995):
        """
        Initialize Q-Learning Agent with configurable parameters
        
        Args:
            action_space (gym.spaces.Discrete): Action space of the environment
            observation_space (gym.spaces.Box): Observation space of the environment
            learning_rate (float): Rate of learning (alpha)
            discount_factor (float): Future reward discount factor (gamma)
            exploration_rate (float): Initial exploration probability
            min_exploration_rate (float): Minimum exploration probability
            exploration_decay_rate (float): Rate of exploration decay
        """
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        
        # Q-Table initialization
        self.q_table = {}
    
    def get_state_key(self, state):
        """
        Convert state to hashable representation
        
        Args:
            state (np.ndarray): Environment state
        
        Returns:
            tuple: Hashable state representation
        """
        return tuple(state.flatten())
    
    def initialize_q_value(self, state):
        """
        Initialize Q-values for a new state
        
        Args:
            state (np.ndarray): Environment state
        """
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy
        
        Args:
            state (np.ndarray): Current environment state
        
        Returns:
            int: Selected action
        """
        self.initialize_q_value(state)
        state_key = self.get_state_key(state)
        
        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update rule
        
        Args:
            state (np.ndarray): Current state
            action (int): Taken action
            reward (float): Received reward
            next_state (np.ndarray): Next state
            done (bool): Episode termination flag
        """
        self.initialize_q_value(state)
        self.initialize_q_value(next_state)
        
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Q-Value update
        if done:
            max_next_q_value = 0
        else:
            max_next_q_value = np.max(self.q_table[next_state_key])
        
        # Q-Learning update equation
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q_value - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def decay_exploration(self):
        """
        Decay exploration rate
        """
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay_rate
        )
    
    def get_exploration_rate(self):
        """
        Get current exploration rate
        
        Returns:
            float: Current exploration probability
        """
        return self.exploration_rate

class MultiAgentQLearning:
    def __init__(self, env, num_agents=2):
        """
        Multi-Agent Q-Learning Coordinator
        
        Args:
            env (gym.Env): OpenAI Gym compatible environment
            num_agents (int): Number of agents
        """
        self.env = env
        self.agents = [
            QLearningAgent(
                action_space=env.action_space, 
                observation_space=env.observation_space
            ) for _ in range(num_agents)
        ]
    
    def train(self, num_episodes=1000):
        """
        Train multiple agents in the environment
        
        Args:
            num_episodes (int): Number of training episodes
        
        Returns:
            list: Reward history for each agent
        """
        agent_rewards = [[] for _ in self.agents]
        
        for episode in range(num_episodes):
            states = self.env.reset()
            done = False
            
            while not done:
                actions = []
                # Get actions from each agent
                for i, agent in enumerate(self.agents):
                    action = agent.choose_action(states[i])
                    actions.append(action)
                
                # Perform actions in environment
                next_states, rewards, done, _, _ = self.env.step(actions)
                
                # Update each agent
                for i, agent in enumerate(self.agents):
                    agent.update(
                        states[i], 
                        actions[i], 
                        rewards[i], 
                        next_states[i], 
                        done
                    )
                    agent_rewards[i].append(rewards[i])
                    agent.decay_exploration()
                
                states = next_states
        
        return agent_rewards

# Visualization Utility
def plot_learning_curves(agent_rewards):
    """
    Plot learning curves for multiple agents
    
    Args:
        agent_rewards (list): Reward history for each agent
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    for i, rewards in enumerate(agent_rewards):
        plt.plot(
            np.cumsum(rewards), 
            label=f'Agent {i+1} Cumulative Reward'
        )
    
    plt.title('Multi-Agent Learning Dynamics')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

# Experimental Configuration Class
class ExperimentConfig:
    def __init__(self):
        # Hyperparameter grid for experimentation
        self.learning_rates = [0.1, 0.3, 0.5]
        self.discount_factors = [0.9, 0.95, 0.99]
        self.exploration_rates = [0.5, 0.7, 1.0]
    
    def generate_experiments(self):
        """
        Generate experiment configurations
        
        Returns:
            list: Experiment configurations
        """
        experiments = []
        for lr in self.learning_rates:
            for df in self.discount_factors:
                for er in self.exploration_rates:
                    experiments.append({
                        'learning_rate': lr,
                        'discount_factor': df,
                        'exploration_rate': er
                    })
        return experiments