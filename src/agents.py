import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MultiAgentQLearning:
    def __init__(self, env, num_agents=2, learning_rate=0.1, discount_factor=0.95):
        """
        Multi-Agent Q-Learning Coordinator
        
        Args:
            env (gym.Env): OpenAI Gym compatible environment
            num_agents (int): Number of agents
            learning_rate (float): Learning rate for agents
            discount_factor (float): Discount factor for future rewards
        """
        self.env = env
        self.agents = [
            GPUQLearningAgent(
                state_dim=np.prod(env.observation_space.shape),
                action_dim=env.action_space.n,
                learning_rate=learning_rate,
                gamma=discount_factor
            ) for _ in range(num_agents)
        ]
    
    def train(self, num_episodes=1000, update_frequency=10):
        """
        Train multiple agents in the environment
        
        Args:
            num_episodes (int): Number of training episodes
            update_frequency (int): Frequency of target network updates
        
        Returns:
            list: Reward history for each agent
        """
        agent_rewards = [[] for _ in self.agents]
        
        for episode in range(num_episodes):
            # Reset environment
            states, _ = self.env.reset()
            done = False
            
            while not done:
                # Get actions from each agent
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(states)
                    actions.append(action)
                
                # Perform actions in environment
                next_states, rewards, done, _, _ = self.env.step(actions)
                
                # Update each agent
                for i, agent in enumerate(self.agents):
                    agent.train_step(
                        states, 
                        actions[i], 
                        rewards[i], 
                        next_states, 
                        done
                    )
                    agent_rewards[i].append(rewards[i])
                
                # Periodic target network update
                if episode % update_frequency == 0:
                    for agent in self.agents:
                        agent.update_target_network()
                
                states = next_states
        
        return agent_rewards
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Deep Q-Network for GPU-accelerated learning
        
        Args:
            input_dim (int): Input dimension (flattened state space)
            output_dim (int): Number of possible actions
        """
        super(DeepQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(device)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input state
        
        Returns:
            torch.Tensor: Q-values for actions
        """
        return self.network(x)

class GPUQLearningAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    ):
        """
        GPU-Accelerated Q-Learning Agent
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Number of possible actions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-Network and Target Network
        self.q_network = DeepQNetwork(state_dim, action_dim).to(device)
        self.target_network = DeepQNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy strategy
        
        Args:
            state (np.ndarray): Current state
        
        Returns:
            int: Selected action
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).flatten().to(device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a training step
        
        Args:
            state (np.ndarray): Current state
            action (int): Taken action
            reward (float): Received reward
            next_state (np.ndarray): Next state
            done (bool): Episode termination flag
        """
        # Convert to tensors
        state = torch.FloatTensor(state).flatten().to(device)
        next_state = torch.FloatTensor(next_state).flatten().to(device)
        action = torch.LongTensor([action]).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)
        
        # Current Q-values
        current_q = self.q_network(state)[action]
        
        # Next Q-values
        with torch.no_grad():
            next_q = self.target_network(next_state).max()
        
        # Compute target Q-value
        target_q = reward + (1 - done) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.decay_exploration()
    
    def update_target_network(self):
        """
        Soft update of the target network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_exploration(self):
        """
        Decay exploration rate
        """
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )
    
    def get_exploration_rate(self):
        """
        Get current exploration rate
        
        Returns:
            float: Current exploration probability
        """
        return self.epsilon

class GPUMultiAgentTrainer:
    def __init__(self, env, num_agents=2):
        """
        Multi-Agent GPU Training
        
        Args:
            env (gym.Env): Environment
            num_agents (int): Number of agents
        """
        self.env = env
        
        # Flatten state dimension
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.n
        
        # Create agents
        self.agents = [
            GPUQLearningAgent(
                state_dim=state_dim, 
                action_dim=action_dim
            ) for _ in range(num_agents)
        ]
    
    def train(self, num_episodes=1000, update_frequency=10):
        """
        Train agents on GPU
        
        Args:
            num_episodes (int): Number of training episodes
            update_frequency (int): How often to update target networks
        
        Returns:
            list: Reward history for each agent
        """
        agent_rewards = [[] for _ in self.agents]
        
        for episode in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            done = False
            
            while not done:
                # Select actions
                actions = [
                    agent.select_action(state) 
                    for agent in self.agents
                ]
                
                # Environment step
                next_state, rewards, done, _, _ = self.env.step(actions)
                
                # Train each agent
                for i, agent in enumerate(self.agents):
                    agent.train_step(
                        state, 
                        actions[i], 
                        rewards[i], 
                        next_state, 
                        done
                    )
                    agent_rewards[i].append(rewards[i])
                
                state = next_state
                
                # Periodic target network update
                if episode % update_frequency == 0:
                    for agent in self.agents:
                        agent.update_target_network()
        
        return agent_rewards

# TensorBoard Logging
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir='./runs'):
        """
        TensorBoard logging for training metrics
        
        Args:
            log_dir (str): Directory to save logs
        """
        self.writer = SummaryWriter(log_dir)
    
    def log_episode(self, episode, rewards, losses):
        """
        Log training metrics
        
        Args:
            episode (int): Current episode
            rewards (list): Rewards for each agent
            losses (list): Training losses
        """
        for i, (reward, loss) in enumerate(zip(rewards, losses)):
            self.writer.add_scalar(f'Agent_{i}/Reward', reward, episode)
            self.writer.add_scalar(f'Agent_{i}/Loss', loss, episode)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

# Example Training Script
def main():
    import gymnasium as gym
    
    # Initialize environment
    env = gym.make('CartPole-v1')  # Example environment
    
    # Create multi-agent trainer
    trainer = GPUMultiAgentTrainer(env)
    
    # TensorBoard logging
    logger = TensorBoardLogger()
    
    # Training loop
    for episode in range(1000):
        rewards = trainer.train(num_episodes=1)
        
        # Log to TensorBoard
        logger.log_episode(episode, rewards[0], [0])  # Placeholder for losses
    
    # Close logger
    logger.close()

if __name__ == "__main__":
    main()