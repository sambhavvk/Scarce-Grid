import numpy as np
import random

# GPU/DQN support is optional - torch may not be installed
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    device = None


class QLearningAgent:
    """
    Tabular Q-Learning Agent using a Q-table (no neural network).
    BUG 2 FIX: Added missing QLearningAgent class that was imported
    by scripts/train.py, scripts/evaluate.py, and tests/test_agents.py.
    """
    def __init__(
        self,
        action_space,
        observation_space,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay=0.995
    ):
        """
        Q-Learning Agent with tabular state-action values.
        
        Args:
            action_space: Gymnasium action space
            observation_space: Gymnasium observation space
            learning_rate (float): Learning rate for Q-value updates
            discount_factor (float): Discount factor for future rewards
            exploration_rate (float): Initial exploration probability
            min_exploration_rate (float): Minimum exploration probability
            exploration_decay (float): Multiplicative decay per episode
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
    
    def get_state_key(self, state):
        """Convert state to hashable key for Q-table lookup"""
        return tuple(state.flatten())
    
    def initialize_q_value(self, state):
        """Initialize Q-values for a state if not already present"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
    
    def choose_action(self, state):
        """Select action using epsilon-greedy strategy"""
        self.initialize_q_value(state)
        state_key = self.get_state_key(state)
        
        if random.random() < self.exploration_rate:
            return self.action_space.sample()
        
        return int(np.argmax(self.q_table[state_key]))
    
    def update(self, state, action, reward, next_state, done):
        """Perform a Q-value update using the Q-learning rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        self.initialize_q_value(state)
        self.initialize_q_value(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_exploration(self):
        """Decay exploration rate after an episode"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def get_exploration_rate(self):
        """Get current exploration rate"""
        return self.exploration_rate


# All torch-dependent classes are guarded by TORCH_AVAILABLE
if TORCH_AVAILABLE:

    class DeepQNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            """Deep Q-Network for GPU-accelerated learning"""
            super(DeepQNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        
        def forward(self, x):
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
            """GPU-Accelerated Q-Learning Agent using Deep Q-Networks"""
            self.state_dim = state_dim
            self.action_dim = action_dim
            
            self.q_network = DeepQNetwork(state_dim, action_dim).to(device)
            self.target_network = DeepQNetwork(state_dim, action_dim).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
        
        def select_action(self, state):
            """Select action using epsilon-greedy strategy"""
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            # BUG 5 FIX: Add batch dimension with unsqueeze(0) for network input
            state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            return q_values.argmax(dim=1).item()
        
        def train_step(self, state, action, reward, next_state, done):
            """Perform a training step"""
            # BUG 5 FIX: Add batch dimension with unsqueeze(0) for network input
            state = torch.FloatTensor(state).flatten().unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).flatten().unsqueeze(0).to(device)
            action = torch.LongTensor([action]).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            done = torch.FloatTensor([done]).to(device)
            
            current_q = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q = self.target_network(next_state).max(dim=1)[0]
            
            target_q = reward + (1 - done) * self.gamma * next_q
            loss = self.loss_fn(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.decay_exploration()
        
        def update_target_network(self):
            """Copy weights from Q-network to target network"""
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        def decay_exploration(self):
            """Decay exploration rate"""
            self.epsilon = max(
                self.epsilon_end, 
                self.epsilon * self.epsilon_decay
            )
        
        def get_exploration_rate(self):
            """Get current exploration rate"""
            return self.epsilon

    class GPUMultiAgentTrainer:
        def __init__(self, env, num_agents=2):
            """Multi-Agent GPU Training using Deep Q-Networks"""
            self.env = env
            state_dim = np.prod(env.observation_space.shape)
            action_dim = env.action_space.n
            
            self.agents = [
                GPUQLearningAgent(
                    state_dim=state_dim, 
                    action_dim=action_dim
                ) for _ in range(num_agents)
            ]
        
        def train(self, num_episodes=1000, update_frequency=10):
            """Train agents on GPU"""
            agent_rewards = [[] for _ in self.agents]
            
            for episode in range(num_episodes):
                state, _ = self.env.reset()
                done = False
                # BUG 11 FIX: Track per-episode rewards
                episode_rewards = [0.0 for _ in self.agents]
                
                while not done:
                    actions = [
                        agent.select_action(state) 
                        for agent in self.agents
                    ]
                    
                    next_state, rewards, done, _, _ = self.env.step(actions)
                    
                    for i, agent in enumerate(self.agents):
                        agent.train_step(
                            state, actions[i], rewards[i], next_state, done
                        )
                        episode_rewards[i] += rewards[i]
                    
                    state = next_state
                
                # BUG 10 FIX: Move target network update OUTSIDE the while loop
                if episode % update_frequency == 0:
                    for agent in self.agents:
                        agent.update_target_network()
                
                # BUG 11 FIX: Append per-episode total reward
                for i in range(len(self.agents)):
                    agent_rewards[i].append(episode_rewards[i])
            
            return agent_rewards


class MultiAgentQLearning:
    def __init__(self, env, num_agents=2, learning_rate=0.1, discount_factor=0.95):
        """
        Multi-Agent Q-Learning Coordinator using tabular Q-learning
        """
        self.env = env
        # BUG 2 FIX: Use QLearningAgent (tabular) instead of GPUQLearningAgent
        self.agents = [
            QLearningAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                learning_rate=learning_rate,
                discount_factor=discount_factor
            ) for _ in range(num_agents)
        ]
    
    def train(self, num_episodes=1000, update_frequency=10):
        """
        Train multiple agents in the environment
        Returns per-episode reward history for each agent.
        """
        agent_rewards = [[] for _ in self.agents]
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            # BUG 11 FIX: Track per-episode rewards, not per-step
            episode_rewards = [0.0 for _ in self.agents]
            
            while not done:
                actions = [
                    agent.choose_action(state) 
                    for agent in self.agents
                ]
                
                next_state, rewards, done, _, _ = self.env.step(actions)
                
                for i, agent in enumerate(self.agents):
                    agent.update(state, actions[i], rewards[i], next_state, done)
                    episode_rewards[i] += rewards[i]
                
                state = next_state
            
            for i, agent in enumerate(self.agents):
                agent.decay_exploration()
                # BUG 11 FIX: Append per-episode total reward
                agent_rewards[i].append(episode_rewards[i])
        
        return agent_rewards


# TensorBoard Logging (requires torch)
if TORCH_AVAILABLE:
    from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir='./runs'):
        if not TORCH_AVAILABLE:
            raise ImportError("TensorBoardLogger requires PyTorch to be installed")
        self.writer = SummaryWriter(log_dir)
    
    def log_episode(self, episode, rewards, losses):
        for i, (reward, loss) in enumerate(zip(rewards, losses)):
            self.writer.add_scalar(f'Agent_{i}/Reward', reward, episode)
            self.writer.add_scalar(f'Agent_{i}/Loss', loss, episode)
    
    def close(self):
        self.writer.close()


def main():
    from src.environment import ScarceGridEnv
    
    print(f"Using device: {device}")
    env = ScarceGridEnv(grid_size=5)
    
    trainer = GPUMultiAgentTrainer(env)
    logger = TensorBoardLogger()
    
    for episode in range(1000):
        agent_rewards = trainer.train(num_episodes=1)
        episode_rewards = [agent_rewards[i][-1] for i in range(len(agent_rewards))]
        num_agents = len(episode_rewards)
        logger.log_episode(episode, episode_rewards, [0.0] * num_agents)
    
    logger.close()

if __name__ == "__main__":
    main()